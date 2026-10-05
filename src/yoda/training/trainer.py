"""Training engine and optimization loops for Yoda System 1 decision models."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss
from yoda.probabilistic.belnap import BelnapEvidence, FuzzyBelnapLoss
from yoda.training.losses import FocalMarginLoss

# logger initialized by loguru import

__all__: list[str] = [
    "CyclicalConstraintScheduler",
    "YodaTrainer",
]


class CyclicalConstraintScheduler:
    """Oscillating cosine schedule for loss constraint weights with warm restarts."""

    def __init__(
        self,
        max_weight: float,
        t0_steps: int,
        t_mult: int = 2,
        start_step: int = 0,
    ) -> None:
        """Initializes CyclicalConstraintScheduler.

        Args:
            max_weight: Peak weight at the top of the cosine cycle.
            t0_steps: Duration (in steps) of the initial cycle.
            t_mult: Multiplier to lengthen cycle duration after each restart.
            start_step: Step at which schedule begins (returns 0.0 prior to this).
        """
        self.max_weight = float(max_weight)
        self.t0_steps = max(1, int(t0_steps))
        self.t_mult = max(1, int(t_mult))
        self.start_step = max(0, int(start_step))

    def get_weight(self, step: int) -> float:
        """Computes scheduled weight at the given step.

        Args:
            step: Current global training step.

        Returns:
            Cosine-annealed constraint weight.
        """
        if step < self.start_step:
            return 0.0
        step -= self.start_step
        curr_t0 = self.t0_steps
        curr_step = step
        while curr_step >= curr_t0:
            curr_step -= curr_t0
            curr_t0 = int(curr_t0 * self.t_mult)
        progress = curr_step / curr_t0
        return self.max_weight * 0.5 * (1.0 - math.cos(math.pi * progress))


class YodaTrainer:
    """Trainer for YodaDecisionEngine combining Focal-Margin, Belnap, and LTN regularization."""

    def __init__(
        self,
        model: YodaDecisionEngine,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        backbone_lr: float | None = None,
        min_lr: float | None = None,
        t0_epochs: int = 2,
        t_mult: int = 2,
        lr_decay: float = 0.75,
        belnap_weight: float = 1.0,
        ltn_weight: float = 0.1,
        assertion_weight: float = 0.0,
        focal_gamma: float = 2.0,
        margin: float = 0.2,
        margin_weight: float = 0.1,
        use_scheduler: bool = True,
        pct_start: float = 0.05,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        independent_eval: bool = False,
        tensorboard_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes YodaTrainer.

        Args:
            model: Top-level YodaDecisionEngine module.
            optimizer: Torch optimizer; defaults to AdamW with specified learning rate.
            lr: Learning rate if default AdamW optimizer is instantiated.
            backbone_lr: Learning rate for text encoder backbone; defaults to lr if None.
            min_lr: Minimum learning rate for scheduler warmup/annealing floor.
            t0_epochs: Number of epochs for initial cosine cycle length T_0.
            t_mult: Cycle lengthening factor for cosine annealing warm restarts.
            lr_decay: Decay factor for base learning rate upon restart.
            belnap_weight: Weight coefficient lambda for FuzzyBelnapLoss regularization
                (default: 1.0).
            ltn_weight: Weight coefficient for LTNConstraintLoss regularization.
            assertion_weight: Weight coefficient for assertion loss regularization.
            focal_gamma: Focusing exponent for FocalLoss.
            margin: Target separation margin for MarginLoss.
            margin_weight: Weight coefficient for MarginLoss term.
            use_scheduler: Whether to automatically instantiate
                CosineAnnealingWarmRestarts during fit().
            pct_start: Deprecated; kept for backward compatibility.
            scheduler: Optional pre-configured PyTorch learning rate scheduler.
            independent_eval: Whether to use independent choice assessment with grouped losses.
            tensorboard_dir: Directory to record TensorBoard event scalars.
            checkpoint_dir: Directory to save per-epoch checkpoints and history.json.
            device: Target execution device.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.backbone_lr = float(backbone_lr) if backbone_lr is not None else self.lr
        self.min_lr = float(min_lr) if min_lr is not None else None
        self.t0_epochs = int(t0_epochs)
        self.t_mult = int(t_mult)
        self.lr_decay = float(lr_decay)
        self.belnap_weight = float(belnap_weight)
        self.ltn_weight = float(ltn_weight)
        self.assertion_weight = float(assertion_weight)
        self.current_ltn_w: float = self.ltn_weight
        self.current_assertion_w: float = self.assertion_weight
        self.focal_gamma = float(focal_gamma)
        self.margin = float(margin)
        self.margin_weight = float(margin_weight)
        self.use_scheduler = bool(use_scheduler)
        self.pct_start = float(pct_start)
        self.scheduler = scheduler
        self.independent_eval = bool(independent_eval)
        self._global_step: int = 0

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            backbone_params: list[torch.nn.Parameter] = []
            head_params: list[torch.nn.Parameter] = []
            for name, param in self.model.named_parameters():
                if "text_encoder.model" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": self.backbone_lr},
                    {"params": head_params, "lr": self.lr},
                ]
            )
        self.focal_margin_loss_fn = FocalMarginLoss(
            gamma=self.focal_gamma,
            margin=self.margin,
            margin_weight=self.margin_weight,
            reduction="mean",
        )
        self.belnap_loss_fn = FuzzyBelnapLoss(reduction="mean")
        self.ltn_criterion = LTNConstraintLoss()

        if tensorboard_dir is not None:
            self.writer: SummaryWriter | None = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info("TensorBoard SummaryWriter initialized at {}", tensorboard_dir)
        else:
            self.writer = None

        if checkpoint_dir is not None:
            self.checkpoint_dir: Path | None = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Per-epoch checkpointing enabled at {}", self.checkpoint_dir)
        else:
            self.checkpoint_dir = None

        logger.debug(
            "YodaTrainer initialized on device={}, lr={}, backbone_lr={}, min_lr={}, "
            "t0_epochs={}, t_mult={}, lr_decay={}, "
            "belnap_w={}, ltn_w={}, assertion_w={}",
            self.device,
            self.lr,
            self.backbone_lr,
            self.min_lr,
            self.t0_epochs,
            self.t_mult,
            self.lr_decay,
            self.belnap_weight,
            self.ltn_weight,
            self.assertion_weight,
        )

    @property
    def ltn_w(self) -> float:
        """Returns base LTN constraint weight."""
        return self.ltn_weight

    @ltn_w.setter
    def ltn_w(self, value: float) -> None:
        self.ltn_weight = float(value)

    @property
    def assertion_w(self) -> float:
        """Returns base assertion loss weight."""
        return self.assertion_weight

    @assertion_w.setter
    def assertion_w(self, value: float) -> None:
        self.assertion_weight = float(value)

    def _compute_independent_loss_and_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes loss and metrics for independent candidate evaluations."""
        c_queries: list[str] = batch["candidate_queries"]
        c_states: list[dict[str, Any]] = batch["candidate_states"]
        candidates: list[str] = batch["candidates"]
        group_ids: torch.Tensor = batch["candidate_group_ids"].to(self.device)
        labels: torch.Tensor = batch["candidate_labels"].to(self.device)
        task_scalars: torch.Tensor | None = batch.get("candidate_task_scalars")
        if task_scalars is not None:
            task_scalars = task_scalars.to(self.device)

        out = self.model(
            queries=c_queries,
            states=c_states,
            candidates=candidates,
            task_scalars=task_scalars,
        )
        logits = out["logits"]

        # Base classification loss: focal_loss logged, margin_loss drives total_loss
        _, loss_parts = self.focal_margin_loss_fn(logits, labels, group_ids=group_ids)
        focal_loss = loss_parts["focal_loss"]
        margin_loss = loss_parts["margin_loss"]

        # Assertion loss: penalize confident incorrect predictions
        unique_groups = torch.unique(group_ids)
        group_assertion_losses: list[torch.Tensor] = []
        for gid in unique_groups:
            mask = group_ids == gid
            g_logits = logits[mask]
            g_target = torch.argmax(labels[mask])
            g_probs = torch.nn.functional.softmax(g_logits, dim=-1)
            max_prob, pred = g_probs.max(dim=-1)
            incorrect_mask = (pred != g_target).float()
            group_assertion_losses.append(incorrect_mask * max_prob)
        if group_assertion_losses:
            assertion_loss = torch.stack(group_assertion_losses).mean()
        else:
            assertion_loss = torch.tensor(0.0, device=self.device)

        # Belnap semantic regularization
        target_evidence = BelnapEvidence(t=labels, f=1.0 - labels)
        pred_evidence = BelnapEvidence(t=out["choice_pos"], f=out["choice_neg"])
        belnap_loss = self.belnap_loss_fn(pred_evidence, target_evidence)

        # Grouped LTN constraint loss
        if self.current_ltn_w > 0.0:
            hierarchy_edges = batch.get("hierarchy_edges")
            ltn_loss = self.ltn_criterion(
                out,
                task_scalars=task_scalars,
                group_ids=group_ids,
                hierarchy_edges=hierarchy_edges,
            )
        else:
            ltn_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            (belnap_loss * self.belnap_weight)
            + (margin_loss * self.margin_weight)
            + (ltn_loss * self.current_ltn_w)
            + (assertion_loss * self.current_assertion_w)
        )

        correct_count = 0
        for gid in unique_groups:
            mask = group_ids == gid
            pred_idx = torch.argmax(logits[mask])
            gt_idx = torch.argmax(labels[mask])
            if pred_idx == gt_idx:
                correct_count += 1

        num_groups = len(unique_groups)
        knowledge_mean = out["knowledge"].mean().item() if "knowledge" in out else 0.0

        metrics = {
            "loss": total_loss.item(),
            "ce_loss": focal_loss.item(),
            "focal_loss": focal_loss.item(),
            "margin_loss": margin_loss.item(),
            "belnap_loss": belnap_loss.item(),
            "ltn_loss": ltn_loss.item(),
            "assertion_loss": assertion_loss.item(),
            "correct": float(correct_count),
            "total": float(num_groups),
            "knowledge_sum": float(knowledge_mean * num_groups),
        }
        return total_loss, metrics

    def _compute_legacy_loss_and_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes loss and metrics for legacy fixed-dimension batches."""
        queries: list[str] = batch["queries"]
        states: list[dict[str, Any]] = batch["states"]
        constraints: list[list[str]] = batch["constraints"]
        target_indices: torch.Tensor = batch["target_indices"].to(self.device)
        task_scalars = batch.get("task_scalars")
        if task_scalars is not None:
            task_scalars = task_scalars.to(self.device)

        active_mask: torch.Tensor | None = batch.get("active_mask")
        if active_mask is not None:
            active_mask = active_mask.to(self.device)

        try:
            out = self.model(
                queries=queries,
                states=states,
                constraints=constraints,
                task_scalars=task_scalars,
                active_mask=active_mask,
            )
        except TypeError:
            out = self.model(
                queries=queries,
                states=states,
                constraints=constraints,
                task_scalars=task_scalars,
            )
        logits = out["logits"]
        batch_size, num_choices = logits.shape

        _, loss_parts = self.focal_margin_loss_fn(
            logits, target_indices, active_mask=active_mask
        )
        focal_loss = loss_parts["focal_loss"]
        margin_loss = loss_parts["margin_loss"]

        # Assertion loss: penalize confident incorrect predictions
        labels = target_indices
        probs = torch.nn.functional.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)
        incorrect_mask = (preds != labels).float()
        assertion_loss = (incorrect_mask * max_probs).mean()

        target_t = torch.zeros(
            (batch_size, num_choices), device=self.device, dtype=torch.float32
        )
        target_f = torch.ones(
            (batch_size, num_choices), device=self.device, dtype=torch.float32
        )

        target_t.scatter_(1, target_indices.unsqueeze(1), 1.0)
        target_f.scatter_(1, target_indices.unsqueeze(1), 0.0)

        for b_idx in range(batch_size):
            for c_idx, c_str in enumerate(constraints[b_idx]):
                if c_idx != target_indices[b_idx].item() and c_str.startswith("none:"):
                    target_f[b_idx, c_idx] = 0.0

        target_evidence = BelnapEvidence(t=target_t, f=target_f)

        if "choice_pos" in out and "choice_neg" in out:
            pred_evidence = BelnapEvidence(t=out["choice_pos"], f=out["choice_neg"])
        else:
            choice_state = BelnapState.from_tk(out["truth"], out["knowledge"])
            pred_evidence = BelnapEvidence(t=choice_state.e_pos, f=choice_state.e_neg)

        belnap_loss = self.belnap_loss_fn(pred_evidence, target_evidence)

        if self.current_ltn_w > 0.0:
            hierarchy_edges = batch.get("hierarchy_edges")
            ltn_loss = self.ltn_criterion(
                out,
                task_scalars=task_scalars,
                active_mask=active_mask,
                hierarchy_edges=hierarchy_edges,
            )
        else:
            ltn_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            (belnap_loss * self.belnap_weight)
            + (margin_loss * self.margin_weight)
            + (ltn_loss * self.current_ltn_w)
            + (assertion_loss * self.current_assertion_w)
        )

        preds = out["choice"] if "choice" in out else torch.argmax(logits, dim=-1)
        correct = (preds == target_indices).sum().item()
        knowledge_mean = out["knowledge"].mean().item() if "knowledge" in out else 0.0

        metrics = {
            "loss": total_loss.item(),
            "ce_loss": focal_loss.item(),
            "focal_loss": focal_loss.item(),
            "margin_loss": margin_loss.item(),
            "belnap_loss": belnap_loss.item(),
            "ltn_loss": ltn_loss.item(),
            "assertion_loss": assertion_loss.item(),
            "correct": float(correct),
            "total": float(batch_size),
            "knowledge_sum": float(knowledge_mean * batch_size),
        }
        return total_loss, metrics

    def _compute_loss_and_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes joint Focal-Margin loss, Belnap loss, LTN loss, and batch accuracy metrics."""
        if self.independent_eval and "candidates" in batch and "candidate_queries" in batch:
            return self._compute_independent_loss_and_metrics(batch)
        return self._compute_legacy_loss_and_metrics(batch)

    def train_epoch(self, dataloader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        """Runs one full training epoch over the dataloader.

        Args:
            dataloader: DataLoader yielding decision batches.

        Returns:
            Dictionary with average loss, component losses, accuracy, and knowledge.
        """
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_focal_loss = 0.0
        total_margin_loss = 0.0
        total_belnap_loss = 0.0
        total_ltn_loss = 0.0
        total_assertion_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        total_knowledge = 0.0

        running_loss = 0.0

        for batch in dataloader:
            current_ltn_w = (
                self.ltn_scheduler.get_weight(self._global_step)
                if hasattr(self, "ltn_scheduler")
                else self.ltn_w
            )
            current_assertion_w = (
                self.assertion_scheduler.get_weight(self._global_step)
                if hasattr(self, "assertion_scheduler")
                else self.assertion_w
            )
            self.current_ltn_w = current_ltn_w
            self.current_assertion_w = current_assertion_w

            if self.writer is not None:
                self.writer.add_scalar("train/ltn_weight", current_ltn_w, self._global_step)
                self.writer.add_scalar(
                    "train/assertion_weight", current_assertion_w, self._global_step
                )

            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss_and_metrics(batch)

            if torch.isnan(loss):
                logger.warning("Loss is NaN; skipping batch")
                continue

            if running_loss > 0.0 and loss.item() > 5.0 * running_loss:
                logger.warning(
                    "Loss spike detected: loss={:.4f}, running_loss={:.4f}",
                    loss.item(),
                    running_loss,
                )

            running_loss = (
                0.9 * running_loss + 0.1 * loss.item() if running_loss > 0.0 else loss.item()
            )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if float(grad_norm) < 1e-7:
                logger.warning("Infinitesimal gradient detected: norm = {:.8f}", float(grad_norm))

            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                logger.warning("NaN gradient detected; skipping optimizer step")
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                if self._global_step > 0 and getattr(self.scheduler, "T_cur", -1) == 0:
                    self.scheduler.base_lrs = [
                        base_lr * self.lr_decay for base_lr in self.scheduler.base_lrs
                    ]
                    for param_group, base_lr in zip(
                        self.optimizer.param_groups, self.scheduler.base_lrs, strict=False
                    ):
                        param_group["lr"] = base_lr
            self._global_step += 1

            b_size = metrics["total"]
            total_loss += metrics["loss"] * b_size
            total_ce_loss += metrics["ce_loss"] * b_size
            total_focal_loss += metrics["focal_loss"] * b_size
            total_margin_loss += metrics["margin_loss"] * b_size
            total_belnap_loss += metrics["belnap_loss"] * b_size
            total_ltn_loss += metrics["ltn_loss"] * b_size
            total_assertion_loss += metrics["assertion_loss"] * b_size
            total_correct += metrics["correct"]
            total_samples += b_size
            total_knowledge += metrics["knowledge_sum"]

        if total_samples == 0.0:
            return {
                "loss": 0.0,
                "ce_loss": 0.0,
                "focal_loss": 0.0,
                "margin_loss": 0.0,
                "belnap_loss": 0.0,
                "ltn_loss": 0.0,
                "assertion_loss": 0.0,
                "accuracy": 0.0,
                "mean_knowledge": 0.0,
            }

        return {
            "loss": total_loss / total_samples,
            "ce_loss": total_ce_loss / total_samples,
            "focal_loss": total_focal_loss / total_samples,
            "margin_loss": total_margin_loss / total_samples,
            "belnap_loss": total_belnap_loss / total_samples,
            "ltn_loss": total_ltn_loss / total_samples,
            "assertion_loss": total_assertion_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "mean_knowledge": total_knowledge / total_samples,
        }

    def evaluate(self, dataloader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        """Evaluates model performance over the dataloader without computing gradients.

        Args:
            dataloader: DataLoader yielding evaluation decision batches.

        Returns:
            Dictionary with eval metrics including loss, components, accuracy, and knowledge.
        """
        self.model.eval()
        self.current_ltn_w = self.ltn_w
        self.current_assertion_w = self.assertion_w
        total_loss = 0.0
        total_ce_loss = 0.0
        total_focal_loss = 0.0
        total_margin_loss = 0.0
        total_belnap_loss = 0.0
        total_ltn_loss = 0.0
        total_assertion_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        total_knowledge = 0.0

        with torch.no_grad():
            for batch in dataloader:
                _, metrics = self._compute_loss_and_metrics(batch)
                b_size = metrics["total"]
                total_loss += metrics["loss"] * b_size
                total_ce_loss += metrics["ce_loss"] * b_size
                total_focal_loss += metrics["focal_loss"] * b_size
                total_margin_loss += metrics["margin_loss"] * b_size
                total_belnap_loss += metrics["belnap_loss"] * b_size
                total_ltn_loss += metrics["ltn_loss"] * b_size
                total_assertion_loss += metrics["assertion_loss"] * b_size
                total_correct += metrics["correct"]
                total_samples += b_size
                total_knowledge += metrics["knowledge_sum"]

        if total_samples == 0.0:
            return {
                "loss": 0.0,
                "ce_loss": 0.0,
                "focal_loss": 0.0,
                "margin_loss": 0.0,
                "belnap_loss": 0.0,
                "ltn_loss": 0.0,
                "assertion_loss": 0.0,
                "accuracy": 0.0,
                "mean_knowledge": 0.0,
            }

        return {
            "loss": total_loss / total_samples,
            "ce_loss": total_ce_loss / total_samples,
            "focal_loss": total_focal_loss / total_samples,
            "margin_loss": total_margin_loss / total_samples,
            "belnap_loss": total_belnap_loss / total_samples,
            "ltn_loss": total_ltn_loss / total_samples,
            "assertion_loss": total_assertion_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "mean_knowledge": total_knowledge / total_samples,
        }

    def fit(
        self,
        train_loader: DataLoader[dict[str, Any]],
        eval_loader: DataLoader[dict[str, Any]] | None = None,
        epochs: int = 2,
    ) -> list[dict[str, float]]:
        """Trains the model across multiple epochs and evaluates validation metrics.

        Args:
            train_loader: DataLoader for training steps.
            eval_loader: Optional DataLoader for validation metrics after each epoch.
            epochs: Number of training epochs to execute.

        Returns:
            List of per-epoch metric dictionaries.
        """
        history: list[dict[str, float]] = []

        if self.use_scheduler and self.scheduler is None and len(train_loader) > 0:
            t_0 = max(1, len(train_loader) * max(1, self.t0_epochs))
            eta_min = self.min_lr if self.min_lr is not None else 0.0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=t_0,
                T_mult=self.t_mult,
                eta_min=eta_min,
            )
            self._global_step = 0
            logger.info(
                "Initialized CosineAnnealingWarmRestarts: T_0={}, T_mult={}, eta_min={}, "
                "lr_decay={:.2f}",
                t_0,
                self.t_mult,
                eta_min,
                self.lr_decay,
            )

        if self.scheduler is not None and hasattr(self.scheduler, "T_0"):
            self.ltn_scheduler = CyclicalConstraintScheduler(
                self.ltn_w,
                t0_steps=self.scheduler.T_0,
                t_mult=self.scheduler.T_mult,
                start_step=0,
            )
            self.assertion_scheduler = CyclicalConstraintScheduler(
                self.assertion_w,
                t0_steps=self.scheduler.T_0,
                t_mult=self.scheduler.T_mult,
                start_step=self.scheduler.T_0,
            )

        best_eval_loss = float("inf")
        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self.train_epoch(train_loader)
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                epoch_record: dict[str, float] = {
                    "epoch": float(epoch),
                    "lr": current_lr,
                    "train_loss": train_metrics["loss"],
                    "train_ce_loss": train_metrics["ce_loss"],
                    "train_focal_loss": train_metrics["focal_loss"],
                    "train_margin_loss": train_metrics["margin_loss"],
                    "train_belnap_loss": train_metrics["belnap_loss"],
                    "train_ltn_loss": train_metrics.get("ltn_loss", 0.0),
                    "train_assertion_loss": train_metrics.get("assertion_loss", 0.0),
                    "train_accuracy": train_metrics["accuracy"],
                }

                eval_str = ""
                if eval_loader is not None:
                    eval_metrics = self.evaluate(eval_loader)
                    epoch_record["eval_loss"] = eval_metrics["loss"]
                    epoch_record["eval_ce_loss"] = eval_metrics["ce_loss"]
                    epoch_record["eval_focal_loss"] = eval_metrics["focal_loss"]
                    epoch_record["eval_margin_loss"] = eval_metrics["margin_loss"]
                    epoch_record["eval_belnap_loss"] = eval_metrics["belnap_loss"]
                    epoch_record["eval_ltn_loss"] = eval_metrics.get("ltn_loss", 0.0)
                    epoch_record["eval_assertion_loss"] = eval_metrics.get("assertion_loss", 0.0)
                    epoch_record["eval_accuracy"] = eval_metrics["accuracy"]
                    epoch_record["eval_mean_knowledge"] = eval_metrics["mean_knowledge"]
                    eval_str = (
                        f" | Eval Loss: {eval_metrics['loss']:.4f} | "
                        f"Eval Acc: {eval_metrics['accuracy'] * 100:.2f}% | "
                        f"Knowledge: {eval_metrics['mean_knowledge']:.4f}"
                    )

                history.append(epoch_record)

                logger.info(
                    "Epoch {}/{} (lr={:.2e}) | "
                    "Train Loss: {:.4f} (CE: {:.4f}, Belnap: {:.4f}, "
                    "LTN: {:.4f}, Assertion: {:.4f}) | "
                    "Train Acc: {:.2f}%{}",
                    epoch,
                    epochs,
                    current_lr,
                    epoch_record["train_loss"],
                    epoch_record["train_ce_loss"],
                    epoch_record["train_belnap_loss"],
                    epoch_record["train_ltn_loss"],
                    epoch_record["train_assertion_loss"],
                    epoch_record["train_accuracy"] * 100,
                    eval_str,
                )

                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar("lr", current_lr, epoch)
                    self.writer.add_scalar("train/loss", epoch_record["train_loss"], epoch)
                    self.writer.add_scalar("train/ce_loss", epoch_record["train_ce_loss"], epoch)
                    self.writer.add_scalar(
                        "train/focal_loss", epoch_record["train_focal_loss"], epoch
                    )
                    self.writer.add_scalar(
                        "train/margin_loss", epoch_record["train_margin_loss"], epoch
                    )
                    self.writer.add_scalar(
                        "train/belnap_loss", epoch_record["train_belnap_loss"], epoch
                    )
                    self.writer.add_scalar("train/ltn_loss", epoch_record["train_ltn_loss"], epoch)
                    self.writer.add_scalar(
                        "train/assertion_loss", epoch_record["train_assertion_loss"], epoch
                    )
                    self.writer.add_scalar("train/accuracy", epoch_record["train_accuracy"], epoch)
                    if eval_loader is not None:
                        self.writer.add_scalar("eval/loss", epoch_record["eval_loss"], epoch)
                        self.writer.add_scalar("eval/ce_loss", epoch_record["eval_ce_loss"], epoch)
                        self.writer.add_scalar(
                            "eval/focal_loss", epoch_record["eval_focal_loss"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/margin_loss", epoch_record["eval_margin_loss"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/belnap_loss", epoch_record["eval_belnap_loss"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/ltn_loss", epoch_record["eval_ltn_loss"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/assertion_loss", epoch_record["eval_assertion_loss"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/accuracy", epoch_record["eval_accuracy"], epoch
                        )
                        self.writer.add_scalar(
                            "eval/mean_knowledge", epoch_record["eval_mean_knowledge"], epoch
                        )
                    self.writer.flush()

                # Per-epoch checkpointing
                if self.checkpoint_dir is not None:
                    latest_payload = {
                        "epoch": epoch,
                        "global_step": self._global_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict()
                        if self.scheduler is not None
                        else None,
                        "epoch_record": epoch_record,
                        "history": history,
                    }
                    torch.save(latest_payload, self.checkpoint_dir / "latest_checkpoint.pt")
                    with (self.checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
                        json.dump(history, f, indent=2)

                    if eval_loader is not None and epoch_record["eval_loss"] < best_eval_loss:
                        best_eval_loss = epoch_record["eval_loss"]
                        torch.save(latest_payload, self.checkpoint_dir / "best_checkpoint.pt")

            if self.writer is not None:
                self.writer.flush()

            return history
        except Exception:
            if self.writer is not None:
                self.writer.flush()
            raise

    def close(self) -> None:
        """Flushes and closes TensorBoard SummaryWriter if active."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
