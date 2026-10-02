"""Training engine and optimization loops for Yoda System 1 decision models."""

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss
from yoda.probabilistic.belnap import BelnapEvidence, FuzzyBelnapLoss
from yoda.training.losses import FocalMarginLoss

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "YodaTrainer",
]


class YodaTrainer:
    """Trainer for YodaDecisionEngine combining Focal-Margin, Belnap, and LTN regularization."""

    def __init__(
        self,
        model: YodaDecisionEngine,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        belnap_weight: float = 0.1,
        ltn_weight: float = 0.1,
        focal_gamma: float = 2.0,
        margin: float = 0.2,
        margin_weight: float = 0.1,
        use_scheduler: bool = True,
        pct_start: float = 0.3,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes YodaTrainer.

        Args:
            model: Top-level YodaDecisionEngine module.
            optimizer: Torch optimizer; defaults to AdamW with specified learning rate.
            lr: Learning rate if default AdamW optimizer is instantiated.
            belnap_weight: Weight coefficient lambda for FuzzyBelnapLoss regularization.
            ltn_weight: Weight coefficient for LTNConstraintLoss regularization.
            focal_gamma: Focusing exponent for FocalLoss.
            margin: Target separation margin for MarginLoss.
            margin_weight: Weight coefficient for MarginLoss term.
            use_scheduler: Whether to automatically instantiate OneCycleLR during fit().
            pct_start: Percentage of training cycle spent warming up learning rate.
            scheduler: Optional pre-configured PyTorch learning rate scheduler.
            device: Target execution device.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.belnap_weight = float(belnap_weight)
        self.ltn_weight = float(ltn_weight)
        self.focal_gamma = float(focal_gamma)
        self.margin = float(margin)
        self.margin_weight = float(margin_weight)
        self.use_scheduler = bool(use_scheduler)
        self.pct_start = float(pct_start)
        self.scheduler = scheduler

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        )
        self.focal_margin_loss_fn = FocalMarginLoss(
            gamma=self.focal_gamma,
            margin=self.margin,
            margin_weight=self.margin_weight,
            reduction="mean",
        )
        self.belnap_loss_fn = FuzzyBelnapLoss(reduction="mean")
        self.ltn_criterion = LTNConstraintLoss()

        logger.debug(
            "training.trainer.init",
            extra={
                "device": str(self.device),
                "belnap_weight": self.belnap_weight,
                "ltn_weight": self.ltn_weight,
                "focal_gamma": self.focal_gamma,
                "margin": self.margin,
                "margin_weight": self.margin_weight,
                "use_scheduler": self.use_scheduler,
                "lr": self.lr,
            },
        )

    def _compute_loss_and_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes joint Focal-Margin loss, Belnap loss, LTN loss, and batch accuracy metrics."""
        queries: list[str] = batch["queries"]
        states: list[dict[str, Any]] = batch["states"]
        constraints: list[list[str]] = batch["constraints"]
        target_indices: torch.Tensor = batch["target_indices"].to(self.device)
        task_scalars: torch.Tensor | None = batch.get("task_scalars")
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

        # 1. Base classification loss: Focal-Margin hybrid
        base_cls_loss, loss_parts = self.focal_margin_loss_fn(
            logits, target_indices, active_mask=active_mask
        )
        focal_loss = loss_parts["focal_loss"]
        margin_loss = loss_parts["margin_loss"]

        # 2. Belnap semantic regularization
        if self.belnap_weight > 0.0:
            target_t = torch.zeros(
                (batch_size, num_choices), device=self.device, dtype=torch.float32
            )
            target_f = torch.ones(
                (batch_size, num_choices), device=self.device, dtype=torch.float32
            )

            target_t.scatter_(1, target_indices.unsqueeze(1), 1.0)
            target_f.scatter_(1, target_indices.unsqueeze(1), 0.0)

            # Mark padded options as ignorance (t=0.0, f=0.0)
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
            base_loss = base_cls_loss + self.belnap_weight * belnap_loss
        else:
            belnap_loss = torch.tensor(0.0, device=self.device)
            base_loss = base_cls_loss

        # 3. LTN constraint loss (Multiplicative gating against reward hacking)
        if self.ltn_weight > 0.0:
            ltn_loss = self.ltn_criterion(out, task_scalars, active_mask=active_mask)
            total_loss = base_loss * (1.0 + self.ltn_weight * ltn_loss)
        else:
            ltn_loss = torch.tensor(0.0, device=self.device)
            total_loss = base_loss

        preds = out["choice"] if "choice" in out else torch.argmax(logits, dim=-1)
        correct = (preds == target_indices).sum().item()
        knowledge_mean = out["knowledge"].mean().item() if "knowledge" in out else 0.0

        metrics = {
            "loss": total_loss.item(),
            "ce_loss": focal_loss.item(),  # Aliased to focal_loss for backward compat
            "focal_loss": focal_loss.item(),
            "margin_loss": margin_loss.item(),
            "belnap_loss": belnap_loss.item(),
            "ltn_loss": ltn_loss.item(),
            "correct": float(correct),
            "total": float(batch_size),
            "knowledge_sum": float(knowledge_mean * batch_size),
        }
        return total_loss, metrics

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
        total_correct = 0.0
        total_samples = 0.0
        total_knowledge = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss_and_metrics(batch)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            b_size = metrics["total"]
            total_loss += metrics["loss"] * b_size
            total_ce_loss += metrics["ce_loss"] * b_size
            total_focal_loss += metrics["focal_loss"] * b_size
            total_margin_loss += metrics["margin_loss"] * b_size
            total_belnap_loss += metrics["belnap_loss"] * b_size
            total_ltn_loss += metrics["ltn_loss"] * b_size
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
        total_loss = 0.0
        total_ce_loss = 0.0
        total_focal_loss = 0.0
        total_margin_loss = 0.0
        total_belnap_loss = 0.0
        total_ltn_loss = 0.0
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
            total_steps = len(train_loader) * epochs
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
            )
            logger.debug(
                "training.trainer.onecycle_init",
                extra={
                    "total_steps": total_steps,
                    "max_lr": self.lr,
                    "pct_start": self.pct_start,
                },
            )

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
                "train_accuracy": train_metrics["accuracy"],
            }

            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                epoch_record["eval_loss"] = eval_metrics["loss"]
                epoch_record["eval_ce_loss"] = eval_metrics["ce_loss"]
                epoch_record["eval_focal_loss"] = eval_metrics["focal_loss"]
                epoch_record["eval_margin_loss"] = eval_metrics["margin_loss"]
                epoch_record["eval_belnap_loss"] = eval_metrics["belnap_loss"]
                epoch_record["eval_accuracy"] = eval_metrics["accuracy"]
                epoch_record["eval_mean_knowledge"] = eval_metrics["mean_knowledge"]

            history.append(epoch_record)
            logger.info("training.epoch_complete", extra=epoch_record)

        return history
