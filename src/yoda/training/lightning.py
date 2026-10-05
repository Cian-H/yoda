"""PyTorch Lightning adapter for decoupled training of YodaDecisionEngine."""

import logging
from typing import Any

import pytorch_lightning as pl
import torch

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss
from yoda.probabilistic.belnap import BelnapEvidence, FuzzyBelnapLoss
from yoda.training.losses import FocalMarginLoss

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "YodaLightningAdapter",
]


class YodaLightningAdapter(pl.LightningModule):
    """Thin PyTorch Lightning adapter wrapping a pure YodaDecisionEngine.

    The underlying model remains a 100% standalone torch.nn.Module, allowing
    seamless extraction for deployment, export, or inference without Lightning dependencies.
    """

    def __init__(
        self,
        model: YodaDecisionEngine,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        belnap_weight: float = 0.1,
        ltn_weight: float = 0.1,
        focal_gamma: float = 2.0,
        margin: float = 0.2,
        margin_weight: float = 0.1,
        use_scheduler: bool = True,
        pct_start: float = 0.3,
        total_steps: int | None = None,
        independent_eval: bool = False,
    ) -> None:
        """Initializes YodaLightningAdapter.

        Args:
            model: Pure PyTorch YodaDecisionEngine instance.
            lr: Learning rate for AdamW optimizer.
            weight_decay: Weight decay regularizer.
            belnap_weight: Multiplier for FuzzyBelnapLoss.
            ltn_weight: Multiplier for task-conditioned LTNConstraintLoss.
            focal_gamma: Focusing exponent for FocalLoss.
            margin: Target separation margin for MarginLoss.
            margin_weight: Weight coefficient for MarginLoss term.
            use_scheduler: Whether to configure OneCycleLR scheduler.
            pct_start: Fraction of total steps for learning rate warmup.
            total_steps: Total training steps for OneCycleLR; inferred if None.
            independent_eval: Whether to use independent choice assessment with grouped losses.
        """
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.belnap_weight = float(belnap_weight)
        self.ltn_weight = float(ltn_weight)
        self.focal_gamma = float(focal_gamma)
        self.margin = float(margin)
        self.margin_weight = float(margin_weight)
        self.use_scheduler = bool(use_scheduler)
        self.pct_start = float(pct_start)
        self.total_steps = total_steps
        self.independent_eval = bool(independent_eval)

        self.focal_margin_loss_fn = FocalMarginLoss(
            gamma=self.focal_gamma,
            margin=self.margin,
            margin_weight=self.margin_weight,
            reduction="mean",
        )
        self.belnap_loss_fn = FuzzyBelnapLoss(reduction="mean")
        self.ltn_criterion = LTNConstraintLoss()

        self.save_hyperparameters(ignore=["model"])
        logger.debug(
            "training.lightning_adapter.init",
            extra={
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "belnap_weight": self.belnap_weight,
                "ltn_weight": self.ltn_weight,
                "focal_gamma": self.focal_gamma,
                "margin": self.margin,
                "margin_weight": self.margin_weight,
                "use_scheduler": self.use_scheduler,
                "independent_eval": self.independent_eval,
            },
        )

    def _shared_independent_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        base_cls_loss, loss_parts = self.focal_margin_loss_fn(logits, labels, group_ids=group_ids)
        focal_loss = loss_parts["focal_loss"]
        margin_loss = loss_parts["margin_loss"]

        if self.belnap_weight > 0.0:
            target_evidence = BelnapEvidence(t=labels, f=1.0 - labels)
            pred_evidence = BelnapEvidence(t=out["choice_pos"], f=out["choice_neg"])
            belnap_loss = self.belnap_loss_fn(pred_evidence, target_evidence)
            base_loss = base_cls_loss + self.belnap_weight * belnap_loss
        else:
            belnap_loss = torch.tensor(0.0, device=self.device)
            base_loss = base_cls_loss

        if self.ltn_weight > 0.0:
            ltn_loss = self.ltn_criterion(out, task_scalars=task_scalars, group_ids=group_ids)
            total_loss = base_loss * (1.0 + self.ltn_weight * ltn_loss)
        else:
            ltn_loss = torch.tensor(0.0, device=self.device)
            total_loss = base_loss

        unique_groups = torch.unique(group_ids)
        correct_count = 0
        for gid in unique_groups:
            mask = group_ids == gid
            pred_idx = torch.argmax(logits[mask])
            gt_idx = torch.argmax(labels[mask])
            if pred_idx == gt_idx:
                correct_count += 1

        num_groups = len(unique_groups)
        acc = torch.tensor(correct_count / max(1, num_groups), device=self.device)
        knowledge = (
            out["knowledge"].mean() if "knowledge" in out else torch.tensor(0.0, device=self.device)
        )

        metrics = {
            f"{stage}_loss": total_loss,
            f"{stage}_ce_loss": focal_loss,
            f"{stage}_focal_loss": focal_loss,
            f"{stage}_margin_loss": margin_loss,
            f"{stage}_belnap_loss": belnap_loss,
            f"{stage}_ltn_loss": ltn_loss,
            f"{stage}_acc": acc,
            f"{stage}_knowledge": knowledge,
        }

        if self._trainer is not None:
            self.log_dict(
                metrics,
                prog_bar=(stage != "test"),
                batch_size=num_groups,
                on_step=(stage == "train"),
                on_epoch=True,
            )
        return total_loss, metrics

    def _shared_legacy_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes loss and metrics for legacy fixed-dimension batches."""
        queries: list[str] = batch["queries"]
        states: list[dict[str, Any]] = batch["states"]
        constraints: list[list[str]] = batch["constraints"]
        target_indices: torch.Tensor = batch["target_indices"]
        task_scalars: torch.Tensor | None = batch.get("task_scalars")
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
        batch_size = logits.size(0)

        base_cls_loss, loss_parts = self.focal_margin_loss_fn(
            logits, target_indices, active_mask=active_mask
        )
        focal_loss = loss_parts["focal_loss"]
        margin_loss = loss_parts["margin_loss"]

        if self.belnap_weight > 0.0:
            target_t = torch.zeros_like(logits)
            target_f = torch.ones_like(logits)

            for b_idx in range(batch_size):
                t_idx = int(target_indices[b_idx].item())
                target_t[b_idx, t_idx] = 1.0
                target_f[b_idx, t_idx] = 0.0

                for c_idx, c_str in enumerate(constraints[b_idx]):
                    if c_idx != t_idx and c_str.startswith("none:"):
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

        if self.ltn_weight > 0.0:
            ltn_loss = self.ltn_criterion(out, task_scalars, active_mask=active_mask)
            total_loss = base_loss * (1.0 + self.ltn_weight * ltn_loss)
        else:
            ltn_loss = torch.tensor(0.0, device=self.device)
            total_loss = base_loss

        preds = out["choice"] if "choice" in out else torch.argmax(logits, dim=-1)
        acc = (preds == target_indices).float().mean()
        knowledge = (
            out["knowledge"].mean() if "knowledge" in out else torch.tensor(0.0, device=self.device)
        )

        metrics = {
            f"{stage}_loss": total_loss,
            f"{stage}_ce_loss": focal_loss,
            f"{stage}_focal_loss": focal_loss,
            f"{stage}_margin_loss": margin_loss,
            f"{stage}_belnap_loss": belnap_loss,
            f"{stage}_ltn_loss": ltn_loss,
            f"{stage}_acc": acc,
            f"{stage}_knowledge": knowledge,
        }

        if self._trainer is not None:
            self.log_dict(
                metrics,
                prog_bar=(stage != "test"),
                batch_size=batch_size,
                on_step=(stage == "train"),
                on_epoch=True,
            )
        return total_loss, metrics

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes joint forward pass and multi-objective loss for train/val/test."""
        if self.independent_eval and "candidates" in batch and "candidate_queries" in batch:
            return self._shared_independent_step(batch, stage)
        return self._shared_legacy_step(batch, stage)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="val")
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="test")
        return loss

    def configure_optimizers(self) -> Any:
        """Configures AdamW optimizer and optional OneCycleLR scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if not self.use_scheduler:
            return optimizer

        steps = self.total_steps
        if steps is None and self._trainer is not None:
            try:
                est = self._trainer.estimated_stepping_batches
                if est not in (None, float("inf")) and est > 0:
                    steps = int(est)
            except Exception:
                pass

        if steps is not None and steps > 0:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=steps,
                pct_start=self.pct_start,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer
