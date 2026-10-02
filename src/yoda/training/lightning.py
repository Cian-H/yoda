"""PyTorch Lightning adapter for decoupled training of YodaDecisionEngine."""

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss
from yoda.probabilistic.belnap import BelnapEvidence, FuzzyBelnapLoss

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
    ) -> None:
        """Initializes the Lightning adapter.

        Args:
            model: Pure PyTorch YodaDecisionEngine instance.
            lr: Learning rate for AdamW optimizer.
            weight_decay: Weight decay regularizer.
            belnap_weight: Multiplier for FuzzyBelnapLoss.
            ltn_weight: Multiplier for task-conditioned LTNConstraintLoss.
        """
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.belnap_weight = float(belnap_weight)
        self.ltn_weight = float(ltn_weight)

        self.ce_loss_fn = nn.CrossEntropyLoss()
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
            },
        )

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes joint forward pass and multi-objective loss for train/val/test."""
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

        # 1. Primary Cross-Entropy loss
        ce_loss = self.ce_loss_fn(logits, target_indices)

        # 2. Belnap fuzzy logic regularizer
        if self.belnap_weight > 0.0:
            target_t = torch.zeros_like(logits)
            target_f = torch.ones_like(logits)

            for b_idx in range(batch_size):
                t_idx = target_indices[b_idx].item()
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
            base_loss = ce_loss + self.belnap_weight * belnap_loss
        else:
            belnap_loss = torch.tensor(0.0, device=self.device)
            base_loss = ce_loss

        # 3. Task-Conditioned Multiplicative LTN constraint loss
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
            f"{stage}_ce_loss": ce_loss,
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

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="val")
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, stage="test")
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures AdamW optimizer over model parameters."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
