"""Training engine and optimization loops for Yoda System 1 decision models."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import YodaDecisionEngine
from yoda.probabilistic.belnap import BelnapEvidence, FuzzyBelnapLoss

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "YodaTrainer",
]


class YodaTrainer:
    """Trainer for YodaDecisionEngine combining CrossEntropy with Belnap semantic regularization."""

    def __init__(
        self,
        model: YodaDecisionEngine,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        belnap_weight: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes YodaTrainer.

        Args:
            model: Top-level YodaDecisionEngine module.
            optimizer: Torch optimizer; defaults to AdamW with specified learning rate.
            lr: Learning rate if default AdamW optimizer is instantiated.
            belnap_weight: Weight coefficient lambda for FuzzyBelnapLoss regularization.
            device: Target execution device.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.belnap_weight = float(belnap_weight)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=lr)
        )
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.belnap_loss_fn = FuzzyBelnapLoss(reduction="mean")

        logger.debug(
            "training.trainer.init",
            extra={
                "device": str(self.device),
                "belnap_weight": self.belnap_weight,
                "lr": lr,
            },
        )

    def _compute_loss_and_metrics(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes joint CrossEntropy and Belnap loss along with batch accuracy metrics."""
        queries: list[str] = batch["queries"]
        states: list[dict[str, Any]] = batch["states"]
        constraints: list[list[str]] = batch["constraints"]
        target_indices: torch.Tensor = batch["target_indices"].to(self.device)

        out = self.model(queries=queries, states=states, constraints=constraints)
        logits = out["logits"]
        batch_size, num_choices = logits.shape

        # 1. Calibrated decision Cross-Entropy loss
        ce_loss = self.ce_loss_fn(logits, target_indices)

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
            total_loss = ce_loss + self.belnap_weight * belnap_loss
        else:
            belnap_loss = torch.tensor(0.0, device=self.device)
            total_loss = ce_loss

        preds = out["choice"] if "choice" in out else torch.argmax(logits, dim=-1)
        correct = (preds == target_indices).sum().item()
        knowledge_mean = out["knowledge"].mean().item() if "knowledge" in out else 0.0

        metrics = {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "belnap_loss": belnap_loss.item(),
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
            Dictionary with average loss, ce_loss, belnap_loss, and accuracy.
        """
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_belnap_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        total_knowledge = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss_and_metrics(batch)
            loss.backward()
            self.optimizer.step()

            b_size = metrics["total"]
            total_loss += metrics["loss"] * b_size
            total_ce_loss += metrics["ce_loss"] * b_size
            total_belnap_loss += metrics["belnap_loss"] * b_size
            total_correct += metrics["correct"]
            total_samples += b_size
            total_knowledge += metrics["knowledge_sum"]

        if total_samples == 0.0:
            return {
                "loss": 0.0,
                "ce_loss": 0.0,
                "belnap_loss": 0.0,
                "accuracy": 0.0,
                "mean_knowledge": 0.0,
            }

        return {
            "loss": total_loss / total_samples,
            "ce_loss": total_ce_loss / total_samples,
            "belnap_loss": total_belnap_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "mean_knowledge": total_knowledge / total_samples,
        }

    def evaluate(self, dataloader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        """Evaluates model performance over the dataloader without computing gradients.

        Args:
            dataloader: DataLoader yielding evaluation decision batches.

        Returns:
            Dictionary with eval loss, ce_loss, belnap_loss, accuracy, and mean_knowledge.
        """
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_belnap_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        total_knowledge = 0.0

        with torch.no_grad():
            for batch in dataloader:
                _, metrics = self._compute_loss_and_metrics(batch)
                b_size = metrics["total"]
                total_loss += metrics["loss"] * b_size
                total_ce_loss += metrics["ce_loss"] * b_size
                total_belnap_loss += metrics["belnap_loss"] * b_size
                total_correct += metrics["correct"]
                total_samples += b_size
                total_knowledge += metrics["knowledge_sum"]

        if total_samples == 0.0:
            return {
                "loss": 0.0,
                "ce_loss": 0.0,
                "belnap_loss": 0.0,
                "accuracy": 0.0,
                "mean_knowledge": 0.0,
            }

        return {
            "loss": total_loss / total_samples,
            "ce_loss": total_ce_loss / total_samples,
            "belnap_loss": total_belnap_loss / total_samples,
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

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            epoch_record: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_belnap_loss": train_metrics["belnap_loss"],
                "train_accuracy": train_metrics["accuracy"],
            }

            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                epoch_record["eval_loss"] = eval_metrics["loss"]
                epoch_record["eval_ce_loss"] = eval_metrics["ce_loss"]
                epoch_record["eval_belnap_loss"] = eval_metrics["belnap_loss"]
                epoch_record["eval_accuracy"] = eval_metrics["accuracy"]
                epoch_record["eval_mean_knowledge"] = eval_metrics["mean_knowledge"]

            history.append(epoch_record)
            logger.info("training.epoch_complete", extra=epoch_record)

        return history
