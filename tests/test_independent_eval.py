"""Tests for Independent Choice Assessment architecture, collation, losses, and engine."""

import pytest
import torch

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data import collate_decision_batch
from yoda.nesy import LTNConstraintLoss
from yoda.training.losses import FocalLoss, FocalMarginLoss, MarginLoss
from yoda.training.trainer import YodaTrainer


class TestIndependentCollation:
    """Verifies that collate_decision_batch unrolls candidates and assigns group_ids."""

    def test_unrolled_candidate_collation(self) -> None:
        batch_samples = [
            {
                "query": "Which route is safest?",
                "state": {"traffic": "heavy"},
                "constraints": ["Route A", "Route B", "Route C", "none: Unused", "none: Unused"],
                "target_idx": 1,
                "num_active": 3,
                "task_scalar": 1.0,
            },
            {
                "query": "Select action",
                "state": {"status": "alert"},
                "constraints": ["Act 1", "Act 2", "none: Unused", "none: Unused", "none: Unused"],
                "target_idx": 0,
                "num_active": 2,
                "task_scalar": 1.0,
            },
        ]

        batch = collate_decision_batch(batch_samples)

        # Total active candidates = 3 + 2 = 5
        assert "candidates" in batch
        assert "candidate_queries" in batch
        assert "candidate_states" in batch
        assert "candidate_labels" in batch
        assert "candidate_group_ids" in batch
        assert "candidate_task_scalars" in batch

        candidates = batch["candidates"]
        assert len(candidates) == 5
        assert candidates == ["Route A", "Route B", "Route C", "Act 1", "Act 2"]

        queries = batch["candidate_queries"]
        assert len(queries) == 5
        assert queries[:3] == ["Which route is safest?"] * 3
        assert queries[3:] == ["Select action"] * 2

        group_ids = batch["candidate_group_ids"]
        assert group_ids.tolist() == [0, 0, 0, 1, 1]

        labels = batch["candidate_labels"]
        # Group 0 target is 1 (Route B), Group 1 target is 0 (Act 1)
        assert labels.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]

        task_scalars = batch["candidate_task_scalars"]
        assert task_scalars.shape == (5, 1)
        assert torch.all(task_scalars == 1.0)


class TestGroupedLosses:
    """Verifies FocalLoss, MarginLoss, and FocalMarginLoss with group_ids."""

    def test_grouped_focal_loss(self) -> None:
        # Group 0: 3 candidates, target is index 1
        # Group 1: 2 candidates, target is index 0
        logits = torch.tensor([1.0, 3.0, 0.5, 2.5, 0.2], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1])

        focal = FocalLoss(gamma=2.0)
        loss = focal(logits, labels, group_ids=group_ids)

        assert loss.dim() == 0
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_grouped_margin_loss(self) -> None:
        # Group 0: Target logit is 5.0, competitors are 1.0, 2.0 -> margin 0.5 satisfied (0 loss)
        # Group 1: Target is 1.0, comp 1.2 -> violation max(0, 0.5 - (1.0 - 1.2)) = 0.7
        logits = torch.tensor([1.0, 5.0, 2.0, 1.0, 1.2], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1])

        margin = MarginLoss(margin=0.5)
        loss = margin(logits, labels, group_ids=group_ids)

        # Group 0 loss = 0.0. Group 1 loss = 0.7. Mean = 0.35.
        assert pytest.approx(loss.item(), rel=1e-4) == 0.35

    def test_grouped_focal_margin_loss(self) -> None:
        logits = torch.tensor([1.0, 3.0, 0.5, 2.5, 0.2], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1])

        hybrid = FocalMarginLoss(gamma=2.0, margin=0.5, margin_weight=0.1)
        total_loss, metrics = hybrid(logits, labels, group_ids=group_ids)

        assert "focal_loss" in metrics
        assert "margin_loss" in metrics
        assert total_loss.item() > 0.0


class TestGroupedLTN:
    """Verifies LTNConstraintLoss with 1D evidence coordinates and group_ids."""

    def test_grouped_ltn_xor_and_nc(self) -> None:
        # Group 0: 3 candidates (Task scalar 1.0 -> XOR evaluated)
        # Group 1: 2 candidates (Task scalar -1.0 -> no XOR evaluated)
        choice_pos = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.2], requires_grad=True)
        choice_neg = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.8], requires_grad=True)
        truth = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.2], requires_grad=True)
        group_ids = torch.tensor([0, 0, 0, 1, 1])
        task_scalars = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0]])

        outputs = {
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
            "truth": truth,
        }

        criterion = LTNConstraintLoss()
        loss = criterion(outputs, task_scalars=task_scalars, group_ids=group_ids)

        assert loss.dim() == 0
        assert loss.item() > 0.0
        loss.backward()
        assert choice_pos.grad is not None


class TestIndependentEngine:
    """Verifies YodaDecisionEngine assessing candidates independently."""

    def test_single_candidate_forward_pass(self) -> None:
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Q1", "Q1", "Q2"]
        states = [{"k": 1}, {"k": 1}, {"k": 2}]
        candidates = ["Candidate A", "Candidate B", "Candidate C"]

        out = engine(queries=queries, states=states, candidates=candidates)

        assert out["logits"].shape == (3,)
        assert out["truth"].shape == (3,)
        assert out["knowledge"].shape == (3,)
        assert out["choice_pos"].shape == (3,)
        assert out["choice_neg"].shape == (3,)
        assert torch.all(out["truth"] >= 0.0) and torch.all(out["truth"] <= 1.0)

    def test_trainer_independent_step(self) -> None:
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )
        trainer = YodaTrainer(model=engine, lr=1e-3, independent_eval=True)

        batch = {
            "queries": ["Q1", "Q2"],
            "states": [{"k": 1}, {"k": 2}],
            "constraints": [["c1", "c2"], ["c3", "c4"]],
            "target_indices": torch.tensor([0, 1]),
            "task_scalars": torch.tensor([[1.0], [1.0]]),
            "candidates": ["c1", "c2", "c3", "c4"],
            "candidate_queries": ["Q1", "Q1", "Q2", "Q2"],
            "candidate_states": [{"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}],
            "candidate_labels": torch.tensor([1.0, 0.0, 0.0, 1.0]),
            "candidate_group_ids": torch.tensor([0, 0, 1, 1]),
            "candidate_task_scalars": torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        }

        loss, metrics = trainer._compute_loss_and_metrics(batch)
        assert loss.item() > 0.0
        assert "focal_loss" in metrics
        assert "margin_loss" in metrics
        assert metrics["total"] == 2.0  # 2 question groups
