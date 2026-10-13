"""Unit tests for Paraconsistent Contradiction Loss and Linear Warmup Schedule."""

import pytest
import torch

from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy.ltn import LTNConstraintLoss
from yoda.training.lightning import YodaLightningAdapter
from yoda.training.trainer import YodaTrainer


class TestBelnapCorners:
    """Verifies contradiction loss behavior across all 4 Belnap-Dunn valuation corners."""

    def test_true_consistent_zero_loss(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.ones((2, 3)),
            "choice_neg": torch.zeros((2, 3)),
            "truth": torch.ones((2, 3)),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_false_consistent_zero_loss(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.zeros((2, 3)),
            "choice_neg": torch.ones((2, 3)),
            "truth": torch.zeros((2, 3)),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_neither_unmentioned_zero_loss(self) -> None:
        """Neither (pos=0, neg=0) must incur ZERO loss, preventing false penalties."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.zeros((2, 3)),
            "choice_neg": torch.zeros((2, 3)),
            "truth": torch.full((2, 3), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_both_contradiction_maximum_loss(self) -> None:
        """Both (pos=1, neg=1) incurs maximal penalty equal to current_decisiveness_weight."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        outputs = {
            "choice_pos": torch.ones((2, 3)),
            "choice_neg": torch.ones((2, 3)),
            "truth": torch.full((2, 3), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.2))


class TestContradictionGradients:
    """Verifies gradient properties of paraconsistent Both penalty."""

    def test_zero_gradient_when_conflicting_evidence_absent(self) -> None:
        """When neg=0, d(loss)/d(pos) must be exactly zero."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.2)
        pos = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        neg = torch.zeros(3)
        outputs = {
            "choice_pos": pos,
            "choice_neg": neg,
            "truth": torch.full((3,), 0.5),
        }
        loss = criterion(outputs)
        assert torch.isclose(loss, torch.tensor(0.0))
        loss.backward()
        assert pos.grad is not None
        assert torch.all(pos.grad == 0.0)

    def test_gradient_proportional_to_opposite_polarity(self) -> None:
        """Gradient on pos is strictly proportional to neg."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.5)
        pos = torch.full((4,), 0.5, requires_grad=True)
        neg = torch.tensor([0.1, 0.3, 0.5, 0.9])
        outputs = {
            "choice_pos": pos,
            "choice_neg": neg,
            "truth": torch.full((4,), 0.5),
        }
        loss = criterion(outputs)
        loss.backward()
        assert pos.grad is not None
        # dL/d(pos_i) = neg_i / 4 * 0.5
        expected_grads = neg / 4.0 * 0.5
        assert torch.allclose(pos.grad, expected_grads, atol=1e-6)

    def test_symmetry_commutativity(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.1)
        a = torch.rand(5, 5)
        b = torch.rand(5, 5)
        loss1 = criterion({"choice_pos": a, "choice_neg": b, "truth": torch.zeros(5, 5)})
        loss2 = criterion({"choice_pos": b, "choice_neg": a, "truth": torch.zeros(5, 5)})
        assert torch.isclose(loss1, loss2)


class TestLinearWarmupSchedule:
    """Verifies warmup progression across epochs."""

    def test_warmup_progression(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.2, warmup_epochs=4)
        assert criterion.current_decisiveness_weight == 0.0

        # Epoch 0
        w0 = criterion.step_epoch(0)
        assert w0 == 0.0

        # Epoch 1
        w1 = criterion.step_epoch(1)
        assert pytest.approx(w1) == 0.05

        # Epoch 2
        w2 = criterion.step_epoch(2)
        assert pytest.approx(w2) == 0.10

        # Epoch 3
        w3 = criterion.step_epoch(3)
        assert pytest.approx(w3) == 0.15

        # Epoch 4
        w4 = criterion.step_epoch(4)
        assert pytest.approx(w4) == 0.20

        # Epoch 5 (saturated)
        w5 = criterion.step_epoch(5)
        assert pytest.approx(w5) == 0.20

    def test_auto_increment_step_epoch(self) -> None:
        criterion = LTNConstraintLoss(decisiveness_weight=0.3, warmup_epochs=3)
        assert pytest.approx(criterion.step_epoch()) == 0.1  # epoch 1
        assert pytest.approx(criterion.step_epoch()) == 0.2  # epoch 2
        assert pytest.approx(criterion.step_epoch()) == 0.3  # epoch 3
        assert pytest.approx(criterion.step_epoch()) == 0.3  # epoch 4

    def test_trainer_warmup_initialization(self) -> None:
        model = YodaDecisionEngine(text_model_name="dummy", embed_dim=16, num_choices=2)
        trainer = YodaTrainer(
            model=model,
            decisiveness_weight=0.25,
            decisiveness_warmup_epochs=5,
        )
        assert trainer.decisiveness_weight == 0.25
        assert trainer.decisiveness_warmup_epochs == 5
        assert trainer.ltn_criterion.current_decisiveness_weight == 0.0

    def test_lightning_adapter_warmup_initialization(self) -> None:
        model = YodaDecisionEngine(text_model_name="dummy", embed_dim=16, num_choices=2)
        adapter = YodaLightningAdapter(
            model=model,
            decisiveness_weight=0.3,
            decisiveness_warmup_epochs=4,
        )
        assert adapter.decisiveness_weight == 0.3
        assert adapter.decisiveness_warmup_epochs == 4
        assert adapter.ltn_criterion.current_decisiveness_weight == 0.0
