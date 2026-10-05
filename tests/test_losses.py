"""Tests for FocalLoss, MarginLoss, and FocalMarginLoss."""

import pytest
import torch
import torch.nn.functional as F

from yoda.training.losses import (
    FocalLoss,
    FocalMarginLoss,
    MarginLoss,
)


class TestFocalLoss:
    """Verifies behavior of Categorical Focal Loss."""

    def test_gamma_zero_equals_cross_entropy(self) -> None:
        """Verifies that FocalLoss with gamma=0 matches standard CrossEntropyLoss."""
        torch.manual_seed(42)
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.tensor([0, 2, 4, 1])

        focal = FocalLoss(gamma=0.0, reduction="mean")
        loss_focal = focal(logits, targets)
        loss_ce = F.cross_entropy(logits, targets, reduction="mean")

        assert torch.allclose(loss_focal, loss_ce, atol=1e-6)

    def test_gamma_downweights_easy_examples(self) -> None:
        """Verifies that gamma > 0 downweights high-confidence correct predictions."""
        # High confidence for class 0
        logits_easy = torch.tensor([[2.5, -2.5, -2.5]])
        # Low confidence (ambiguous) for class 0
        logits_hard = torch.tensor([[0.5, 0.4, 0.4]])
        targets = torch.tensor([0])

        focal_gamma0 = FocalLoss(gamma=0.0, reduction="none")
        focal_gamma2 = FocalLoss(gamma=2.0, reduction="none")

        loss_easy_g0 = focal_gamma0(logits_easy, targets)
        loss_easy_g2 = focal_gamma2(logits_easy, targets)

        loss_hard_g0 = focal_gamma0(logits_hard, targets)
        loss_hard_g2 = focal_gamma2(logits_hard, targets)

        # Ratio of easy loss to hard loss should be significantly smaller with gamma=2 than gamma=0
        ratio_g0 = (loss_easy_g0 / loss_hard_g0).item()
        ratio_g2 = (loss_easy_g2 / loss_hard_g2).item()
        assert ratio_g2 < ratio_g0

    def test_gradients_flow(self) -> None:
        """Verifies that backward() produces valid non-zero gradients."""
        logits = torch.randn(2, 4, requires_grad=True)
        targets = torch.tensor([1, 3])

        loss = FocalLoss(gamma=2.0)(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert (logits.grad != 0).any()


class TestMarginLoss:
    """Verifies behavior of Active-Mask-Aware Margin Loss."""

    def test_zero_loss_when_margin_satisfied(self) -> None:
        """Verifies that if target logit exceeds all other logits by >= margin, loss is 0."""
        # Target index 1: value 5.0, others 2.0 (diff is 3.0 >= margin 0.5)
        logits = torch.tensor([[2.0, 5.0, 1.0, 0.0]])
        targets = torch.tensor([1])

        margin_loss = MarginLoss(margin=0.5, reduction="mean")
        loss = margin_loss(logits, targets)
        assert loss.item() == 0.0

    def test_positive_loss_on_margin_violation(self) -> None:
        """Verifies loss value when target does not exceed competitors by margin."""
        # Target 0: value 1.0, competitor 1 is 1.2. Margin is 0.5.
        # Violation for comp 1: max(0, 0.5 - (1.0 - 1.2)) = 0.7
        # Competitor 2 is -1.0: diff is 2.0 >= 0.5 -> violation 0
        logits = torch.tensor([[1.0, 1.2, -1.0]])
        targets = torch.tensor([0])

        margin_loss = MarginLoss(margin=0.5, reduction="mean")
        loss = margin_loss(logits, targets)
        # 0.7 / 2 competitors = 0.35
        assert pytest.approx(loss.item(), rel=1e-5) == 0.35

    def test_active_mask_excludes_inactive_choices(self) -> None:
        """Verifies that inactive choices marked in active_mask do not contribute to loss."""
        # Choice 2 is inactive
        logits = torch.tensor([[1.0, 0.8, 10.0]])  # If active, choice 2 would cause huge loss
        targets = torch.tensor([0])
        active_mask = torch.tensor([[True, True, False]])

        margin_loss = MarginLoss(margin=0.5, reduction="mean")
        loss = margin_loss(logits, targets, active_mask=active_mask)

        # Only choice 1 is active competitor: target (1.0) vs comp 1 (0.8)
        # violation = max(0, 0.5 - (1.0 - 0.8)) = 0.3. Normalized by 1 active competitor -> 0.3
        assert pytest.approx(loss.item(), rel=1e-5) == 0.3


class TestFocalMarginLoss:
    """Verifies hybrid Focal-Margin loss combination."""

    def test_hybrid_computation_and_metrics(self) -> None:
        logits = torch.randn(3, 5, requires_grad=True)
        targets = torch.tensor([0, 1, 4])
        active_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        )

        criterion = FocalMarginLoss(gamma=2.0, margin=0.2, margin_weight=0.1)
        total_loss, metrics = criterion(logits, targets, active_mask=active_mask)

        assert "focal_loss" in metrics
        assert "margin_loss" in metrics
        assert total_loss.item() > 0.0

        # Gradient flow
        total_loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
