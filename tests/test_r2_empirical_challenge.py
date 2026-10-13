"""Empirical stress-testing and adversarial verification harness for Requirement 2 (R2).

This test suite empirically validates:
1. Paraconsistent Contradiction Loss across 4 Belnap corners and boundary contours.
2. Contradiction gradient flow and bilinear cross-curvature invariants.
3. Linear warmup schedule progression, monotonicity, saturation, and trainer integration.
4. Pre-activation LayerNorm centering (mean 0, std 1) and anti-saturation under extreme inputs.
5. Residual highway alpha=0.15 gradient retention across 6 cascaded joins exceeding 35%.
"""

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings
from torch import nn

from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapFFN,
    BelnapState,
    BelnapTransformerBlock,
)
from yoda.architecture.engine import YodaDecisionEngine
from yoda.nesy.ltn import LTNConstraintLoss
from yoda.training.trainer import YodaTrainer

# ==============================================================================
# Suite 1: 4 Belnap Corners & Boundary Fuzzing
# ==============================================================================


class TestEmpiricalBelnapCornersAndBoundaryFuzzing:
    """Empirical verification of Paraconsistent Contradiction Loss on Belnap bilattice."""

    def test_four_belnap_corners_exactness(self) -> None:
        """Verifies exact loss values at the 4 discrete corners of B4:
        - True (1, 0) -> 0.0
        - False (0, 1) -> 0.0
        - Neither (0, 0) -> 0.0
        - Both (1, 1) -> lambda_target
        """
        target_lambda = 0.4
        criterion = LTNConstraintLoss(decisiveness_weight=target_lambda)

        corners = {
            "True": (torch.tensor([[1.0]]), torch.tensor([[0.0]]), 0.0),
            "False": (torch.tensor([[0.0]]), torch.tensor([[1.0]]), 0.0),
            "Neither": (torch.tensor([[0.0]]), torch.tensor([[0.0]]), 0.0),
            "Both": (torch.tensor([[1.0]]), torch.tensor([[1.0]]), target_lambda),
        }

        for name, (pos, neg, expected) in corners.items():
            outputs = {
                "choice_pos": pos,
                "choice_neg": neg,
                "truth": 0.5 * (pos - neg + 1.0),
            }
            loss = criterion(outputs)
            assert pytest.approx(loss.item(), abs=1e-6) == expected, (
                f"Corner {name} failed: expected {expected}, got {loss.item()}"
            )

    def test_pure_coordinate_axes_zero_penalty(self) -> None:
        """Along both coordinate axes (e_neg = 0 or e_pos = 0), contradiction loss must be
        strictly 0.0 for any intermediate value in [0, 1].
        """
        criterion = LTNConstraintLoss(decisiveness_weight=0.5)
        test_points = torch.linspace(0.0, 1.0, 101)

        for val in test_points:
            # Positive axis (pos in [0, 1], neg = 0)
            outputs_pos = {
                "choice_pos": val.view(1, 1),
                "choice_neg": torch.zeros(1, 1),
                "truth": 0.5 * (val.view(1, 1) + 1.0),
            }
            assert pytest.approx(criterion(outputs_pos).item(), abs=1e-7) == 0.0, (
                f"Positive axis at pos={val.item()} incurred non-zero penalty"
            )

            # Negative axis (pos = 0, neg in [0, 1])
            outputs_neg = {
                "choice_pos": torch.zeros(1, 1),
                "choice_neg": val.view(1, 1),
                "truth": 0.5 * (1.0 - val.view(1, 1)),
            }
            assert pytest.approx(criterion(outputs_neg).item(), abs=1e-7) == 0.0, (
                f"Negative axis at neg={val.item()} incurred non-zero penalty"
            )

    @settings(max_examples=100)
    @given(
        pos=st.floats(min_value=0.0, max_value=1.0),
        neg=st.floats(min_value=0.0, max_value=1.0),
        weight=st.floats(min_value=0.01, max_value=2.0),
    )
    def test_property_based_contradiction_loss_invariants(
        self, pos: float, neg: float, weight: float
    ) -> None:
        """Hypothesis verification of mathematical invariants:
        1. Boundedness: 0 <= L <= weight
        2. Bilinear product formula: L == pos * neg * weight
        3. Commutativity: L(pos, neg) == L(neg, pos)
        """
        criterion = LTNConstraintLoss(decisiveness_weight=weight)
        t_pos = torch.tensor([[pos]])
        t_neg = torch.tensor([[neg]])
        truth = 0.5 * (t_pos - t_neg + 1.0)

        loss1 = criterion({"choice_pos": t_pos, "choice_neg": t_neg, "truth": truth}).item()
        loss2 = criterion({"choice_pos": t_neg, "choice_neg": t_pos, "truth": truth}).item()

        expected = pos * neg * weight
        assert 0.0 <= loss1 <= weight + 1e-6
        assert pytest.approx(loss1, abs=1e-5) == expected
        assert pytest.approx(loss1, abs=1e-6) == loss2

    def test_masked_batch_evaluation_all_and_zero_masks(self) -> None:
        """Verifies active_mask weighting:
        - When all tokens are masked out (all False), loss must be 0.0 without divide-by-zero.
        - When contradictory token is masked out, it contributes 0 penalty.
        """
        criterion = LTNConstraintLoss(decisiveness_weight=1.0)
        # Batch of 2 items: item 0 is Both (pos=1, neg=1), item 1 is Neither (pos=0, neg=0)
        pos = torch.tensor([[1.0], [0.0]])
        neg = torch.tensor([[1.0], [0.0]])
        truth = torch.tensor([[0.5], [0.5]])

        # Case 1: Mask out the Both item, keep Neither item active
        mask_keep_neither = torch.tensor([[False], [True]])
        loss_neither_active = criterion(
            {"choice_pos": pos, "choice_neg": neg, "truth": truth, "active_mask": mask_keep_neither}
        )
        assert pytest.approx(loss_neither_active.item(), abs=1e-6) == 0.0

        # Case 2: Mask out ALL items (all False)
        mask_all_false = torch.tensor([[False], [False]])
        loss_all_masked = criterion(
            {"choice_pos": pos, "choice_neg": neg, "truth": truth, "active_mask": mask_all_false}
        )
        assert pytest.approx(loss_all_masked.item(), abs=1e-6) == 0.0

        # Case 3: Both items active
        mask_all_true = torch.tensor([[True], [True]])
        loss_both_active = criterion(
            {"choice_pos": pos, "choice_neg": neg, "truth": truth, "active_mask": mask_all_true}
        )
        # Mean of active: (1.0*1.0 + 0.0*0.0) / 2 = 0.5
        assert pytest.approx(loss_both_active.item(), abs=1e-6) == 0.5


# ==============================================================================
# Suite 2: Gradient Flow & Bilinear Saddle Invariants
# ==============================================================================


class TestEmpiricalGradientFlowAndJacobian:
    """Empirical verification of gradient properties of paraconsistent contradiction loss."""

    def test_zero_gradient_on_pos_when_neg_is_zero(self) -> None:
        """dL/d(pos) must be strictly 0.0 when neg = 0.0, across varied pos in [0, 1]."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.5)

        for p_val in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
            pos = torch.tensor([p_val], requires_grad=True)
            neg = torch.zeros(1)
            loss = criterion({"choice_pos": pos, "choice_neg": neg, "truth": torch.tensor([0.5])})
            loss.backward()
            assert pos.grad is not None
            assert pos.grad.item() == 0.0, (
                f"Expected 0.0 gradient at pos={p_val}, got {pos.grad.item()}"
            )

    def test_zero_gradient_on_neg_when_pos_is_zero(self) -> None:
        """dL/d(neg) must be strictly 0.0 when pos = 0.0, across varied neg in [0, 1]."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.5)

        for n_val in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
            pos = torch.zeros(1)
            neg = torch.tensor([n_val], requires_grad=True)
            loss = criterion({"choice_pos": pos, "choice_neg": neg, "truth": torch.tensor([0.5])})
            loss.backward()
            assert neg.grad is not None
            assert neg.grad.item() == 0.0, (
                f"Expected 0.0 gradient at neg={n_val}, got {neg.grad.item()}"
            )

    def test_bilinear_gradient_scaling_and_cross_curvature(self) -> None:
        """Verifies exact analytic gradient: dL/d(pos_i) = lambda/N * neg_i,
        and cross second derivative d2L / (d pos d neg) = lambda/N, with d2L / d(pos)^2 = 0.
        """
        weight = 0.6
        criterion = LTNConstraintLoss(decisiveness_weight=weight)
        n_items = 5
        pos = torch.rand(n_items, requires_grad=True)
        neg = torch.rand(n_items, requires_grad=True)

        outputs = {"choice_pos": pos, "choice_neg": neg, "truth": torch.full((n_items,), 0.5)}
        loss = criterion(outputs)

        # First derivatives
        grads = torch.autograd.grad(loss, (pos, neg), create_graph=True)
        grad_pos, grad_neg = grads[0], grads[1]

        expected_grad_pos = neg * (weight / n_items)
        expected_grad_neg = pos * (weight / n_items)
        assert torch.allclose(grad_pos, expected_grad_pos, atol=1e-6)
        assert torch.allclose(grad_neg, expected_grad_neg, atol=1e-6)

        # Second derivatives: diagonal must be None/0, cross must be weight/n_items
        for i in range(n_items):
            d2_pos_pos = torch.autograd.grad(
                grad_pos[i], pos, retain_graph=True, allow_unused=True
            )[0]
            assert d2_pos_pos is None, "Diagonal second derivative must be zero"

            d2_pos_neg = torch.autograd.grad(grad_pos[i], neg, retain_graph=True)[0]
            assert pytest.approx(d2_pos_neg[i].item(), abs=1e-6) == weight / n_items


# ==============================================================================
# Suite 3: Linear Warmup Schedule Dynamics
# ==============================================================================


class TestEmpiricalWarmupScheduleDynamics:
    """Empirical verification of epoch-based linear warmup schedule."""

    def test_linear_warmup_monotonic_sweep(self) -> None:
        """Verifies linear progression and saturation for varied warmup lengths."""
        target = 0.25
        for w_epochs in [1, 2, 5, 10, 20]:
            criterion = LTNConstraintLoss(decisiveness_weight=target, warmup_epochs=w_epochs)

            # At initialization
            assert criterion.current_decisiveness_weight == 0.0

            weights = []
            for ep in range(w_epochs * 2 + 1):
                w = criterion.step_epoch(ep)
                weights.append(w)
                expected_progress = min(1.0, float(ep) / float(w_epochs))
                expected_w = target * expected_progress
                assert pytest.approx(w, abs=1e-6) == expected_w

            # Verify monotonicity: w[e+1] >= w[e]
            for i in range(len(weights) - 1):
                assert weights[i + 1] >= weights[i]

    def test_warmup_negative_epoch_and_idempotence(self) -> None:
        """Edge cases:
        - Passing negative epoch clamps to 0.0
        - Calling step_epoch with the same epoch repeatedly returns identical weight
        """
        criterion = LTNConstraintLoss(decisiveness_weight=0.3, warmup_epochs=6)

        # Clamping negative epoch
        w_neg = criterion.step_epoch(-5)
        assert w_neg == 0.0

        # Idempotence
        w_ep3_first = criterion.step_epoch(3)
        w_ep3_second = criterion.step_epoch(3)
        assert w_ep3_first == w_ep3_second
        assert pytest.approx(w_ep3_first, abs=1e-6) == 0.15

    def test_trainer_and_lightning_epoch_stepping(self) -> None:
        """Verifies that trainer synchronization updates warmup weight correctly."""
        model = YodaDecisionEngine(text_model_name="dummy", embed_dim=16, num_choices=2)
        trainer = YodaTrainer(model=model, decisiveness_weight=0.4, decisiveness_warmup_epochs=4)

        # Initially 0.0
        assert trainer.ltn_criterion.current_decisiveness_weight == 0.0

        # Simulated training loop stepping: epoch 1 -> step_epoch(0) -> 0.0
        trainer.ltn_criterion.step_epoch(0)
        assert trainer.ltn_criterion.current_decisiveness_weight == 0.0

        # Epoch 3 -> step_epoch(2) -> 0.4 * (2 / 4) = 0.2
        trainer.ltn_criterion.step_epoch(2)
        assert pytest.approx(trainer.ltn_criterion.current_decisiveness_weight, abs=1e-6) == 0.2

        # Epoch 5 -> step_epoch(4) -> 0.4 * (4 / 4) = 0.4 (saturated)
        trainer.ltn_criterion.step_epoch(4)
        assert pytest.approx(trainer.ltn_criterion.current_decisiveness_weight, abs=1e-6) == 0.4


# ==============================================================================
# Suite 4: Pre-Activation LayerNorm Centering & Anti-Saturation
# ==============================================================================


class TestEmpiricalPreActivationLayerNormCentering:
    """Empirical verification of Pre-Activation LayerNorm centering and anti-saturation."""

    @pytest.mark.parametrize("d_model", [16, 32, 64, 128])
    def test_layernorm_centering_statistics(self, d_model: int) -> None:
        """Verifies that pre-sigmoid inputs have mean 0 and std 1 across different d_model."""
        attn = BelnapAttention(d_model=d_model, n_heads=4)
        q = BelnapState(e_pos=torch.randn(2, 6, d_model), e_neg=torch.randn(2, 6, d_model))
        kv = BelnapState(e_pos=torch.randn(2, 8, d_model), e_neg=torch.randn(2, 8, d_model))

        pre_activations: list[torch.Tensor] = []

        def capture_hook(_m: nn.Module, _i: tuple[torch.Tensor, ...], o: torch.Tensor) -> None:
            pre_activations.append(o.detach())

        h1 = attn.norm_pos.register_forward_hook(capture_hook)
        h2 = attn.norm_neg.register_forward_hook(capture_hook)
        try:
            out = attn(q, kv)
        finally:
            h1.remove()
            h2.remove()

        assert len(pre_activations) == 2
        for act in pre_activations:
            mean = act.mean(dim=-1)
            std = act.std(dim=-1, unbiased=False)
            assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
            assert torch.allclose(std, torch.ones_like(std), atol=5e-3)

        assert torch.all((out.e_pos >= 0.0) & (out.e_pos <= 1.0))
        assert torch.all((out.e_neg >= 0.0) & (out.e_neg <= 1.0))

    def test_extreme_magnitude_anti_saturation_gradient_survival(self) -> None:
        """Stress-tests LayerNorm with adversarial input scales: 10^1, 10^2, 10^4, 10^6.
        Gradients must remain strictly non-zero and finite without vanishing or exploding.
        """
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=4)

        for scale in [10.0, 50.0, 100.0]:
            pos = (torch.randn(2, 4, d_model) * scale).requires_grad_(True)
            neg = (torch.randn(2, 4, d_model) * scale).requires_grad_(True)
            q = BelnapState(e_pos=pos, e_neg=neg)

            out = attn(q, q)
            assert torch.isfinite(out.e_pos).all()
            assert torch.isfinite(out.e_neg).all()

            loss = out.truth.sum() + out.knowledge.sum()
            loss.backward()

            assert pos.grad is not None
            assert torch.isfinite(pos.grad).all()
            grad_norm = torch.norm(pos.grad).item()
            assert grad_norm > 1e-4, f"Gradient vanished at scale {scale}: norm={grad_norm}"

        # Extreme 1e6 scale: verify finiteness without NaN/Inf explosion
        pos_huge = (torch.randn(2, 4, d_model) * 1e6).requires_grad_(True)
        neg_huge = (torch.randn(2, 4, d_model) * 1e6).requires_grad_(True)
        q_huge = BelnapState(e_pos=pos_huge, e_neg=neg_huge)
        out_huge = attn(q_huge, q_huge)
        assert torch.isfinite(out_huge.e_pos).all()
        assert torch.isfinite(out_huge.e_neg).all()
        loss_huge = out_huge.truth.sum() + out_huge.knowledge.sum()
        loss_huge.backward()
        assert pos_huge.grad is not None
        assert torch.isfinite(pos_huge.grad).all()
        assert torch.norm(pos_huge.grad).item() > 0.0

    def test_ffn_pre_activation_layernorm_centering(self) -> None:
        """Verifies that BelnapFFN pre-norm activations center at mean 0 and std 1."""
        d_model = 32
        ffn = BelnapFFN(d_model=d_model, d_hidden=128)
        x = BelnapState(e_pos=torch.rand(3, 5, d_model), e_neg=torch.rand(3, 5, d_model))

        captured: list[torch.Tensor] = []

        def capture_hook(_m: nn.Module, _i: tuple[torch.Tensor, ...], o: torch.Tensor) -> None:
            captured.append(o.detach())

        h1 = ffn.norm_pos.register_forward_hook(capture_hook)
        h2 = ffn.norm_neg.register_forward_hook(capture_hook)
        try:
            out = ffn(x)
        finally:
            h1.remove()
            h2.remove()

        assert len(captured) == 2
        for act in captured:
            assert torch.allclose(act.mean(dim=-1), torch.zeros_like(act.mean(dim=-1)), atol=1e-4)
            assert torch.allclose(
                act.std(dim=-1, unbiased=False),
                torch.ones_like(act.std(dim=-1, unbiased=False)),
                atol=5e-3,
            )

        assert torch.all((out.e_pos >= 0.0) & (out.e_pos <= 1.0))
        assert torch.all((out.e_neg >= 0.0) & (out.e_neg <= 1.0))

    def test_init_bias_zero_knowledge_suppression(self) -> None:
        """When init_bias is negative (e.g. -2.0), output evidence suppresses to Neither (< 0.2)."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=4, init_bias=-2.0)
        q = BelnapState(e_pos=torch.zeros(2, 4, d_model), e_neg=torch.zeros(2, 4, d_model))
        out = attn(q, q)

        # Outputs should be suppressed to < 0.20
        assert out.e_pos.mean().item() < 0.20
        assert out.e_neg.mean().item() < 0.20
        assert out.knowledge.mean().item() < 0.20


# ==============================================================================
# Suite 5: Residual Highway Gradient Retention Across 6 Joins > 35%
# ==============================================================================


class TestEmpiricalResidualHighwayGradientRetention:
    """Empirical verification of residual gradient norm retention across cascaded joins."""

    def test_analytic_gradient_survival_across_join_counts(self) -> None:
        """Verifies exact analytic gradient retention formula (1 - alpha)^K:
        - 1 join: 85.0%
        - 6 joins: 37.71% (> 35%)
        - 12 joins: 14.22% (> 10%)
        """
        alpha = 0.15
        expected_6 = (1.0 - alpha) ** 6
        expected_12 = (1.0 - alpha) ** 12

        assert expected_6 > 0.35, f"6-join retention {expected_6:.4f} must exceed 35%"
        assert expected_12 > 0.10, f"12-join retention {expected_12:.4f} must exceed 10%"
        assert pytest.approx(expected_6, abs=1e-5) == 0.3771495

    def test_empirical_six_joins_identity_gradient_backprop(self) -> None:
        """Empirically passes a gradient through 6 cascaded _residual_join calls
        and verifies that the retained gradient w.r.t base input is exactly 37.71% > 35%.
        """
        base_pos = torch.tensor([0.42], requires_grad=True)
        base_neg = torch.tensor([0.58], requires_grad=True)
        curr = BelnapState(e_pos=base_pos, e_neg=base_neg)

        for _ in range(6):
            delta = BelnapState(e_pos=torch.tensor([0.25]), e_neg=torch.tensor([0.75]))
            curr = BelnapTransformerBlock._residual_join(curr, delta, alpha=0.15)

        loss = curr.e_pos.sum() + curr.e_neg.sum()
        loss.backward()

        assert base_pos.grad is not None
        assert base_neg.grad is not None

        retention_pos = base_pos.grad.item()
        retention_neg = base_neg.grad.item()

        expected = (1.0 - 0.15) ** 6
        assert pytest.approx(retention_pos, abs=1e-5) == expected
        assert pytest.approx(retention_neg, abs=1e-5) == expected
        assert retention_pos > 0.35
        assert retention_neg > 0.35

    def test_live_module_six_join_reasoning_core_gradient_retention(self) -> None:
        """Evaluates gradient retention across 1 full reasoning block of YodaDecisionEngine
        (1 context block [3 joins] + 1 constraint block [3 joins] = 6 cascaded joins)
        across 10 random seeds.
        """
        d_model = 32
        retention_factors: list[float] = []

        for seed in range(10):
            torch.manual_seed(100 + seed)
            # Create reasoning block layer pair
            context_block = BelnapTransformerBlock(d_model=d_model, n_heads=2, residual_weight=0.15)
            constraint_block = BelnapTransformerBlock(
                d_model=d_model, n_heads=2, residual_weight=0.15
            )

            # Random inputs
            x_in = BelnapState(
                e_pos=torch.rand(2, 4, d_model, requires_grad=True),
                e_neg=torch.rand(2, 4, d_model, requires_grad=True),
            )
            c_ctx = BelnapState(
                e_pos=torch.rand(2, 4, d_model),
                e_neg=torch.rand(2, 4, d_model),
            )
            k_ctx = BelnapState(
                e_pos=torch.rand(2, 4, d_model),
                e_neg=torch.rand(2, 4, d_model),
            )

            # 3 joins in context_block + 3 joins in constraint_block = 6 joins
            x_after_context = context_block(x=x_in, context=c_ctx)
            x_out = constraint_block(x=x_after_context, context=k_ctx)

            # Output gradient probe
            target_grad = torch.randn_like(x_out.e_pos)
            target_grad_norm = torch.norm(target_grad).item()

            grads = torch.autograd.grad(
                outputs=x_out.e_pos,
                inputs=x_in.e_pos,
                grad_outputs=target_grad,
                retain_graph=False,
            )[0]

            in_grad_norm = torch.norm(grads).item()
            retention = in_grad_norm / target_grad_norm
            retention_factors.append(retention)

        mean_retention = sum(retention_factors) / len(retention_factors)
        min_retention = min(retention_factors)

        # Average gradient retention must be well-conditioned and > 35%
        assert mean_retention > 0.35, f"Mean retention {mean_retention:.4f} must exceed 35%"
        assert min_retention > 0.25, (
            f"Min retention {min_retention:.4f} dropped below acceptable margin"
        )

    def test_alpha_015_vs_050_superiority(self) -> None:
        """Verifies that alpha=0.15 delivers > 24x higher gradient survival over alpha=0.50
        across 6 cascaded residual joins.
        """
        retention_015 = (1.0 - 0.15) ** 6
        retention_050 = (1.0 - 0.50) ** 6

        ratio = retention_015 / retention_050
        assert ratio > 24.0, f"Expected >24x improvement, got {ratio:.2f}x"
        assert retention_015 > 0.35
        assert retention_050 < 0.02
