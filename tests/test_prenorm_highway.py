"""Unit and regression tests for pre-activation LayerNorm and residual gradient highway (R2)."""

import torch
from torch import nn

from experiments.optuna_study_narrow import FIXED_RESIDUAL_WEIGHT
from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapDecisionTransformer,
    BelnapFFN,
    BelnapState,
    BelnapTransformerBlock,
)


class TestPreActivationNormalization:
    """Verifies LayerNorm insertion before output sigmoids in BelnapAttention and BelnapFFN."""

    def test_belnap_attention_prenorm_layers_exist(self) -> None:
        """Verifies that BelnapAttention instantiates LayerNorm for dual evidence projections."""
        attn = BelnapAttention(d_model=32, n_heads=4)
        assert hasattr(attn, "norm_pos")
        assert hasattr(attn, "norm_neg")
        assert isinstance(attn.norm_pos, nn.LayerNorm)
        assert isinstance(attn.norm_neg, nn.LayerNorm)
        assert attn.norm_pos.normalized_shape == (32,)
        assert attn.norm_neg.normalized_shape == (32,)

    def test_belnap_attention_prenorm_bounds_and_distribution(self) -> None:
        """Verifies that pre-activation LayerNorm keeps evidence strictly in [0, 1]
        and centers pre-sigmoid activations around 0 with unit variance.
        """
        torch.manual_seed(42)
        attn = BelnapAttention(d_model=64, n_heads=4)
        q = BelnapState(e_pos=torch.rand(4, 8, 64), e_neg=torch.rand(4, 8, 64))
        kv = BelnapState(e_pos=torch.rand(4, 12, 64), e_neg=torch.rand(4, 12, 64))

        captured_pre_pos: list[torch.Tensor] = []
        captured_pre_neg: list[torch.Tensor] = []

        def hook_pos(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            captured_pre_pos.append(out.detach())

        def hook_neg(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            captured_pre_neg.append(out.detach())

        h1 = attn.norm_pos.register_forward_hook(hook_pos)
        h2 = attn.norm_neg.register_forward_hook(hook_neg)
        try:
            out = attn(q, kv)
        finally:
            h1.remove()
            h2.remove()

        assert len(captured_pre_pos) == 1
        assert len(captured_pre_neg) == 1

        pre_pos = captured_pre_pos[0]
        pre_neg = captured_pre_neg[0]

        # LayerNorm guarantees mean 0 and std 1 across the normalized dimension
        assert torch.allclose(
            pre_pos.mean(dim=-1), torch.zeros_like(pre_pos.mean(dim=-1)), atol=1e-4
        )
        assert torch.allclose(
            pre_neg.mean(dim=-1), torch.zeros_like(pre_neg.mean(dim=-1)), atol=1e-4
        )
        assert torch.allclose(
            pre_pos.std(dim=-1, unbiased=False),
            torch.ones_like(pre_pos.std(dim=-1, unbiased=False)),
            atol=5e-3,
        )
        assert torch.allclose(
            pre_neg.std(dim=-1, unbiased=False),
            torch.ones_like(pre_neg.std(dim=-1, unbiased=False)),
            atol=5e-3,
        )

        # Output bounds
        assert torch.all(out.e_pos >= 0.0) and torch.all(out.e_pos <= 1.0)
        assert torch.all(out.e_neg >= 0.0) and torch.all(out.e_neg <= 1.0)
        assert torch.all(out.truth >= 0.0) and torch.all(out.truth <= 1.0)
        assert torch.all(out.knowledge >= 0.0) and torch.all(out.knowledge <= 1.0)

    def test_belnap_attention_anti_saturation_large_inputs(self) -> None:
        """Verifies that large scale inputs do not cause sigmoid saturation or gradient death."""
        attn = BelnapAttention(d_model=32, n_heads=4)
        # Inputs with extreme variance/scale
        pos = (torch.randn(2, 4, 32) * 50.0).detach().requires_grad_(True)
        neg = (torch.randn(2, 4, 32) * 50.0).detach().requires_grad_(True)
        q = BelnapState(e_pos=pos, e_neg=neg)
        out = attn(q, q)
        assert not torch.isnan(out.e_pos).any()
        assert not torch.isnan(out.e_neg).any()
        assert torch.all(out.e_pos >= 0.0) and torch.all(out.e_pos <= 1.0)

        # Backpropagation should retain non-zero gradients
        loss = out.truth.sum() + out.knowledge.sum()
        loss.backward()
        assert q.e_pos.grad is not None
        assert torch.isfinite(q.e_pos.grad).all()
        assert torch.norm(q.e_pos.grad) > 1e-4

    def test_belnap_ffn_prenorm_layers_exist(self) -> None:
        """Verifies that BelnapFFN instantiates LayerNorm for output evidence projections."""
        ffn = BelnapFFN(d_model=32)
        assert hasattr(ffn, "norm_pos")
        assert hasattr(ffn, "norm_neg")
        assert isinstance(ffn.norm_pos, nn.LayerNorm)
        assert isinstance(ffn.norm_neg, nn.LayerNorm)
        assert ffn.norm_pos.normalized_shape == (32,)
        assert ffn.norm_neg.normalized_shape == (32,)

    def test_belnap_ffn_prenorm_forward_and_bounds(self) -> None:
        """Verifies that BelnapFFN pre-norm outputs remain bounded in [0, 1]
        with active gradients.
        """
        ffn = BelnapFFN(d_model=32, d_hidden=128)
        x = BelnapState(
            e_pos=torch.rand(2, 6, 32, requires_grad=True),
            e_neg=torch.rand(2, 6, 32, requires_grad=True),
        )
        out = ffn(x)
        assert torch.all(out.e_pos >= 0.0) and torch.all(out.e_pos <= 1.0)
        assert torch.all(out.e_neg >= 0.0) and torch.all(out.e_neg <= 1.0)

        loss = out.truth.sum() + out.knowledge.sum()
        loss.backward()
        assert x.e_pos.grad is not None
        assert torch.isfinite(x.e_pos.grad).all()
        assert torch.norm(x.e_pos.grad) > 1e-4


class TestResidualGradientHighway:
    """Verifies residual interpolation recalibration to alpha=0.15 for >35% gradient survival."""

    def test_residual_join_default_weight_015(self) -> None:
        """Verifies default alpha parameter is 0.15 in BelnapTransformerBlock and _residual_join."""
        block = BelnapTransformerBlock(d_model=32, n_heads=4)
        assert block.residual_weight == 0.15

        base = BelnapState(e_pos=torch.tensor([0.6]), e_neg=torch.tensor([0.4]))
        delta = BelnapState(e_pos=torch.tensor([0.2]), e_neg=torch.tensor([0.8]))

        # Calling without explicit alpha must default to 0.15
        res = BelnapTransformerBlock._residual_join(base, delta)
        expected_pos = 0.85 * 0.6 + 0.15 * 0.2
        expected_neg = 0.85 * 0.4 + 0.15 * 0.8
        assert torch.isclose(res.e_pos, torch.tensor([expected_pos]), atol=1e-5)
        assert torch.isclose(res.e_neg, torch.tensor([expected_neg]), atol=1e-5)

    def test_single_join_identity_gradient_survival(self) -> None:
        """Verifies that skip connection gradient d(res)/d(base) is exactly 1 - alpha = 0.85."""
        base = BelnapState(
            e_pos=torch.tensor([0.7], requires_grad=True),
            e_neg=torch.tensor([0.3], requires_grad=True),
        )
        delta = BelnapState(
            e_pos=torch.tensor([0.4]),
            e_neg=torch.tensor([0.5]),
        )
        res = BelnapTransformerBlock._residual_join(base, delta, alpha=0.15)
        res.e_pos.backward()
        assert base.e_pos.grad is not None
        assert torch.isclose(base.e_pos.grad, torch.tensor(0.85), atol=1e-6)

    def test_six_joins_gradient_retention_exceeds_35_percent(self) -> None:
        """Verifies that across 6 cascaded joins (2 blocks x 3 joins), residual gradient retention
        exceeds 35% ((1 - 0.15)^6 = 37.7%), resolving the 1.5% collapse under alpha=0.5.
        """
        base = BelnapState(
            e_pos=torch.tensor([0.5], requires_grad=True),
            e_neg=torch.tensor([0.5], requires_grad=True),
        )
        delta = BelnapState(
            e_pos=torch.tensor([0.3]),
            e_neg=torch.tensor([0.3]),
        )

        curr = base
        for _ in range(6):
            curr = BelnapTransformerBlock._residual_join(curr, delta, alpha=0.15)

        curr.e_pos.backward()
        assert base.e_pos.grad is not None
        grad_015 = base.e_pos.grad.item()

        expected_retention = (1.0 - 0.15) ** 6
        assert torch.isclose(torch.tensor(grad_015), torch.tensor(expected_retention), atol=1e-5)
        assert grad_015 > 0.35, f"Gradient retention {grad_015:.4f} must exceed 35% threshold"

        # Compare with old alpha=0.5
        base_old = BelnapState(
            e_pos=torch.tensor([0.5], requires_grad=True),
            e_neg=torch.tensor([0.5], requires_grad=True),
        )
        curr_old = base_old
        for _ in range(6):
            curr_old = BelnapTransformerBlock._residual_join(curr_old, delta, alpha=0.5)
        curr_old.e_pos.backward()
        assert base_old.e_pos.grad is not None
        grad_05 = base_old.e_pos.grad.item()

        ratio = grad_015 / grad_05
        msg = f"Alpha 0.15 should retain >24x more gradient than alpha 0.5 (got {ratio:.2f}x)"
        assert ratio > 24.0, msg

    def test_decision_transformer_default_residual_weight(self) -> None:
        """Verifies that BelnapDecisionTransformer layers adopt residual_weight=0.15."""
        model = BelnapDecisionTransformer(d_model=32, n_heads=4, n_layers=2)
        for layer in model.layers:
            assert isinstance(layer, BelnapTransformerBlock)
            assert layer.residual_weight == 0.15


class TestOptunaStudyNarrowConsistency:
    """Verifies that Optuna study configurations align with the 0.15 residual highway."""

    def test_optuna_narrow_fixed_residual_weight(self) -> None:
        """Verifies FIXED_RESIDUAL_WEIGHT in optuna_study_narrow is recalibrated to 0.15."""
        assert FIXED_RESIDUAL_WEIGHT == 0.15
