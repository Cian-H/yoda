"""Unit and property-based test suite for Belnap Bilattice Native Transformer."""

from __future__ import annotations

import pytest
import torch

# The module to be implemented by TDD agent
from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapDecisionTransformer,
    BelnapFFN,
    BelnapState,
    BelnapTransformerBlock,
)


class TestBelnapState:
    """Verifies invariants of the four-valued bilattice state representation."""

    def test_initialization_and_coordinates(self) -> None:
        """Verifies truth and knowledge coordinate calculations."""
        # True: e+ = 1.0, e- = 0.0 -> t = 1.0, k = 0.5
        t_state = BelnapState(
            e_pos=torch.tensor([1.0]),
            e_neg=torch.tensor([0.0]),
        )
        assert torch.isclose(t_state.truth, torch.tensor([1.0]))
        assert torch.isclose(t_state.knowledge, torch.tensor([0.5]))

        # False: e+ = 0.0, e- = 1.0 -> t = 0.0, k = 0.5
        f_state = BelnapState(
            e_pos=torch.tensor([0.0]),
            e_neg=torch.tensor([1.0]),
        )
        assert torch.isclose(f_state.truth, torch.tensor([0.0]))
        assert torch.isclose(f_state.knowledge, torch.tensor([0.5]))

        # Neither (Unknown): e+ = 0.0, e- = 0.0 -> t = 0.5, k = 0.0
        n_state = BelnapState(
            e_pos=torch.tensor([0.0]),
            e_neg=torch.tensor([0.0]),
        )
        assert torch.isclose(n_state.truth, torch.tensor([0.5]))
        assert torch.isclose(n_state.knowledge, torch.tensor([0.0]))

        # Both (Contradiction): e+ = 1.0, e- = 1.0 -> t = 0.5, k = 1.0
        b_state = BelnapState(
            e_pos=torch.tensor([1.0]),
            e_neg=torch.tensor([1.0]),
        )
        assert torch.isclose(b_state.truth, torch.tensor([0.5]))
        assert torch.isclose(b_state.knowledge, torch.tensor([1.0]))

    def test_logical_negation(self) -> None:
        """Verifies that logical negation swaps evidence and inverts truth
        while preserving knowledge.
        """
        state = BelnapState(
            e_pos=torch.tensor([0.8, 0.2]),
            e_neg=torch.tensor([0.1, 0.9]),
        )
        negated = state.negate()

        # e+ and e- must be swapped
        assert torch.allclose(negated.e_pos, state.e_neg)
        assert torch.allclose(negated.e_neg, state.e_pos)

        # Truth must invert: t' = 1 - t
        assert torch.allclose(negated.truth, 1.0 - state.truth)
        # Knowledge must be invariant: k' = k
        assert torch.allclose(negated.knowledge, state.knowledge)

    def test_from_tk_roundtrip(self) -> None:
        """Verifies construction from (truth, knowledge) coordinates."""
        t_orig = torch.tensor([0.9, 0.1, 0.5, 0.5])
        k_orig = torch.tensor([0.5, 0.5, 0.0, 1.0])

        state = BelnapState.from_tk(t_orig, k_orig)
        assert torch.allclose(state.truth, t_orig, atol=1e-5)
        assert torch.allclose(state.knowledge, k_orig, atol=1e-5)


class TestBelnapAttention:
    """Verifies the Bipolar Support/Refutation Attention mechanism."""

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        return BelnapAttention(d_model=32, n_heads=4)

    def test_forward_shape_and_bounds(self, attn_layer: BelnapAttention) -> None:
        """Verifies output shapes and [0, 1] evidence bounds."""
        batch_size = 2
        l_q = 4
        l_kv = 6
        d_model = 32

        q_state = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )
        kv_state = BelnapState(
            e_pos=torch.rand(batch_size, l_kv, d_model),
            e_neg=torch.rand(batch_size, l_kv, d_model),
        )

        out_state = attn_layer(q_state, kv_state)

        assert isinstance(out_state, BelnapState)
        assert out_state.e_pos.shape == (batch_size, l_q, d_model)
        assert out_state.e_neg.shape == (batch_size, l_q, d_model)
        assert torch.all(out_state.e_pos >= 0.0) and torch.all(out_state.e_pos <= 1.0)
        assert torch.all(out_state.e_neg >= 0.0) and torch.all(out_state.e_neg <= 1.0)
        assert torch.all(out_state.truth >= 0.0) and torch.all(out_state.truth <= 1.0)
        assert torch.all(out_state.knowledge >= 0.0) and torch.all(out_state.knowledge <= 1.0)

    def test_zero_knowledge_suppression(self) -> None:
        """Verifies that completely unknown context (k=0) suppresses attention output
        to Neither when negative bias is configured.
        """
        batch_size = 1
        l_q = 2
        l_kv = 4
        d_model = 32

        q_state = BelnapState(
            e_pos=torch.ones(batch_size, l_q, d_model) * 0.8,
            e_neg=torch.zeros(batch_size, l_q, d_model),
        )
        # All KV tokens have 0 evidence (Neither, k=0)
        kv_state = BelnapState(
            e_pos=torch.zeros(batch_size, l_kv, d_model),
            e_neg=torch.zeros(batch_size, l_kv, d_model),
        )

        suppressed_layer = BelnapAttention(d_model=d_model, n_heads=4, init_bias=-3.0)
        out_state = suppressed_layer(q_state, kv_state)

        # Output knowledge should collapse toward zero (Neither / uninformative)
        assert torch.mean(out_state.knowledge) < 0.1
        # Truth should hover at 0.5 (neutral indifference)
        assert torch.allclose(out_state.truth, torch.tensor(0.5), atol=0.2)

        # Default init_bias=0.0 preserves neutral gradient-rich knowledge around 0.5
        default_layer = BelnapAttention(d_model=d_model, n_heads=4)
        out_default = default_layer(q_state, kv_state)
        assert torch.mean(out_default.knowledge) > 0.4

    def test_gradient_flow(self, attn_layer: BelnapAttention) -> None:
        """Verifies end-to-end backpropagation through dual evidence paths."""
        q_state = BelnapState(
            e_pos=torch.rand(2, 3, 32, requires_grad=True),
            e_neg=torch.rand(2, 3, 32, requires_grad=True),
        )
        kv_state = BelnapState(
            e_pos=torch.rand(2, 5, 32, requires_grad=True),
            e_neg=torch.rand(2, 5, 32, requires_grad=True),
        )

        out_state = attn_layer(q_state, kv_state)
        loss = out_state.truth.sum() + out_state.knowledge.sum()
        loss.backward()

        assert q_state.e_pos.grad is not None
        assert not torch.isnan(q_state.e_pos.grad).any()
        assert attn_layer.w_q_pos.weight.grad is not None


class TestBelnapFFN:
    """Verifies the Bilattice Feed-Forward Network with conflation."""

    def test_ffn_forward_and_conflation(self) -> None:
        """Verifies FFN output dimensions and conflation effect."""
        ffn = BelnapFFN(d_model=32, d_hidden=64, conflation_weight=0.1)
        in_state = BelnapState(
            e_pos=torch.rand(2, 4, 32),
            e_neg=torch.rand(2, 4, 32),
        )
        out_state = ffn(in_state)

        assert isinstance(out_state, BelnapState)
        assert out_state.e_pos.shape == (2, 4, 32)
        assert torch.all(out_state.e_pos >= 0.0) and torch.all(out_state.e_pos <= 1.0)


class TestBelnapTransformerBlock:
    """Verifies the unified Belnap Transformer Block with residual joins."""

    def test_block_forward(self) -> None:
        block = BelnapTransformerBlock(d_model=32, n_heads=4)
        x = BelnapState(
            e_pos=torch.rand(2, 5, 32),
            e_neg=torch.rand(2, 5, 32),
        )
        out = block(x)

        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (2, 5, 32)
        assert torch.all(out.knowledge >= 0.0) and torch.all(out.knowledge <= 1.0)

    def test_residual_join_convex_combination(self) -> None:
        """Verifies that residual join uses convex combination to guarantee [0, 1] bounds
        and maintain non-zero gradients even when base + delta > 1.0.
        """
        block = BelnapTransformerBlock(d_model=32, n_heads=4, residual_weight=0.5)
        base = BelnapState(
            e_pos=torch.full((2, 3, 32), 0.9, requires_grad=True),
            e_neg=torch.full((2, 3, 32), 0.8, requires_grad=True),
        )
        delta = BelnapState(
            e_pos=torch.full((2, 3, 32), 0.7),
            e_neg=torch.full((2, 3, 32), 0.6),
        )

        res = block._residual_join(base, delta, alpha=0.5)

        # In old clamp implementation, base.e_pos (0.9) + delta.e_pos (0.7) = 1.6 -> clamped to 1.0.
        # In convex combination, 0.5 * 0.9 + 0.5 * 0.7 = 0.8.
        assert torch.allclose(res.e_pos, torch.tensor(0.8))
        assert torch.allclose(res.e_neg, torch.tensor(0.7))
        assert torch.all(res.e_pos >= 0.0) and torch.all(res.e_pos <= 1.0)
        assert torch.all(res.e_neg >= 0.0) and torch.all(res.e_neg <= 1.0)

        # Verify gradient is non-zero (unlike clamp which has 0 gradient at saturated ceiling)
        res.e_pos.sum().backward()
        assert base.e_pos.grad is not None
        assert torch.allclose(base.e_pos.grad, torch.tensor(0.5))


class TestBelnapDecisionTransformer:
    """Verifies the end-to-end multi-branch Belnap Decision Transformer."""

    @pytest.fixture
    def model(self) -> BelnapDecisionTransformer:
        torch.manual_seed(42)
        return BelnapDecisionTransformer(
            d_model=32,
            n_heads=4,
            n_layers=2,
            num_choices=5,
        )

    def test_model_forward(self, model: BelnapDecisionTransformer) -> None:
        """Verifies end-to-end forward pass producing calibrated choice logits and (t, k) coords."""
        batch_size = 2
        l_state = 6

        q_emb = torch.randn(batch_size, 32)
        state_emb = torch.randn(batch_size, l_state, 32)

        output = model(q_emb=q_emb, state_emb=state_emb)

        assert "logits" in output
        assert "truth" in output
        assert "knowledge" in output
        assert "choice" in output

        assert output["logits"].shape == (batch_size, 5)
        assert output["truth"].shape == (batch_size, 5)
        assert output["knowledge"].shape == (batch_size, 5)

        # Truth and knowledge coordinates must be strictly in [0, 1]
        assert torch.all(output["truth"] >= 0.0) and torch.all(output["truth"] <= 1.0)
        assert torch.all(output["knowledge"] >= 0.0) and torch.all(output["knowledge"] <= 1.0)
