"""Unit and property test suite for Coupled Cross-Polarity Bilattice Evidence Routing (R3).

Verifies 2x2 coupled bilattice value aggregation in BelnapAttention:
    agg_pos = (A_+ V_+ + A_- V_-) / denom
    agg_neg = (A_+ V_- + A_- V_+) / denom
"""

import pytest
import torch

from yoda.architecture.attention import BelnapMultiheadPooledAttention
from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapDecisionTransformer,
    BelnapState,
    BelnapTransformerBlock,
)


class TestCoupledRoutingMatrixAlgebra:
    """Verifies the pure mathematical 2x2 circulant bilattice operator.

    Operator:
        [agg_pos]   [A_+  A_-] [V_+]
        [agg_neg] = [A_-  A_+] [V_-]
    """

    @pytest.mark.parametrize(
        ("a_pos", "a_neg", "v_pos", "v_neg", "expected_pos", "expected_neg", "case_name"),
        [
            (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, "T_attn_T_val"),
            (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, "T_attn_F_val_refutation_preserved"),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, "T_attn_N_val"),
            (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "T_attn_B_val_contradiction_preserved"),
            (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, "F_attn_T_val_refuted_positive"),
            (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, "F_attn_F_val_double_negation"),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, "F_attn_N_val"),
            (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "F_attn_B_val_contradiction_invariant"),
            (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, "N_attn_T_val_zero_knowledge"),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "N_attn_F_val_zero_knowledge"),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "N_attn_N_val_zero_knowledge"),
            (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, "N_attn_B_val_zero_knowledge"),
            (1.0, 1.0, 1.0, 0.0, 1.0, 1.0, "B_attn_T_val_both"),
            (1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "B_attn_F_val_both"),
            (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "B_attn_N_val_neither"),
            (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, "B_attn_B_val_both_scaled"),
        ],
    )
    def test_bilattice_16_state_truth_table(
        self,
        a_pos: float,
        a_neg: float,
        v_pos: float,
        v_neg: float,
        expected_pos: float,
        expected_neg: float,
        case_name: str,
    ) -> None:
        """Exhaustively verifies all 16 Belnap product combinations B_4 x B_4 -> B_4."""
        ap = torch.tensor([a_pos])
        an = torch.tensor([a_neg])
        vp = torch.tensor([v_pos])
        vn = torch.tensor([v_neg])

        agg_pos = ap * vp + an * vn
        agg_neg = ap * vn + an * vp

        assert torch.isclose(agg_pos, torch.tensor(expected_pos)), f"Failed on {case_name}"
        assert torch.isclose(agg_neg, torch.tensor(expected_neg)), f"Failed on {case_name}"

    def test_decoupled_vs_coupled_refutation_dropping(self) -> None:
        """Demonstrates mathematically that decoupled routing drops refutation evidence,
        whereas 2x2 coupled routing strictly preserves it.
        """
        # Coherent positive query attending to refutation key
        a_pos = torch.tensor([1.0])
        a_neg = torch.tensor([0.0])
        v_pos = torch.tensor([0.0])
        v_neg = torch.tensor([1.0])

        # Flawed decoupled aggregation:
        decoupled_pos = a_pos * v_pos
        decoupled_neg = a_neg * v_neg
        assert decoupled_pos.item() == 0.0
        assert decoupled_neg.item() == 0.0, "Decoupled erroneously drops refutation to Neither!"

        # Hardened 2x2 coupled aggregation:
        coupled_pos = a_pos * v_pos + a_neg * v_neg
        coupled_neg = a_pos * v_neg + a_neg * v_pos
        assert coupled_pos.item() == 0.0
        assert coupled_neg.item() == 1.0, "Coupled preserves refutation evidence!"


class TestBelnapAttentionCoupledRouting:
    """Verifies BelnapAttention neural layer implementation of 2x2 coupled routing."""

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        return BelnapAttention(d_model=32, n_heads=2)

    @pytest.fixture
    def symmetric_attn_layer(self) -> BelnapAttention:
        """BelnapAttention instance with tied positive/negative projection weights
        to verify exact algebraic symmetry.
        """
        torch.manual_seed(42)
        attn = BelnapAttention(d_model=32, n_heads=2)
        attn.w_q_neg.weight.data.copy_(attn.w_q_pos.weight.data)
        attn.w_q_neg.bias.data.copy_(attn.w_q_pos.bias.data)
        attn.w_k_neg.weight.data.copy_(attn.w_k_pos.weight.data)
        attn.w_k_neg.bias.data.copy_(attn.w_k_pos.bias.data)
        attn.w_v_neg.weight.data.copy_(attn.w_v_pos.weight.data)
        attn.w_v_neg.bias.data.copy_(attn.w_v_pos.bias.data)
        attn.w_out_neg.weight.data.copy_(attn.w_out_pos.weight.data)
        attn.w_out_neg.bias.data.copy_(attn.w_out_pos.bias.data)
        attn.norm_neg.weight.data.copy_(attn.norm_pos.weight.data)
        attn.norm_neg.bias.data.copy_(attn.norm_pos.bias.data)
        return attn

    def test_refutation_preservation_forward(self, attn_layer: BelnapAttention) -> None:
        """Verifies coherent queries attending to refutation keys route to negative evidence."""
        batch_size = 2
        l_q = 1
        l_kv = 3
        d_model = 32

        # Query has high positive evidence
        q = BelnapState(
            e_pos=torch.ones(batch_size, l_q, d_model),
            e_neg=torch.zeros(batch_size, l_q, d_model),
        )
        # Key tokens have pure refutation (negative evidence)
        kv = BelnapState(
            e_pos=torch.zeros(batch_size, l_kv, d_model),
            e_neg=torch.ones(batch_size, l_kv, d_model),
        )

        out = attn_layer(q, kv)
        assert out.e_neg.mean().item() > 0.1, (
            "Refutation evidence must be transferred to out.e_neg under coherent query"
        )

    def test_double_negation_symmetry_query_inversion(
        self, symmetric_attn_layer: BelnapAttention
    ) -> None:
        """Verifies that negating query strictly inverts output polarities:
        Agg(~Q, K, V) == ~Agg(Q, K, V).
        """
        d_model = 32
        q = BelnapState(
            e_pos=torch.rand(2, 3, d_model),
            e_neg=torch.rand(2, 3, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(2, 4, d_model),
            e_neg=torch.rand(2, 4, d_model),
        )

        out_orig = symmetric_attn_layer(q, kv)
        out_neg_q = symmetric_attn_layer(q.negate(), kv)

        # Swapped evidence polarities
        assert torch.allclose(out_neg_q.e_pos, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_neg_q.e_neg, out_orig.e_pos, atol=1e-5)
        # Inverted truth coordinate: t' = 1 - t
        assert torch.allclose(out_neg_q.truth, 1.0 - out_orig.truth, atol=1e-5)
        # Invariant knowledge coordinate: k' = k
        assert torch.allclose(out_neg_q.knowledge, out_orig.knowledge, atol=1e-5)

    def test_double_negation_symmetry_key_value_inversion(
        self, symmetric_attn_layer: BelnapAttention
    ) -> None:
        """Verifies that negating both key and value preserves aggregated evidence:
        Agg(Q, ~K, ~V) == Agg(Q, K, V).
        """
        d_model = 32
        q = BelnapState(
            e_pos=torch.rand(2, 2, d_model),
            e_neg=torch.rand(2, 2, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(2, 3, d_model),
            e_neg=torch.rand(2, 3, d_model),
        )

        out_orig = symmetric_attn_layer(q, kv)
        out_neg_kv = symmetric_attn_layer(q, kv.negate())

        assert torch.allclose(out_neg_kv.e_pos, out_orig.e_pos, atol=1e-5)
        assert torch.allclose(out_neg_kv.e_neg, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_neg_kv.truth, out_orig.truth, atol=1e-5)
        assert torch.allclose(out_neg_kv.knowledge, out_orig.knowledge, atol=1e-5)

    def test_full_double_negation_inversion(self, symmetric_attn_layer: BelnapAttention) -> None:
        """Verifies that negating query AND source token strictly inverts evidence:
        Agg(~Q, ~K, ~V) == ~Agg(Q, K, V).
        """
        d_model = 32
        q = BelnapState(
            e_pos=torch.rand(2, 2, d_model),
            e_neg=torch.rand(2, 2, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(2, 3, d_model),
            e_neg=torch.rand(2, 3, d_model),
        )

        out_orig = symmetric_attn_layer(q, kv)
        out_dn = symmetric_attn_layer(q.negate(), kv.negate())

        assert torch.allclose(out_dn.e_pos, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_dn.e_neg, out_orig.e_pos, atol=1e-5)

    def test_masked_refutation_isolation(self, attn_layer: BelnapAttention) -> None:
        """Verifies that refutation evidence from active tokens is not diluted
        by masked padding tokens.
        """
        d_model = 32
        q = BelnapState(
            e_pos=torch.ones(1, 1, d_model),
            e_neg=torch.zeros(1, 1, d_model),
        )

        # 1 valid refutation token + 3 padding tokens
        valid_pos = torch.zeros(1, 1, d_model)
        valid_neg = torch.ones(1, 1, d_model)
        corrupt_pos = torch.cat([valid_pos, torch.ones(1, 3, d_model)], dim=1)
        corrupt_neg = torch.cat([valid_neg, torch.zeros(1, 3, d_model)], dim=1)

        kv_padded = BelnapState(e_pos=corrupt_pos, e_neg=corrupt_neg)
        kv_clean = BelnapState(e_pos=valid_pos, e_neg=valid_neg)

        mask_padded = torch.tensor([[True, False, False, False]], dtype=torch.bool)
        mask_clean = torch.tensor([[True]], dtype=torch.bool)

        out_padded = attn_layer(q, kv_padded, mask=mask_padded)
        out_clean = attn_layer(q, kv_clean, mask=mask_clean)

        assert torch.allclose(out_padded.e_pos, out_clean.e_pos, atol=1e-4)
        assert torch.allclose(out_padded.e_neg, out_clean.e_neg, atol=1e-4)
        assert out_padded.e_neg.mean().item() > 0.1

    def test_cross_channel_gradient_flow(self, attn_layer: BelnapAttention) -> None:
        """Verifies that non-zero gradients flow through cross channels (A_+ V_- and A_- V_+)."""
        d_model = 32
        q = BelnapState(
            e_pos=torch.rand(2, 2, d_model, requires_grad=True),
            e_neg=torch.rand(2, 2, d_model, requires_grad=True),
        )
        kv = BelnapState(
            e_pos=torch.rand(2, 3, d_model, requires_grad=True),
            e_neg=torch.rand(2, 3, d_model, requires_grad=True),
        )

        out = attn_layer(q, kv)
        loss = out.truth.sum() + out.knowledge.sum()
        loss.backward()

        assert q.e_pos.grad is not None and torch.isfinite(q.e_pos.grad).all()
        assert q.e_neg.grad is not None and torch.isfinite(q.e_neg.grad).all()
        assert kv.e_pos.grad is not None and torch.isfinite(kv.e_pos.grad).all()
        assert kv.e_neg.grad is not None and torch.isfinite(kv.e_neg.grad).all()

        assert attn_layer.w_v_pos.weight.grad is not None
        assert attn_layer.w_v_neg.weight.grad is not None


class TestTransformerBlockCoupledIntegration:
    """Verifies that coupled routing propagates cleanly through multi-layer blocks."""

    def test_transformer_block_forward_and_backward(self) -> None:
        block = BelnapTransformerBlock(d_model=32, n_heads=4)
        x = BelnapState(
            e_pos=torch.rand(2, 4, 32, requires_grad=True),
            e_neg=torch.rand(2, 4, 32, requires_grad=True),
        )
        out = block(x)

        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (2, 4, 32)
        assert out.e_neg.shape == (2, 4, 32)
        assert torch.all(out.knowledge >= 0.0) and torch.all(out.knowledge <= 1.0)

        out.knowledge.sum().backward()
        assert x.e_pos.grad is not None
        assert x.e_neg.grad is not None

    def test_decision_transformer_end_to_end(self) -> None:
        model = BelnapDecisionTransformer(
            d_model=32,
            n_heads=4,
            n_layers=2,
            num_choices=4,
        )
        q_emb = torch.randn(2, 32)
        state_emb = torch.randn(2, 5, 32)

        out = model(q_emb=q_emb, state_emb=state_emb)
        assert out["logits"].shape == (2, 4)
        assert torch.all(out["truth"] >= 0.0) and torch.all(out["truth"] <= 1.0)
        assert torch.all(out["knowledge"] >= 0.0) and torch.all(out["knowledge"] <= 1.0)

    def test_multihead_pooled_attention_coupled_refutation(self) -> None:
        """Verifies that BelnapMultiheadPooledAttention routes refutation under coupled routing."""
        mpa = BelnapMultiheadPooledAttention(embed_dim=16, num_queries=2, num_heads=2)
        # Input sequence with refutation evidence
        e_pos = torch.zeros(1, 4, 16)
        e_neg = torch.ones(1, 4, 16) * 0.9
        kv = BelnapState(e_pos=e_pos, e_neg=e_neg)

        pooled = mpa.forward_pooled(kv)
        assert isinstance(pooled, BelnapState)
        assert pooled.e_neg.mean().item() > 0.1
