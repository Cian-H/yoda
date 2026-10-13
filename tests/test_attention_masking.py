"""Comprehensive unit and integration tests for Attention Mask Propagation & Scale Normalization."""

import pytest
import torch

from yoda.architecture.attention import BelnapMultiheadPooledAttention
from yoda.architecture.belnap_transformer import BelnapAttention, BelnapState
from yoda.architecture.engine import YodaDecisionEngine


class TestBelnapAttentionMasking:
    """Tests for padding mask injection and active sequence mass normalization
    in BelnapAttention.
    """

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        return BelnapAttention(d_model=32, n_heads=4)

    def test_2d_boolean_mask_broadcasting(self, attn_layer: BelnapAttention) -> None:
        """Verifies that 2D boolean mask (batch_size, l_kv) broadcasts seamlessly."""
        batch_size = 2
        l_q = 3
        l_kv = 5
        d_model = 32

        q = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(batch_size, l_kv, d_model),
            e_neg=torch.rand(batch_size, l_kv, d_model),
        )

        # 2D mask: sequence 0 has 3 valid tokens, sequence 1 has 4 valid tokens
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
            ],
            dtype=torch.bool,
        )

        out = attn_layer(q=q, kv=kv, mask=mask)

        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (batch_size, l_q, d_model)
        assert out.e_neg.shape == (batch_size, l_q, d_model)
        assert torch.all(out.e_pos >= 0.0) and torch.all(out.e_pos <= 1.0)
        assert torch.all(out.e_neg >= 0.0) and torch.all(out.e_neg <= 1.0)
        assert torch.all(torch.isfinite(out.e_pos))
        assert torch.all(torch.isfinite(out.e_neg))

    def test_padding_isolation(self, attn_layer: BelnapAttention) -> None:
        """Verifies that padding tokens have zero influence on aggregated output."""
        batch_size = 1
        l_q = 2
        d_model = 32

        q = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )

        # Base sequence with 2 valid tokens
        valid_pos = torch.rand(batch_size, 2, d_model)
        valid_neg = torch.rand(batch_size, 2, d_model)
        kv_clean = BelnapState(e_pos=valid_pos, e_neg=valid_neg)

        # Padded sequence with 2 valid tokens + 4 contaminated padding tokens
        corrupt_pos = torch.cat([valid_pos, torch.ones(batch_size, 4, d_model)], dim=1)
        corrupt_neg = torch.cat([valid_neg, torch.ones(batch_size, 4, d_model)], dim=1)
        kv_padded = BelnapState(e_pos=corrupt_pos, e_neg=corrupt_neg)

        mask = torch.tensor([[True, True, False, False, False, False]], dtype=torch.bool)

        mask_clean = torch.tensor([[True, True]], dtype=torch.bool)
        out_clean = attn_layer(q=q, kv=kv_clean, mask=mask_clean)
        out_padded = attn_layer(q=q, kv=kv_padded, mask=mask)

        # Output with masked padding tokens must match unpadded output
        assert torch.allclose(out_clean.e_pos, out_padded.e_pos, atol=1e-3)
        assert torch.allclose(out_clean.e_neg, out_padded.e_neg, atol=1e-3)

    def test_scale_normalization_sequence_length_invariance(
        self, attn_layer: BelnapAttention
    ) -> None:
        """Verifies that repeated identical tokens produce scale-invariant evidence."""
        batch_size = 1
        l_q = 1
        d_model = 32

        q = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )

        token_pos = torch.rand(batch_size, 1, d_model)
        token_neg = torch.rand(batch_size, 1, d_model)

        # Sequence of 2 identical tokens
        kv_2 = BelnapState(
            e_pos=token_pos.repeat(1, 2, 1),
            e_neg=token_neg.repeat(1, 2, 1),
        )
        # Sequence of 20 identical tokens
        kv_20 = BelnapState(
            e_pos=token_pos.repeat(1, 20, 1),
            e_neg=token_neg.repeat(1, 20, 1),
        )

        out_2 = attn_layer(q=q, kv=kv_2)
        out_20 = attn_layer(q=q, kv=kv_20)

        # With active mass normalization, evidence aggregation remains scale-invariant
        assert torch.allclose(out_2.e_pos, out_20.e_pos, atol=1e-4)
        assert torch.allclose(out_2.e_neg, out_20.e_neg, atol=1e-4)

    def test_mask_dtype_compatibility(self, attn_layer: BelnapAttention) -> None:
        """Verifies that boolean, integer, binary float, and additive float masks
        produce identical outputs.
        """
        batch_size = 1
        l_q = 2
        l_kv = 4
        d_model = 32

        q = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(batch_size, l_kv, d_model),
            e_neg=torch.rand(batch_size, l_kv, d_model),
        )

        mask_bool = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        mask_int = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        mask_float_bin = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        mask_additive = torch.tensor([[0.0, 0.0, -1e9, -1e9]], dtype=torch.float32)

        out_bool = attn_layer(q=q, kv=kv, mask=mask_bool)
        out_int = attn_layer(q=q, kv=kv, mask=mask_int)
        out_float_bin = attn_layer(q=q, kv=kv, mask=mask_float_bin)
        out_additive = attn_layer(q=q, kv=kv, mask=mask_additive)

        assert torch.allclose(out_bool.e_pos, out_int.e_pos, atol=1e-6)
        assert torch.allclose(out_bool.e_pos, out_float_bin.e_pos, atol=1e-6)
        assert torch.allclose(out_bool.e_pos, out_additive.e_pos, atol=1e-6)

    def test_all_masked_tokens_stability(self, attn_layer: BelnapAttention) -> None:
        """Verifies numerical stability when all sequence tokens are masked (clamp(min=1.0))."""
        batch_size = 1
        l_q = 2
        l_kv = 3
        d_model = 32

        q = BelnapState(
            e_pos=torch.rand(batch_size, l_q, d_model),
            e_neg=torch.rand(batch_size, l_q, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(batch_size, l_kv, d_model),
            e_neg=torch.rand(batch_size, l_kv, d_model),
        )

        all_masked = torch.tensor([[False, False, False]], dtype=torch.bool)
        out = attn_layer(q=q, kv=kv, mask=all_masked)

        assert torch.all(torch.isfinite(out.e_pos))
        assert torch.all(torch.isfinite(out.e_neg))
        assert torch.all(out.knowledge <= 0.6)


class TestEndToEndAttentionMasking:
    """Verifies end-to-end mask propagation from encoders through MPA into YodaDecisionEngine."""

    def test_belnap_mpa_with_2d_mask(self) -> None:
        """Verifies that BelnapMultiheadPooledAttention directly accepts 2D token masks."""
        batch_size = 2
        seq_len = 6
        embed_dim = 16
        num_queries = 4

        mpa = BelnapMultiheadPooledAttention(embed_dim=embed_dim, num_queries=num_queries)
        kv_state = BelnapState(
            e_pos=torch.rand(batch_size, seq_len, embed_dim),
            e_neg=torch.rand(batch_size, seq_len, embed_dim),
        )

        mask_2d = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, True, False],
            ],
            dtype=torch.bool,
        )

        latent, pooled_state = mpa(kv_state, mask=mask_2d)

        assert latent.shape == (batch_size, embed_dim)
        assert pooled_state.e_pos.shape == (batch_size, num_queries, embed_dim)
        assert torch.all(torch.isfinite(latent))
        assert torch.all(torch.isfinite(pooled_state.e_pos))

    def test_yoda_decision_engine_variable_length_propagation(self) -> None:
        """Verifies end-to-end forward and backward passes with variable-length text inputs."""
        torch.manual_seed(42)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_q_probes=4,
            num_c_probes=4,
            num_k_probes=4,
            num_choices=3,
            n_heads=2,
        )

        queries = ["short", "a much longer query with many words in it"]
        states = [{"key": "val"}, {"a": 1, "b": 2, "c": 3}]
        constraints = [["rule1"], ["rule1", "rule2", "rule3"]]

        out = engine(queries=queries, states=states, constraints=constraints)

        assert out["logits"].shape == (2, 3)
        assert torch.all(torch.isfinite(out["logits"]))

        loss = out["logits"].sum()
        loss.backward()

        assert engine.decision_head.w_pos.weight.grad is not None
        assert not torch.isnan(engine.decision_head.w_pos.weight.grad).any()
        assert engine.q_mpa.raw_q_pos.grad is not None
        assert not torch.isnan(engine.q_mpa.raw_q_pos.grad).any()
