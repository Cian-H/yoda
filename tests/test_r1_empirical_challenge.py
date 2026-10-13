"""Empirical Challenge Stress Test Harness for Requirement R1.

Target: Attention Mask Propagation & Active Sequence Mass Scale Normalization.
Role: Empirical Challenger (Critic / Specialist).
Methodology: Property testing, adversarial input generation, boundary stress,
zero-mass stability, and gradient leakage verification.
"""

import pytest
import torch

from yoda.architecture.attention import BelnapMultiheadPooledAttention
from yoda.architecture.belnap_transformer import BelnapAttention, BelnapState
from yoda.architecture.engine import YodaDecisionEngine


class TestAdversarialPaddingIsolation:
    """Stress-tests that padding tokens, regardless of length, position, or adversarial values,
    have zero influence on the pooled output representation.
    """

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        layer = BelnapAttention(d_model=32, n_heads=4)
        layer.eval()
        return layer

    @pytest.mark.parametrize("pad_len", [1, 10, 50, 128, 255, 511, 1023])
    def test_extreme_padding_ratios_isolation(
        self, attn_layer: BelnapAttention, pad_len: int
    ) -> None:
        """Verifies single active token produces identical outputs across padding lengths."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv_clean = BelnapState(e_pos=torch.rand(1, 1, d_model), e_neg=torch.rand(1, 1, d_model))
        mask_clean = torch.ones(1, 1, dtype=torch.bool)

        out_clean = attn_layer(q, kv_clean, mask=mask_clean)

        # Pad with random noise in [0, 1]
        pad_pos = torch.rand(1, pad_len, d_model)
        pad_neg = torch.rand(1, pad_len, d_model)
        kv_padded = BelnapState(
            e_pos=torch.cat([kv_clean.e_pos, pad_pos], dim=1),
            e_neg=torch.cat([kv_clean.e_neg, pad_neg], dim=1),
        )
        mask_padded = torch.cat(
            [torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, pad_len, dtype=torch.bool)],
            dim=1,
        )

        out_padded = attn_layer(q, kv_padded, mask=mask_padded)

        diff_pos = (out_clean.e_pos - out_padded.e_pos).abs().max().item()
        diff_neg = (out_clean.e_neg - out_padded.e_neg).abs().max().item()

        assert diff_pos < 1e-4, f"Pos diff {diff_pos} exceeded tolerance for pad_len={pad_len}"
        assert diff_neg < 1e-4, f"Neg diff {diff_neg} exceeded tolerance for pad_len={pad_len}"

    @pytest.mark.parametrize("pad_type", ["all_ones", "all_zeros", "alternating", "high_knowledge"])
    def test_adversarial_pad_token_payloads(
        self, attn_layer: BelnapAttention, pad_type: str
    ) -> None:
        """Verifies that maliciously crafted padding token evidence cannot leak into outputs."""
        d_model = 32
        pad_len = 200
        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv_clean = BelnapState(e_pos=torch.rand(1, 3, d_model), e_neg=torch.rand(1, 3, d_model))
        mask_clean = torch.ones(1, 3, dtype=torch.bool)
        out_clean = attn_layer(q, kv_clean, mask=mask_clean)

        if pad_type == "all_ones":
            pad_pos = torch.ones(1, pad_len, d_model)
            pad_neg = torch.ones(1, pad_len, d_model)
        elif pad_type == "all_zeros":
            pad_pos = torch.zeros(1, pad_len, d_model)
            pad_neg = torch.zeros(1, pad_len, d_model)
        elif pad_type == "alternating":
            pad_pos = torch.zeros(1, pad_len, d_model)
            pad_pos[:, ::2, :] = 1.0
            pad_neg = 1.0 - pad_pos
        elif pad_type == "high_knowledge":
            pad_pos = torch.ones(1, pad_len, d_model) * 0.99
            pad_neg = torch.ones(1, pad_len, d_model) * 0.99

        kv_padded = BelnapState(
            e_pos=torch.cat([kv_clean.e_pos, pad_pos], dim=1),
            e_neg=torch.cat([kv_clean.e_neg, pad_neg], dim=1),
        )
        mask_padded = torch.cat(
            [torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, pad_len, dtype=torch.bool)],
            dim=1,
        )

        out_padded = attn_layer(q, kv_padded, mask=mask_padded)

        diff_pos = (out_clean.e_pos - out_padded.e_pos).abs().max().item()
        diff_neg = (out_clean.e_neg - out_padded.e_neg).abs().max().item()

        assert diff_pos < 1e-4, f"Pos diff {diff_pos} for {pad_type}"
        assert diff_neg < 1e-4, f"Neg diff {diff_neg} for {pad_type}"

    def test_arbitrary_active_position_invariance(self, attn_layer: BelnapAttention) -> None:
        """Verifies active token outputs are invariant whether placed at start, middle, or end."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        tok_pos = torch.rand(1, 1, d_model)
        tok_neg = torch.rand(1, 1, d_model)
        kv_single = BelnapState(e_pos=tok_pos, e_neg=tok_neg)
        out_ref = attn_layer(q, kv_single, mask=torch.ones(1, 1, dtype=torch.bool))

        # 1. Active at start: [active, pad x 50]
        kv_start = BelnapState(
            e_pos=torch.cat([tok_pos, torch.rand(1, 50, d_model)], dim=1),
            e_neg=torch.cat([tok_neg, torch.rand(1, 50, d_model)], dim=1),
        )
        mask_start = torch.cat(
            [torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 50, dtype=torch.bool)], dim=1
        )
        out_start = attn_layer(q, kv_start, mask=mask_start)

        # 2. Active in middle: [pad x 25, active, pad x 25]
        kv_mid = BelnapState(
            e_pos=torch.cat(
                [torch.rand(1, 25, d_model), tok_pos, torch.rand(1, 25, d_model)], dim=1
            ),
            e_neg=torch.cat(
                [torch.rand(1, 25, d_model), tok_neg, torch.rand(1, 25, d_model)], dim=1
            ),
        )
        mask_mid = torch.cat(
            [
                torch.zeros(1, 25, dtype=torch.bool),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 25, dtype=torch.bool),
            ],
            dim=1,
        )
        out_mid = attn_layer(q, kv_mid, mask=mask_mid)

        # 3. Active at end: [pad x 50, active]
        kv_end = BelnapState(
            e_pos=torch.cat([torch.rand(1, 50, d_model), tok_pos], dim=1),
            e_neg=torch.cat([torch.rand(1, 50, d_model), tok_neg], dim=1),
        )
        mask_end = torch.cat(
            [torch.zeros(1, 50, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)], dim=1
        )
        out_end = attn_layer(q, kv_end, mask=mask_end)

        assert torch.allclose(out_ref.e_pos, out_start.e_pos, atol=1e-4)
        assert torch.allclose(out_ref.e_pos, out_mid.e_pos, atol=1e-4)
        assert torch.allclose(out_ref.e_pos, out_end.e_pos, atol=1e-4)


class TestScaleInvarianceAcrossLengths:
    """Stress-tests that evidence aggregation is scale-normalized by active sequence mass."""

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        layer = BelnapAttention(d_model=32, n_heads=4)
        layer.eval()
        return layer

    @pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 16, 31, 64, 127, 256, 512])
    def test_homogeneous_token_scale_invariance_sweep(
        self, attn_layer: BelnapAttention, length: int
    ) -> None:
        """Verifies repeated identical tokens produce scale-invariant evidence up to L=512."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        tok_pos = torch.rand(1, 1, d_model)
        tok_neg = torch.rand(1, 1, d_model)

        # Reference output: single token
        kv_ref = BelnapState(e_pos=tok_pos, e_neg=tok_neg)
        out_ref = attn_layer(q, kv_ref, mask=torch.ones(1, 1, dtype=torch.bool))

        # Repeated sequence of length L
        kv_L = BelnapState(
            e_pos=tok_pos.repeat(1, length, 1),
            e_neg=tok_neg.repeat(1, length, 1),
        )
        mask_L = torch.ones(1, length, dtype=torch.bool)
        out_L = attn_layer(q, kv_L, mask=mask_L)

        diff = (out_ref.e_pos - out_L.e_pos).abs().max().item()
        assert diff < 1e-4, f"Scale invariance broke at L={length} with diff={diff}"

    def test_causal_mask_active_mass_normalization(self, attn_layer: BelnapAttention) -> None:
        """Verifies lower-triangular causal masks normalize each query step by its prefix mass."""
        d_model = 32
        seq_len = 8
        tok_pos = torch.rand(1, 1, d_model)
        tok_neg = torch.rand(1, 1, d_model)
        kv = BelnapState(e_pos=tok_pos.repeat(1, seq_len, 1), e_neg=tok_neg.repeat(1, seq_len, 1))

        # 4D Causal mask: [1, 1, seq_len, seq_len]
        causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool))
        out_causal = attn_layer(q=kv, kv=kv, mask=causal_mask)

        # Since all tokens are identical, every causal step must yield identical evidence
        for t in range(seq_len):
            assert torch.allclose(out_causal.e_pos[:, 0, :], out_causal.e_pos[:, t, :], atol=1e-4)


class TestZeroMassAndExtremeBoundaries:
    """Stress-tests zero-mass clamping, all-pad inputs, empty batches, and numerical bounds."""

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        return BelnapAttention(d_model=32, n_heads=4)

    @pytest.mark.parametrize("seq_len", [1, 8, 64, 512])
    def test_all_pad_zero_active_mass_safety(
        self, attn_layer: BelnapAttention, seq_len: int
    ) -> None:
        """Verifies sequences with 0 active tokens clamp safely without NaN/Inf across lengths."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv = BelnapState(
            e_pos=torch.rand(1, seq_len, d_model),
            e_neg=torch.rand(1, seq_len, d_model),
        )
        all_pad_mask = torch.zeros(1, seq_len, dtype=torch.bool)

        out = attn_layer(q, kv, mask=all_pad_mask)

        assert torch.isfinite(out.e_pos).all()
        assert torch.isfinite(out.e_neg).all()
        assert (out.e_pos >= 0.0).all() and (out.e_pos <= 1.0).all()
        assert (out.e_neg >= 0.0).all() and (out.e_neg <= 1.0).all()
        # Clamped zero-mass state should suppress towards Neither (low knowledge)
        assert (out.knowledge <= 0.6).all()

    def test_zero_mass_backward_gradient_stability(self, attn_layer: BelnapAttention) -> None:
        """Verifies backward pass through an all-pad zero-mass input produces finite gradients."""
        d_model = 32
        q_pos = torch.rand(1, 2, d_model, requires_grad=True)
        q_neg = torch.rand(1, 2, d_model, requires_grad=True)
        q = BelnapState(e_pos=q_pos, e_neg=q_neg)

        kv_pos = torch.rand(1, 16, d_model, requires_grad=True)
        kv_neg = torch.rand(1, 16, d_model, requires_grad=True)
        kv = BelnapState(e_pos=kv_pos, e_neg=kv_neg)

        all_pad_mask = torch.zeros(1, 16, dtype=torch.bool)
        out = attn_layer(q, kv, mask=all_pad_mask)

        loss = out.e_pos.sum() + out.e_neg.sum()
        loss.backward()

        assert q_pos.grad is not None and torch.isfinite(q_pos.grad).all()
        assert q_neg.grad is not None and torch.isfinite(q_neg.grad).all()
        # Padding tokens must have 0 gradient
        assert kv_pos.grad is not None and (kv_pos.grad == 0.0).all()
        assert kv_neg.grad is not None and (kv_neg.grad == 0.0).all()

    def test_mixed_zero_mass_and_active_batch_independence(
        self, attn_layer: BelnapAttention
    ) -> None:
        """Verifies zero-mass samples in a batch do not corrupt neighboring active samples."""
        d_model = 32
        attn_layer.eval()

        # Isolated active sample
        q_active = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv_active = BelnapState(e_pos=torch.rand(1, 10, d_model), e_neg=torch.rand(1, 10, d_model))
        m_active = torch.ones(1, 10, dtype=torch.bool)
        out_active_iso = attn_layer(q_active, kv_active, mask=m_active)

        # Batch: Sample 0 is all-pad, Sample 1 is active, Sample 2 is all-pad
        q_batch = BelnapState(
            e_pos=q_active.e_pos.repeat(3, 1, 1),
            e_neg=q_active.e_neg.repeat(3, 1, 1),
        )
        kv_batch = BelnapState(
            e_pos=kv_active.e_pos.repeat(3, 1, 1),
            e_neg=kv_active.e_neg.repeat(3, 1, 1),
        )
        mask_batch = torch.tensor(
            [
                [False] * 10,
                [True] * 10,
                [False] * 10,
            ],
            dtype=torch.bool,
        )

        out_batch = attn_layer(q_batch, kv_batch, mask=mask_batch)

        assert torch.isfinite(out_batch.e_pos).all()
        assert torch.allclose(out_active_iso.e_pos, out_batch.e_pos[1:2], atol=1e-4)


class TestGradientIsolationAndLeakage:
    """Stress-tests that backpropagation through masked attention strictly leaks zero gradients."""

    def test_gradient_leakage_into_padding_positions(self) -> None:
        """Verifies exact zero gradient accumulation in masked key-value positions."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=4)

        q = BelnapState(
            e_pos=torch.rand(2, 3, d_model, requires_grad=True),
            e_neg=torch.rand(2, 3, d_model, requires_grad=True),
        )
        kv_pos = torch.rand(2, 20, d_model, requires_grad=True)
        kv_neg = torch.rand(2, 20, d_model, requires_grad=True)
        kv = BelnapState(e_pos=kv_pos, e_neg=kv_neg)

        # Sample 0 has 5 active tokens, Sample 1 has 12 active tokens
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :12] = True

        out = attn(q, kv, mask=mask)
        loss = (out.truth * out.knowledge).sum()
        loss.backward()

        # Padded positions in sample 0 (index 5 to 19)
        assert kv_pos.grad is not None
        assert kv_neg.grad is not None
        assert (kv_pos.grad[0, 5:] == 0.0).all(), "Non-zero gradient leak in sample 0 pos!"
        assert (kv_neg.grad[0, 5:] == 0.0).all(), "Non-zero gradient leak in sample 0 neg!"

        # Padded positions in sample 1 (index 12 to 19)
        assert (kv_pos.grad[1, 12:] == 0.0).all(), "Non-zero gradient leak in sample 1 pos!"
        assert (kv_neg.grad[1, 12:] == 0.0).all(), "Non-zero gradient leak in sample 1 neg!"

        # Active positions must have non-zero gradients
        assert (kv_pos.grad[0, :5] != 0.0).any()
        assert (kv_pos.grad[1, :12] != 0.0).any()


class TestEndToEndRaggedBatchIndependence:
    """Stress-tests end-to-end integration across encoders, Belnap MPA, and decision engine."""

    def test_ragged_batch_mpa_invariance(self) -> None:
        """Verifies BelnapMultiheadPooledAttention output for a sequence is identical
        when evaluated alone vs when batched with arbitrarily longer sequences.
        """
        torch.manual_seed(42)
        mpa = BelnapMultiheadPooledAttention(embed_dim=32, num_queries=4, n_heads=2)
        mpa.eval()

        # 3 samples of lengths 2, 7, 15
        s0 = torch.randn(1, 2, 32)
        m0 = torch.ones(1, 2, dtype=torch.bool)
        lat0_iso, state0_iso = mpa(s0, mask=m0)

        s1 = torch.randn(1, 7, 32)
        m1 = torch.ones(1, 7, dtype=torch.bool)
        lat1_iso, state1_iso = mpa(s1, mask=m1)

        s2 = torch.randn(1, 15, 32)
        m2 = torch.ones(1, 15, dtype=torch.bool)
        lat2_iso, state2_iso = mpa(s2, mask=m2)

        # Batch padded to max_len = 15
        s0_padded = torch.cat([s0, torch.randn(1, 13, 32)], dim=1)
        m0_padded = torch.cat([m0, torch.zeros(1, 13, dtype=torch.bool)], dim=1)

        s1_padded = torch.cat([s1, torch.randn(1, 8, 32)], dim=1)
        m1_padded = torch.cat([m1, torch.zeros(1, 8, dtype=torch.bool)], dim=1)

        batch_s = torch.cat([s0_padded, s1_padded, s2], dim=0)
        batch_m = torch.cat([m0_padded, m1_padded, m2], dim=0)

        batch_lat, batch_state = mpa(batch_s, mask=batch_m)

        assert torch.allclose(lat0_iso, batch_lat[0:1], atol=1e-4)
        assert torch.allclose(state0_iso.e_pos, batch_state.e_pos[0:1], atol=1e-4)

        assert torch.allclose(lat1_iso, batch_lat[1:2], atol=1e-4)
        assert torch.allclose(state1_iso.e_pos, batch_state.e_pos[1:2], atol=1e-4)

        assert torch.allclose(lat2_iso, batch_lat[2:3], atol=1e-4)
        assert torch.allclose(state2_iso.e_pos, batch_state.e_pos[2:3], atol=1e-4)

    def test_decision_engine_ragged_batch_stability(self) -> None:
        """Verifies YodaDecisionEngine executes forward and backward passes over
        batches with extreme variable-length text, empty strings, and deep states.
        """
        torch.manual_seed(42)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_q_probes=4,
            num_c_probes=4,
            num_k_probes=4,
            num_choices=4,
            n_heads=2,
            num_reasoning_blocks=2,
        )

        queries = [
            "short",
            "",
            "this is a very long and verbose query containing dozens of words for testing",
            "medium length sentence query",
        ]
        states = [
            {"a": 1},
            {},
            {f"key_{i}": f"val_{i}" for i in range(20)},
            {"status": "ok"},
        ]
        constraints = [
            ["rule1"],
            [],
            [f"constraint_clause_{i}" for i in range(10)],
            ["default_rule"],
        ]

        out = engine(
            queries=queries,
            states=states,
            constraints=constraints,
            return_diagnostics=True,
        )

        assert out["logits"].shape == (4, 4)
        assert torch.isfinite(out["logits"]).all()
        assert (out["truth"] >= 0.0).all() and (out["truth"] <= 1.0).all()
        assert (out["knowledge"] >= 0.0).all() and (out["knowledge"] <= 1.0).all()

        # Backward pass check
        loss = out["logits"].sum()
        loss.backward()

        for name, param in engine.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"
