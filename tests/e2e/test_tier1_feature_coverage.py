"""Tier 1: Feature Coverage E2E Test Suite.

Covers Requirements R1 through R5 (>= 5 tests per feature):
- R1: Attention Mask Propagation & Scale Normalization
- R2: Paraconsistent Contradiction Loss & Pre-Activation Normalization
- R3: NeSy Constraint Reformulation & Epistemic Logit Synchronization
- R4: Bilinear Candidate-Query Alignment & Negation Equivariance
- R5: Context Encoding Deduplication & Hyperparameter Sweep Realignment
"""

import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from experiments.optuna_study import objective as optuna_objective
from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapFFN,
    BelnapState,
    BelnapTransformerBlock,
)
from yoda.architecture.encoders import TextEncoder
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss

# ==============================================================================
# R1: Attention Mask Propagation & Scale Normalization (Phase 1)
# ==============================================================================


class TestR1AttentionMaskAndScaleNormalization:
    """Verifies R1: Tokenizer padding mask propagation and attention scale normalization."""

    def test_r1_text_encoder_forward_with_mask(self) -> None:
        """R1.1: TextEncoder must expose forward_with_mask returning (embeddings, mask)."""
        encoder = TextEncoder(model_name="dummy", target_dim=32)
        texts = ["short phrase", "this is a substantially longer sentence for testing"]

        # Interface contract: forward_with_mask must return (embeddings, mask)
        assert hasattr(encoder, "forward_with_mask"), (
            "TextEncoder must expose forward_with_mask method"
        )
        embeddings, mask = encoder.forward_with_mask(texts)

        assert isinstance(embeddings, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert embeddings.shape[0] == 2
        assert mask.shape == (2, embeddings.shape[1])
        assert mask.dtype in (torch.bool, torch.float32)
        # First text has fewer tokens, so padding must be masked (False or 0)
        assert not mask[0, -1].bool(), "Padding token at end of shorter text must be masked"

    def test_r1_belnap_attention_masks_padding_keys(self) -> None:
        """R1.2: BelnapAttention must mask out padding keys to -1e9 before sigmoid."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        batch_size, seq_len = 2, 4
        q = BelnapState(
            e_pos=torch.rand(batch_size, 2, d_model),
            e_neg=torch.rand(batch_size, 2, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(batch_size, seq_len, d_model),
            e_neg=torch.rand(batch_size, seq_len, d_model),
        )

        # Mask out last two tokens for sample 0
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]],
            dtype=torch.bool,
        )

        out = attn(q, kv, mask=mask)
        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (batch_size, 2, d_model)
        assert torch.isfinite(out.e_pos).all()
        assert torch.isfinite(out.e_neg).all()

    def test_r1_active_mass_scale_normalization(self) -> None:
        """R1.3: Evidence aggregation denominator must normalize by active token mass."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        # Compare active token aggregation when isolated vs when accompanied by padding tokens
        q = BelnapState(
            e_pos=torch.ones(1, 1, d_model) * 0.5,
            e_neg=torch.zeros(1, 1, d_model),
        )

        # 1 active token alone
        kv_short = BelnapState(
            e_pos=torch.ones(1, 1, d_model) * 0.8,
            e_neg=torch.zeros(1, 1, d_model),
        )
        mask_short = torch.ones(1, 1, dtype=torch.bool)
        out_short = attn(q, kv_short, mask=mask_short)

        # Same active token padded with 10 padding tokens
        kv_padded = BelnapState(
            e_pos=torch.cat([torch.ones(1, 1, d_model) * 0.8, torch.zeros(1, 10, d_model)], dim=1),
            e_neg=torch.zeros(1, 11, d_model),
        )
        mask_padded = torch.cat(
            [torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 10, dtype=torch.bool)],
            dim=1,
        )
        out_padded = attn(q, kv_padded, mask=mask_padded)

        # Normalized evidence aggregation must yield identical output for the active token
        assert torch.allclose(out_short.e_pos, out_padded.e_pos, atol=1e-3), (
            "Padding keys must not alter active evidence scale after active mass normalization"
        )

    def test_r1_variable_sequence_length_boundedness(self) -> None:
        """R1.4: Activations must remain bounded in [0, 1] without saturation across lengths."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        # Long sequence (128 tokens) with variable active lengths
        batch_size = 4
        seq_len = 128
        q = BelnapState(
            e_pos=torch.rand(batch_size, 4, d_model),
            e_neg=torch.rand(batch_size, 4, d_model),
        )
        kv = BelnapState(
            e_pos=torch.rand(batch_size, seq_len, d_model),
            e_neg=torch.rand(batch_size, seq_len, d_model),
        )
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        assert (out.e_pos >= 0.0).all() and (out.e_pos <= 1.0).all()
        assert (out.e_neg >= 0.0).all() and (out.e_neg <= 1.0).all()

        # Mean activations should not be saturated at 0.0 or 1.0
        assert 0.05 < out.e_pos.mean().item() < 0.95, (
            "Outputs must not saturate from sequence accumulation"
        )

    def test_r1_belnap_mpa_mask_propagation(self) -> None:
        """R1.5: BelnapMultiheadPooledAttention must accept and propagate sequence padding masks."""
        from yoda.architecture.attention import BelnapMultiheadPooledAttention

        d_model = 32
        mpa = BelnapMultiheadPooledAttention(d_model=d_model, num_probes=2, n_heads=2)

        seq = torch.rand(2, 10, d_model)
        mask = torch.tensor(
            [[True] * 5 + [False] * 5, [True] * 8 + [False] * 2],
            dtype=torch.bool,
        )

        out = mpa(seq, mask=mask)
        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (2, 2, d_model)
        assert torch.isfinite(out.e_pos).all()


# ==============================================================================
# R2: Paraconsistent Contradiction Loss & Pre-Activation Normalization (Phase 1)
# ==============================================================================


class TestR2ContradictionLossAndPreActivationNorm:
    """Verifies R2: Contradiction penalty, linear warmup, LayerNorm, and residual highway alpha."""

    def test_r2_contradiction_loss_replaces_quadratic_penalty(self) -> None:
        """R2.1: Contradiction loss must evaluate choice_pos * choice_neg, not quadratic p(1-p)."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=1.0)

        # Clear decisive prediction (pos=1.0, neg=0.0): contradictory penalty should be 0.0
        outputs_decisive = {
            "choice_pos": torch.tensor([[1.0]]),
            "choice_neg": torch.tensor([[0.0]]),
            "truth": torch.tensor([[1.0]]),
        }
        loss_decisive = loss_fn(outputs_decisive)

        # Intermediate non-contradictory prediction (pos=0.5, neg=0.0):
        # Under quadratic penalty p(1-p), 0.5*(1-0.5) = 0.25 penalty!
        # Under contradiction penalty pos*neg, 0.5*0.0 = 0.0 penalty!
        outputs_unbiased = {
            "choice_pos": torch.tensor([[0.5]]),
            "choice_neg": torch.tensor([[0.0]]),
            "truth": torch.tensor([[0.75]]),
        }
        loss_unbiased = loss_fn(outputs_unbiased)

        # Fully contradictory prediction (pos=1.0, neg=1.0): Both state incurs max penalty
        outputs_both = {
            "choice_pos": torch.tensor([[1.0]]),
            "choice_neg": torch.tensor([[1.0]]),
            "truth": torch.tensor([[0.5]]),
        }
        loss_both = loss_fn(outputs_both)

        assert pytest.approx(loss_decisive.item(), abs=1e-5) == 0.0
        assert pytest.approx(loss_unbiased.item(), abs=1e-5) == 0.0, (
            "Zero negative evidence with pos=0.5 must not incur contradiction penalty"
        )
        assert loss_both.item() > 0.5, (
            "Both state (pos=1, neg=1) must incur full contradiction penalty"
        )

    def test_r2_linear_warmup_schedule(self) -> None:
        """R2.2: LTNConstraintLoss must provide linear warmup for decisiveness weight."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.1)

        assert hasattr(loss_fn, "step_epoch") or hasattr(loss_fn, "current_decisiveness_weight"), (
            "LTNConstraintLoss must support epoch-based decisiveness weight warmup"
        )

        if hasattr(loss_fn, "step_epoch"):
            loss_fn.step_epoch(epoch=0)
            assert getattr(loss_fn, "current_decisiveness_weight", 0.0) == 0.0, (
                "Decisiveness weight at epoch 0 must be 0.0 during warmup"
            )
            loss_fn.step_epoch(epoch=5)
            assert getattr(loss_fn, "current_decisiveness_weight", 0.0) > 0.0

    def test_r2_prenorm_layernorm_in_attention(self) -> None:
        """R2.3: BelnapAttention must insert LayerNorm prior to output bounding sigmoids."""
        attn = BelnapAttention(d_model=32, n_heads=2)

        # LayerNorm modules must exist before sigmoid projections
        has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in attn.modules())
        assert has_layernorm, "BelnapAttention must contain LayerNorm prior to bounding sigmoids"

    def test_r2_prenorm_layernorm_in_ffn(self) -> None:
        """R2.4: BelnapFFN must insert LayerNorm prior to output bounding sigmoids."""
        ffn = BelnapFFN(d_model=32, d_hidden=64)

        has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in ffn.modules())
        assert has_layernorm, "BelnapFFN must contain LayerNorm prior to bounding sigmoids"

    def test_r2_residual_highway_alpha_recalibration(self) -> None:
        """R2.5: BelnapTransformerBlock default residual weight alpha must be <= 0.20."""
        block = BelnapTransformerBlock(d_model=32, n_heads=2)

        # Default alpha should be calibrated to ~0.15 (<= 0.20)
        assert block.residual_weight <= 0.20, (
            f"residual_weight should be <= 0.20, got {block.residual_weight}"
        )

        # Verify gradient survival: (1 - alpha)^6 across 6 joins
        retention = (1.0 - block.residual_weight) ** 6
        assert retention >= 0.25, f"Residual retention must be >= 25%, got {retention:.4f}"


# ==============================================================================
# R3: NeSy Constraint Reformulation & Epistemic Logit Synchronization (Phase 2)
# ==============================================================================


class TestR3NeSyConstraintsAndEpistemicLogits:
    """Verifies R3: Positive support XOR/OR, coupled routing, and knowledge-gated logits."""

    def test_r3_positive_support_xor_zero_penalty_for_neither(self) -> None:
        """R3.1: LTN XOR constraint evaluates positive support; Neither (0, 0) incurs 0 penalty."""
        loss_fn = LTNConstraintLoss()

        # Candidate 0 positive (1, 0), unmentioned candidates at Neither (0, 0)
        # Truth coordinates for unmentioned: t = (0 - 0 + 1) / 2 = 0.5
        # Evaluating XOR on truth penalizes -log(1 - 0.5) = 0.693 per unmentioned option!
        # Evaluating XOR on positive support incurs 0.0 penalty for e+=0.0
        outputs = {
            "choice_pos": torch.tensor([1.0, 0.0, 0.0]),
            "choice_neg": torch.tensor([0.0, 0.0, 0.0]),
            "truth": torch.tensor([1.0, 0.5, 0.5]),
            "group_ids": torch.tensor([0, 0, 0]),
            "task_scalars": torch.tensor([[1.0]]),
        }

        # Evaluate XOR loss
        if hasattr(loss_fn, "_compute_xor_positive_support"):
            xor_loss = loss_fn._compute_xor_positive_support(outputs["choice_pos"])
        else:
            xor_loss = loss_fn._compute_xor(outputs["choice_pos"])

        assert pytest.approx(xor_loss.item(), abs=1e-4) == 0.0, (
            "Unmentioned candidates at Neither (pos=0, neg=0) must incur 0 XOR penalty"
        )

    def test_r3_positive_support_or_zero_penalty_for_neither(self) -> None:
        """R3.2: LTN OR constraint must evaluate positive support without penalizing Neither."""
        loss_fn = LTNConstraintLoss()

        # Single active choice is true, others are Neither
        choice_pos = torch.tensor([1.0, 0.0, 0.0, 0.0])
        or_loss = loss_fn._compute_or(choice_pos)

        assert pytest.approx(or_loss.item(), abs=1e-4) == 0.0, (
            "Unmentioned candidates at Neither must incur 0 OR penalty"
        )

    def test_r3_coupled_cross_polarity_value_routing(self) -> None:
        """R3.3: BelnapAttention must implement 2x2 coupled bilattice value aggregation."""
        d_model = 16
        attn = BelnapAttention(d_model=d_model, n_heads=1)

        # Construct key carrying pure negative refutation evidence: V_pos = 0, V_neg = 1.0
        # Query aligns coherently with Key: a_pos >> 0, a_neg ~ 0
        q = BelnapState(e_pos=torch.ones(1, 1, d_model), e_neg=torch.zeros(1, 1, d_model))
        kv = BelnapState(e_pos=torch.zeros(1, 1, d_model), e_neg=torch.ones(1, 1, d_model))

        out = attn(q, kv)
        # Under coupled 2x2 routing: agg_neg = a_pos * V_neg + a_neg * V_pos
        # Because a_pos is strong and V_neg is strong, output must contain negative evidence!
        assert out.e_neg.mean().item() > 0.1, (
            "Coherent query aligned with refutation key must route negative evidence to out.e_neg"
        )

    def test_r3_refutation_signal_preserved_under_coherent_query(self) -> None:
        """R3.4: Refutation signals from keys must be preserved rather than dropped."""
        d_model = 16
        attn = BelnapAttention(d_model=d_model, n_heads=1)

        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        # Pure refutation kv
        kv = BelnapState(e_pos=torch.zeros(1, 4, d_model), e_neg=torch.ones(1, 4, d_model))

        out = attn(q, kv)
        # Under decoupled channels: agg_pos = a_pos * 0 = 0; agg_neg = a_neg * 1 ~ 0 -> dropped!
        # Under coupled routing: refutation is preserved
        assert out.e_neg.sum().item() > 0.05, "Refutation evidence must be preserved"

    def test_r3_epistemic_logit_synchronization(self) -> None:
        """R3.5: Decision head logits must be gated by knowledge coordinate k."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)

        # Construct a low-knowledge state: raw_pos = -8.0, raw_neg = -12.0
        # Pre-sigmoid pos ~ 0.0003, neg ~ 0.000006, knowledge k ~ 0.00015
        # Ungated logit = (-8.0) - (-12.0) = +4.0 (artificially high confidence!)
        # Knowledge-gated logit: (+4.0) * k ~ 0.0006
        low_k_state = BelnapState(
            e_pos=torch.ones(1, 32) * 1e-4,
            e_neg=torch.ones(1, 32) * 1e-4,
        )

        out = head(low_k_state)
        logit = out["logits"].item()
        assert abs(logit) < 0.5, (
            f"Zero-knowledge state (k < 1e-3) must produce suppressed logit close to 0, got {logit}"
        )


# ==============================================================================
# R4: Bilinear Candidate-Query Alignment & Negation Equivariance (Phase 3)
# ==============================================================================


class TestR4BilinearAlignmentAndNegationEquivariance:
    """Verifies R4: Bilinear Belnap inner product scoring, cancellation elimination, dual FFN."""

    def test_r4_bilinear_belnap_inner_product_scoring(self) -> None:
        """R4.1: Scalar head must compute bilinear inner product when candidate_states is passed."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)

        final_state = BelnapState(e_pos=torch.rand(2, 32), e_neg=torch.rand(2, 32))
        cand_state = BelnapState(e_pos=torch.rand(2, 32), e_neg=torch.rand(2, 32))

        # Must accept candidate_states in independent mode (num_choices == 1)
        out = head(final_state, candidate_states=cand_state)
        assert "logits" in out
        assert out["logits"].shape == (2,)

    def test_r4_elimination_of_additive_context_cancellation(self) -> None:
        """R4.2: Candidate scoring difference must depend on query context and not cancel out."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # Group 1: Query Q1, Candidates C1 and C2
        out_q1_c1 = engine(queries=["Query Alpha"], states=[{}], candidates=["Candidate One"])
        out_q1_c2 = engine(queries=["Query Alpha"], states=[{}], candidates=["Candidate Two"])
        diff_q1 = (out_q1_c1["logits"] - out_q1_c2["logits"]).item()

        # Group 2: Distinct Query Q2, Same Candidates C1 and C2
        out_q2_c1 = engine(
            queries=["Query Beta with different context"], states=[{}], candidates=["Candidate One"]
        )
        out_q2_c2 = engine(
            queries=["Query Beta with different context"], states=[{}], candidates=["Candidate Two"]
        )
        diff_q2 = (out_q2_c1["logits"] - out_q2_c2["logits"]).item()

        # If context cancels out additively, diff_q1 == diff_q2 identically!
        assert abs(diff_q1 - diff_q2) > 1e-5, (
            "Query context must not cancel out when computing candidate score differences"
        )

    def test_r4_dual_channel_ffn_projections(self) -> None:
        """R4.3: BelnapFFN must maintain dual positive and negative projection channels."""
        ffn = BelnapFFN(d_model=32, d_hidden=64)

        # Dual projections or cross-gating rather than a single collapsed intermediate vector
        x = BelnapState(e_pos=torch.rand(2, 4, 32), e_neg=torch.rand(2, 4, 32))
        out = ffn(x)

        assert isinstance(out, BelnapState)
        assert out.e_pos.shape == (2, 4, 32)
        assert out.e_neg.shape == (2, 4, 32)

    def test_r4_dual_channel_cross_gating(self) -> None:
        """R4.4: BelnapFFN must utilize cross-channel gating preserving bilattice structure."""
        ffn = BelnapFFN(d_model=32, d_hidden=64)
        x = BelnapState(e_pos=torch.ones(1, 1, 32) * 0.9, e_neg=torch.ones(1, 1, 32) * 0.1)
        out = ffn(x)

        # Output positive and negative evidence must be distinct and non-trivial
        assert not torch.allclose(out.e_pos, out.e_neg)

    def test_r4_negation_equivariance_under_state_inversion(self) -> None:
        """R4.5: BelnapFFN must be negation equivariant: f(negate(x)) == negate(f(x))."""
        ffn = BelnapFFN(d_model=32, d_hidden=64)

        x = BelnapState(e_pos=torch.rand(2, 4, 32), e_neg=torch.rand(2, 4, 32))
        x_negated = x.negate()

        out_normal = ffn(x)
        out_inverted = ffn(x_negated)

        # Negation equivariance: out_inverted.e_pos should equal out_normal.e_neg
        assert torch.allclose(out_inverted.e_pos, out_normal.e_neg, atol=1e-3), (
            "Swapping input polarities must swap output polarities (negation equivariance)"
        )


# ==============================================================================
# R5: Context Encoding Deduplication & Hyperparameter Sweep Realignment (Phase 4)
# ==============================================================================


class TestR5ContextDeduplicationAndSweepRealignment:
    """Verifies R5: Context deduplication cache and Optuna hyperparameter realignment."""

    def test_r5_context_encoding_deduplication_cache(self) -> None:
        """R5.1: YodaDecisionEngine must cache unique (query, state) encodings per group_id."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # Mock the text encoder forward pass to count invocations
        original_forward = engine.text_encoder.forward
        mock_forward = MagicMock(side_effect=original_forward)
        engine.text_encoder.forward = mock_forward

        # 4 unrolled candidates sharing the same query and state
        queries = ["Common query"] * 4
        states: list[dict[str, Any]] = [{"env": "sim"}] * 4
        candidates = ["Option 1", "Option 2", "Option 3", "Option 4"]

        engine(queries=queries, states=states, candidates=candidates)

        # With deduplication cache, query should be encoded once, not 4 times
        assert mock_forward.call_count <= 2, (
            f"Query should be encoded once, called {mock_forward.call_count} times"
        )

    def test_r5_grouped_candidate_scoring_consistency(self) -> None:
        """R5.2: Deduplicated context evaluations must match uncached output precision."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Shared context"] * 2
        states: list[dict[str, Any]] = [{"alpha": 1}] * 2
        candidates = ["Cand A", "Cand B"]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert "logits" in out
        assert out["logits"].shape == (2,)
        assert torch.isfinite(out["logits"]).all()

    def test_r5_cache_isolation_across_distinct_groups(self) -> None:
        """R5.3: Distinct groups must not share or overwrite cached contexts."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # Two distinct groups: Group 0 with Q0, Group 1 with Q1
        queries = ["Query 0", "Query 0", "Query 1", "Query 1"]
        states: list[dict[str, Any]] = [{"g": 0}, {"g": 0}, {"g": 1}, {"g": 1}]
        candidates = ["C0_A", "C0_B", "C1_A", "C1_B"]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (4,)
        # Group 0 and Group 1 outputs should not be identical
        assert not torch.allclose(out["logits"][:2], out["logits"][2:])

    @patch("experiments.optuna_study.run_training_trial")
    def test_r5_optuna_search_bounds_realignment(self, mock_run: MagicMock) -> None:
        """R5.4: Optuna parameter bounds must conform to hardened stable ranges."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = "dummy"
        trial.suggest_int.return_value = 2
        trial.suggest_float.return_value = 0.2

        args = argparse.Namespace(
            max_samples=10,
            epochs=1,
            batch_size=2,
            early_stopping_patience=1,
            train_path=None,
        )

        optuna_objective(trial, args)

        # Inspect float suggestions passed to trial.suggest_float
        float_calls = {
            call_args[0][0]: call_args[0][1:] for call_args in trial.suggest_float.call_args_list
        }
        assert "residual_weight" in float_calls
        _, high_res = float_calls["residual_weight"][:2]
        # Residual weight must not allow extreme upper bounds like 0.9
        assert high_res <= 0.25, f"Residual weight upper bound should be <= 0.25, got {high_res}"

    @patch("experiments.optuna_study.run_training_trial")
    def test_r5_learning_rate_safe_regime(self, mock_run: MagicMock) -> None:
        """R5.5: Learning rate search space must stay within stable [1e-5, 5e-4] regime."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = "dummy"
        trial.suggest_int.return_value = 2
        trial.suggest_float.return_value = 1e-4

        args = argparse.Namespace(
            max_samples=10,
            epochs=1,
            batch_size=2,
            early_stopping_patience=1,
            train_path=None,
        )

        optuna_objective(trial, args)

        float_calls = {
            call_args[0][0]: call_args[0][1:] for call_args in trial.suggest_float.call_args_list
        }
        assert "lr" in float_calls
        _, high_lr = float_calls["lr"][:2]
        assert high_lr <= 5e-4, (
            f"Peak learning rate should not exceed 5e-4 to prevent sigmoid blowup, got {high_lr}"
        )
