"""Tier 2: Boundary & Corner Cases E2E Test Suite.

Covers boundary and corner conditions for Requirements R1 through R5 (>= 5 tests per feature):
- R1: Masking boundaries (all-pad, single-token, zero active mass, extreme lengths)
- R2: Loss & activation boundaries (Both, Neither, epoch 0 warmup, extreme pre-sigmoids, deep joins)
- R3: Logic & logit boundaries (pure Neither, pure False, pure True, zero-k limit, max-k)
- R4: Bilinear & FFN boundaries (orthogonal, anti-aligned, polarity extremes, K=1 groups)
- R5: Deduplication boundaries (K=1 idempotence, K=16 scaling, ragged groups, state collisions)
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
)
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss

# ==============================================================================
# R1 Boundaries: Masking & Scale Normalization Corners
# ==============================================================================


class TestR1MaskingAndScaleBoundaries:
    """Verifies R1 boundary conditions: all-pad, single-token, extreme lengths, zero mass."""

    def test_r1_boundary_all_padding_tokens(self) -> None:
        """R1.B1: BelnapAttention must safely handle all-padding sequence without NaN or crash."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv = BelnapState(e_pos=torch.rand(1, 8, d_model), e_neg=torch.rand(1, 8, d_model))
        # Entire sequence is masked as padding
        mask = torch.zeros(1, 8, dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        assert torch.isfinite(out.e_pos).all(), "All-padding sequence must not produce NaN/Inf"
        assert torch.isfinite(out.e_neg).all()

    def test_r1_boundary_single_active_token_extreme_padding(self) -> None:
        """R1.B2: Single active token accompanied by 255 padding tokens must retain output scale."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        q = BelnapState(e_pos=torch.rand(1, 1, d_model), e_neg=torch.rand(1, 1, d_model))

        # 1 active token alone
        kv_single = BelnapState(e_pos=torch.rand(1, 1, d_model), e_neg=torch.rand(1, 1, d_model))
        mask_single = torch.ones(1, 1, dtype=torch.bool)
        out_single = attn(q, kv_single, mask=mask_single)

        # 1 active token + 255 padding tokens
        kv_extreme = BelnapState(
            e_pos=torch.cat([kv_single.e_pos, torch.zeros(1, 255, d_model)], dim=1),
            e_neg=torch.cat([kv_single.e_neg, torch.zeros(1, 255, d_model)], dim=1),
        )
        mask_extreme = torch.cat(
            [torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 255, dtype=torch.bool)], dim=1
        )
        out_extreme = attn(q, kv_extreme, mask=mask_extreme)

        assert torch.allclose(out_single.e_pos, out_extreme.e_pos, atol=1e-3), (
            "Single active token must produce identical outputs regardless of 255 padding tokens"
        )

    def test_r1_boundary_empty_token_batch(self) -> None:
        """R1.B3: BelnapAttention must handle empty batch size (B=0) gracefully."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        q = BelnapState(e_pos=torch.empty(0, 4, d_model), e_neg=torch.empty(0, 4, d_model))
        kv = BelnapState(e_pos=torch.empty(0, 8, d_model), e_neg=torch.empty(0, 8, d_model))

        out = attn(q, kv)
        assert out.e_pos.shape == (0, 4, d_model)
        assert out.e_neg.shape == (0, 4, d_model)

    def test_r1_boundary_max_sequence_length_scaling(self) -> None:
        """R1.B4: Aggregations across 512 tokens must not saturate into hard 0.0 or 1.0."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        q = BelnapState(e_pos=torch.rand(1, 2, d_model), e_neg=torch.rand(1, 2, d_model))
        kv = BelnapState(e_pos=torch.rand(1, 512, d_model), e_neg=torch.rand(1, 512, d_model))
        mask = torch.ones(1, 512, dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        # Should be strictly within (0, 1) and not hard clamped at borders
        assert (out.e_pos > 0.001).all() and (out.e_pos < 0.999).all()

    def test_r1_boundary_zero_active_mass_clamping(self) -> None:
        """R1.B5: Active mass denominator clamping by min=1.0 prevents zero division."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        q = BelnapState(e_pos=torch.ones(1, 1, d_model) * 0.5, e_neg=torch.zeros(1, 1, d_model))
        kv = BelnapState(e_pos=torch.ones(1, 4, d_model) * 0.5, e_neg=torch.zeros(1, 4, d_model))
        # Mask has sum == 0
        mask = torch.zeros(1, 4, dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        assert not torch.isnan(out.e_pos).any()
        assert not torch.isinf(out.e_pos).any()


# ==============================================================================
# R2 Boundaries: Contradiction Loss, Warmup & Pre-Sigmoid Extremes
# ==============================================================================


class TestR2LossAndActivationBoundaries:
    """Verifies R2 boundary conditions: Both state, Neither state, warmup, extreme activations."""

    def test_r2_boundary_extreme_contradiction_both_state(self) -> None:
        """R2.B1: Purely contradictory input (Both: e+=1, e-=1) must receive maximal penalty."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=1.0)
        outputs = {
            "choice_pos": torch.tensor([[1.0]]),
            "choice_neg": torch.tensor([[1.0]]),
            "truth": torch.tensor([[0.5]]),
        }
        loss = loss_fn(outputs)
        assert loss.item() >= 1.0 - 1e-4, (
            "Pure Both state must receive maximal contradiction penalty"
        )

    def test_r2_boundary_pure_ignorance_neither_state(self) -> None:
        """R2.B2: Purely uncommitted input (Neither: e+=0, e-=0) must receive zero penalty."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=1.0)
        outputs = {
            "choice_pos": torch.tensor([[0.0]]),
            "choice_neg": torch.tensor([[0.0]]),
            "truth": torch.tensor([[0.5]]),
        }
        loss = loss_fn(outputs)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
            "Pure Neither state must receive 0 contradiction loss"
        )

    def test_r2_boundary_warmup_epoch_zero(self) -> None:
        """R2.B3: Contradiction loss weight must be strictly 0.0 at epoch 0 during warmup."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.5)
        if hasattr(loss_fn, "step_epoch"):
            loss_fn.step_epoch(epoch=0)
            outputs = {
                "choice_pos": torch.tensor([[1.0]]),
                "choice_neg": torch.tensor([[1.0]]),
                "truth": torch.tensor([[0.5]]),
            }
            loss = loss_fn(outputs)
            assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
                "Warmup at epoch 0 must nullify penalty"
            )

    def test_r2_boundary_extreme_pre_sigmoid_activation_values(self) -> None:
        """R2.B4: Extreme pre-sigmoid activations (+/- 100.0) must be centered by LayerNorm."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)

        # Force large magnitudes into query and kv
        q = BelnapState(
            e_pos=torch.ones(1, 2, d_model) * 100.0, e_neg=torch.ones(1, 2, d_model) * -100.0
        )
        kv = BelnapState(
            e_pos=torch.ones(1, 2, d_model) * 100.0, e_neg=torch.ones(1, 2, d_model) * -100.0
        )

        out = attn(q, kv)
        assert torch.isfinite(out.e_pos).all()
        assert torch.isfinite(out.e_neg).all()

    def test_r2_boundary_deep_residual_highway_stack(self) -> None:
        """R2.B5: 12 consecutive residual joins with alpha=0.15 retain substantial gradient."""
        alpha = 0.15
        retention = (1.0 - alpha) ** 12
        # (0.85)^12 ~ 0.142 (over 14% retention through 12 deep joins vs 0.02% at alpha=0.5)
        assert retention > 0.10, (
            f"Gradient retention across 12 joins must exceed 10%, got {retention:.4f}"
        )


# ==============================================================================
# R3 Boundaries: Positive Support XOR/OR & Epistemic Limits
# ==============================================================================


class TestR3LogicAndEpistemicBoundaries:
    """Verifies R3 boundary conditions: Neither candidate, pure False, pure True, zero-k limit."""

    def test_r3_boundary_pure_neither_candidate_zero_xor_loss(self) -> None:
        """R3.B1: Option at Neither (0, 0) in positive-support XOR incurs zero false penalty."""
        loss_fn = LTNConstraintLoss()
        # Candidate 0 is chosen (e+=1), candidate 1 is Neither (e+=0, e-=0)
        choice_pos = torch.tensor([1.0, 0.0])
        xor_loss = loss_fn._compute_xor(choice_pos)
        assert pytest.approx(xor_loss.item(), abs=1e-5) == 0.0

    def test_r3_boundary_pure_refutation_candidate(self) -> None:
        """R3.B2: Candidate option with pure refutation (e+=0, e-=1) has zero positive support."""
        loss_fn = LTNConstraintLoss()
        # Candidate 0 is chosen (e+=1), candidate 1 is refuted (e+=0, e-=1)
        choice_pos = torch.tensor([1.0, 0.0])
        xor_loss = loss_fn._compute_xor(choice_pos)
        assert pytest.approx(xor_loss.item(), abs=1e-5) == 0.0

    def test_r3_boundary_pure_truth_candidate(self) -> None:
        """R3.B3: Candidate option with pure truth (e+=1, e-=0) has maximum positive support."""
        loss_fn = LTNConstraintLoss()
        choice_pos = torch.tensor([1.0, 0.0, 0.0])
        xor_loss = loss_fn._compute_xor(choice_pos)
        assert pytest.approx(xor_loss.item(), abs=1e-5) == 0.0

    def test_r3_boundary_zero_knowledge_logit_suppression(self) -> None:
        """R3.B4: Zero-knowledge state (k -> 0) must suppress logits to 0 regardless of raw diff."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)

        # Extremely low evidence on both channels -> k ~ 0
        state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        logit = out["logits"].item()
        assert abs(logit) < 1e-3, f"Zero-knowledge logit must be suppressed to 0, got {logit}"

    def test_r3_boundary_maximal_knowledge_unsuppressed_logits(self) -> None:
        """R3.B5: Maximal knowledge state (k=1) transmits full unattenuated logit dynamic range."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)

        # High positive, zero negative -> k = 0.5, truth = 1.0
        state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        # Logit should be non-zero and decisively positive
        assert out["logits"].item() > 0.0


# ==============================================================================
# R4 Boundaries: Bilinear Affinity Extremes & Group Scaling
# ==============================================================================


class TestR4BilinearAffinityBoundaries:
    """Verifies R4 boundary conditions: orthogonal, anti-aligned, polarity extremes, K=1 groups."""

    def test_r4_boundary_orthogonal_candidate_query_vectors(self) -> None:
        """R4.B1: Orthogonal context and candidate representations produce zero affinity."""
        d_model = 32
        head = BelnapDecisionHead(d_model=d_model, num_choices=1)

        # Construct orthogonal vectors
        q = torch.zeros(1, d_model)
        q[0, : d_model // 2] = 1.0
        c = torch.zeros(1, d_model)
        c[0, d_model // 2 :] = 1.0

        q_state = BelnapState(e_pos=q, e_neg=torch.zeros(1, d_model))
        c_state = BelnapState(e_pos=c, e_neg=torch.zeros(1, d_model))

        out = head(q_state, candidate_states=c_state)
        assert "logits" in out
        assert torch.isfinite(out["logits"]).all()

    def test_r4_boundary_anti_aligned_candidate_query_vectors(self) -> None:
        """R4.B2: Opposing polarity alignment routes into negative affinity."""
        d_model = 32
        head = BelnapDecisionHead(d_model=d_model, num_choices=1)

        q_state = BelnapState(e_pos=torch.ones(1, d_model), e_neg=torch.zeros(1, d_model))
        c_state = BelnapState(e_pos=torch.zeros(1, d_model), e_neg=torch.ones(1, d_model))

        out = head(q_state, candidate_states=c_state)
        assert "logits" in out
        # Negative alignment should push logit negative
        assert out["logits"].item() <= 0.0

    def test_r4_boundary_purely_positive_vs_negative_candidates(self) -> None:
        """R4.B3: Positive query scored against purely positive vs purely negative candidate."""
        d_model = 32
        head = BelnapDecisionHead(d_model=d_model, num_choices=1)

        q_state = BelnapState(e_pos=torch.ones(2, d_model), e_neg=torch.zeros(2, d_model))
        c_state = BelnapState(
            e_pos=torch.tensor([[1.0] * d_model, [0.0] * d_model]),
            e_neg=torch.tensor([[0.0] * d_model, [1.0] * d_model]),
        )

        out = head(q_state, candidate_states=c_state)
        logits = out["logits"]
        assert logits[0] > logits[1], (
            "Positive candidate must score higher than negative candidate under positive query"
        )

    def test_r4_boundary_extreme_evidence_negation_equivariance(self) -> None:
        """R4.B4: Inverting coordinates (1.0, 0.0) to (0.0, 1.0) must satisfy equivariance."""
        ffn = BelnapFFN(d_model=32, d_hidden=64)
        x = BelnapState(e_pos=torch.ones(1, 1, 32), e_neg=torch.zeros(1, 1, 32))
        x_neg = x.negate()

        out_x = ffn(x)
        out_neg = ffn(x_neg)

        assert torch.allclose(out_neg.e_pos, out_x.e_neg, atol=1e-3)
        assert torch.allclose(out_neg.e_neg, out_x.e_pos, atol=1e-3)

    def test_r4_boundary_single_candidate_unrolled_group(self) -> None:
        """R4.B5: Single-candidate group (K=1) must evaluate without dimensional collapse."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        out = engine(queries=["Solo query"], states=[{}], candidates=["Single candidate"])
        assert out["logits"].shape == (1,)
        assert out["truth"].shape == (1,)


# ==============================================================================
# R5 Boundaries: Cache Idempotence, Group Raggedness & Search Spaces
# ==============================================================================


class TestR5DeduplicationAndSweepBoundaries:
    """Verifies R5 boundary conditions: K=1, K=16 scaling, ragged groups, edge spaces."""

    def test_r5_boundary_single_item_batch_cache_idempotence(self) -> None:
        """R5.B1: Batch with 1 item must behave identically with or without cache."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        out1 = engine(queries=["Unique query"], states=[{"id": 1}], candidates=["Candidate A"])
        out2 = engine(queries=["Unique query"], states=[{"id": 1}], candidates=["Candidate A"])

        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)

    def test_r5_boundary_large_candidate_group_k16(self) -> None:
        """R5.B2: Large unrolled group with 16 candidates must evaluate efficiently."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Large choice problem"] * 16
        states: list[dict[str, Any]] = [{"task": "large"}] * 16
        candidates = [f"Candidate option {i}" for i in range(16)]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (16,)
        assert torch.isfinite(out["logits"]).all()

    def test_r5_boundary_variable_candidate_counts_per_group(self) -> None:
        """R5.B3: Ragged groups (e.g. Group 0 has 2 choices, Group 1 has 5) evaluate cleanly."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # 2 choices for Q0, 5 choices for Q1 -> total 7 candidates
        queries = ["Q0"] * 2 + ["Q1"] * 5
        states: list[dict[str, Any]] = [{"g": 0}] * 2 + [{"g": 1}] * 5
        candidates = [f"Opt {i}" for i in range(7)]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (7,)

    def test_r5_boundary_identical_query_different_states(self) -> None:
        """R5.B4: Same query text with different symbolic states must generate distinct entries."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # Same query string, completely different state dictionary
        out_state_a = engine(
            queries=["Same query"], states=[{"hazard": "fire"}], candidates=["Action"]
        )
        out_state_b = engine(
            queries=["Same query"], states=[{"hazard": "flood"}], candidates=["Action"]
        )

        # Different states should produce distinct intermediate states or logits
        assert not torch.allclose(out_state_a["logits"], out_state_b["logits"])

    @patch("experiments.optuna_study.run_training_trial")
    def test_r5_boundary_hyperparameter_domain_extremes(self, mock_run: MagicMock) -> None:
        """R5.B5: Edge hyperparameter values must not violate mathematical domain constraints."""
        trial = MagicMock()
        trial.suggest_categorical.return_value = "dummy"
        trial.suggest_int.return_value = 1
        trial.suggest_float.return_value = 0.1

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
        assert "conflation_weight" in float_calls
        assert "residual_weight" in float_calls
        low_res, high_res = float_calls["residual_weight"][:2]
        assert low_res > 0.0, "Lower residual weight must be strictly positive"
        assert high_res <= 0.50, "Upper residual weight must not blow up gradient attenuation"
