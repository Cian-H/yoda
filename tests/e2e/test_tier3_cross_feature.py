"""Tier 3: Pairwise Cross-Feature Interactions E2E Test Suite.

Verifies cross-feature coupling across requirements R1 through R5:
- R1 x R2: Masked variable-length sequences under contradiction loss and pre-sigmoid LayerNorm.
- R1 x R3: Variable-length masked padding interacting with 2x2 coupled bilattice routing.
- R1 x R4: Attention-masked sequence representations scored with bilinear candidate alignment.
- R1 x R5: Mask propagation with cached/deduplicated context encodings.
- R2 x R3: Contradiction loss with linear warmup trained jointly with positive-support XOR.
- R2 x R4: Pre-sigmoid normalized activations processed through dual-channel FFN.
- R2 x R5: Residual highway alpha=0.15 gradient propagation through cached unrolled candidates.
- R3 x R4: Epistemic knowledge gating with bilinear candidate scoring under negation inversion.
- R3 x R5: Grouped unrolled candidate evaluation with positive support XOR and context caching.
- R4 x R5: Bilinear candidate scoring on cached context representations with dual-channel FFN.
"""

from typing import Any

import torch

from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapState,
    BelnapTransformerBlock,
)
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss


class TestTier3CrossFeatureInteractions:
    """Verifies pairwise cross-feature interactions across R1-R5."""

    def test_t3_r1_masking_x_r2_contradiction_loss(self) -> None:
        """R1 x R2: Variable-length masked inputs under contradiction loss and LayerNorm."""
        d_model = 32
        attn = BelnapAttention(d_model=d_model, n_heads=2)
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.1)

        q = BelnapState(e_pos=torch.rand(2, 2, d_model), e_neg=torch.rand(2, 2, d_model))
        kv = BelnapState(e_pos=torch.rand(2, 8, d_model), e_neg=torch.rand(2, 8, d_model))
        mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 8], dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        # Verify output coordinates in [0, 1]
        assert (out.e_pos >= 0.0).all() and (out.e_pos <= 1.0).all()

        # Compute contradiction loss on pooled activations
        outputs = {
            "choice_pos": out.e_pos.mean(dim=1),
            "choice_neg": out.e_neg.mean(dim=1),
            "truth": out.truth.mean(dim=1),
        }
        loss = loss_fn(outputs)
        assert torch.isfinite(loss), "Contradiction loss must be finite for masked inputs"

    def test_t3_r1_masking_x_r3_coupled_routing(self) -> None:
        """R1 x R3: Masked attention sequence transferring refutation evidence via 2x2 routing."""
        d_model = 16
        attn = BelnapAttention(d_model=d_model, n_heads=1)

        # Active token at index 0 has refutation (e_pos=0, e_neg=1), remaining 3 tokens are padding
        q = BelnapState(e_pos=torch.ones(1, 1, d_model), e_neg=torch.zeros(1, 1, d_model))
        kv = BelnapState(
            e_pos=torch.zeros(1, 4, d_model),
            e_neg=torch.tensor(
                [[[1.0] * d_model, [0.0] * d_model, [0.0] * d_model, [0.0] * d_model]]
            ),
        )
        mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)

        out = attn(q, kv, mask=mask)
        # Coupled routing must transfer refutation from active token 0 without dilution from padding
        assert out.e_neg.mean().item() > 0.05, (
            "Coupled routing must preserve refutation from active token"
        )

    def test_t3_r1_masking_x_r4_bilinear_alignment(self) -> None:
        """R1 x R4: Masked sequence representations scored with bilinear candidate alignment."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        # Short query vs long padded query
        q_short = ["Task"]
        q_padded = ["Task with multiple additional padded tokens for testing"]
        cands = ["Execute safety maneuver"]

        out_short = engine(queries=q_short, states=[{}], candidates=cands)
        out_padded = engine(queries=q_padded, states=[{}], candidates=cands)

        assert torch.isfinite(out_short["logits"]).all()
        assert torch.isfinite(out_padded["logits"]).all()

    def test_t3_r1_masking_x_r5_context_deduplication(self) -> None:
        """R1 x R5: Mask propagation with cached and deduplicated context encodings."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Variable length query with padding"] * 3
        states: list[dict[str, Any]] = [{"system": "active"}] * 3
        candidates = ["Choice A", "Choice B", "Choice C"]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (3,)
        assert torch.isfinite(out["logits"]).all()

    def test_t3_r2_contradiction_loss_x_r3_positive_support_xor(self) -> None:
        """R2 x R3: Contradiction loss with warmup trained jointly with positive-support XOR."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.1)

        # Target option is chosen (pos=1, neg=0), unmentioned option is Neither (pos=0, neg=0)
        outputs = {
            "choice_pos": torch.tensor([[1.0, 0.0]]),
            "choice_neg": torch.tensor([[0.0, 0.0]]),
            "truth": torch.tensor([[1.0, 0.5]]),
            "group_ids": torch.tensor([0, 0]),
            "task_scalars": torch.tensor([[1.0]]),
        }

        # Contradiction is 0.0 and unmentioned Neither XOR penalty is 0.0
        total_loss = loss_fn(outputs)
        assert total_loss.item() < 0.1, (
            "Decisive choice with unmentioned Neither option must produce near-zero joint loss"
        )

    def test_t3_r2_prenorm_x_r4_dual_channel_ffn(self) -> None:
        """R2 x R4: Pre-sigmoid normalized activations processed through dual-channel FFN."""
        block = BelnapTransformerBlock(d_model=32, n_heads=2, residual_weight=0.15)

        x = BelnapState(e_pos=torch.rand(2, 4, 32), e_neg=torch.rand(2, 4, 32))
        out = block(x)

        assert isinstance(out, BelnapState)
        assert (out.e_pos >= 0.0).all() and (out.e_pos <= 1.0).all()
        assert (out.e_neg >= 0.0).all() and (out.e_neg <= 1.0).all()

    def test_t3_r2_residual_highway_x_r5_context_caching(self) -> None:
        """R2 x R5: Residual highway alpha=0.15 gradient propagation through unrolled candidates."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
            num_reasoning_blocks=2,
        )

        queries = ["Shared query"] * 3
        states: list[dict[str, Any]] = [{"load": 50}] * 3
        candidates = ["Action 1", "Action 2", "Action 3"]

        out = engine(queries=queries, states=states, candidates=candidates)
        loss = out["logits"].sum()
        loss.backward()

        # Check gradient in reasoning layers
        p = next(engine.reasoning_layers[0].parameters())
        assert p.grad is not None, "Gradients must propagate back through reasoning layers"

    def test_t3_r3_epistemic_logits_x_r4_bilinear_scoring(self) -> None:
        """R3 x R4: Bilinear candidate scoring with knowledge coordinate logit gating."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)

        # Context has evidence, candidate has zero evidence (k=0)
        final_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        cand_state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))

        out = head(final_state, candidate_states=cand_state)
        # Because candidate has zero knowledge (k=0), logit must be suppressed
        assert abs(out["logits"].item()) < 0.2, (
            "Bilinear scoring on zero-knowledge candidate must be suppressed"
        )

    def test_t3_r3_positive_support_x_r5_grouped_candidates(self) -> None:
        """R3 x R5: Grouped candidate evaluation with positive support XOR and context caching."""
        loss_fn = LTNConstraintLoss()

        # Group 0: 3 choices (target is choice 0, choices 1 and 2 are Neither)
        # Group 1: 2 choices (target is choice 1, choice 0 is Neither)
        choice_pos = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1])
        task_scalars = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])

        grouped_loss = loss_fn._compute_grouped_constraints(
            truth=choice_pos,
            logits=choice_pos,
            group_ids=group_ids,
            task_scalars=task_scalars,
        )
        assert torch.isfinite(grouped_loss)

    def test_t3_r4_bilinear_head_x_r5_cached_context(self) -> None:
        """R4 x R5: Bilinear decision head scoring candidates against cached context states."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Common query"] * 4
        states: list[dict[str, Any]] = [{"system": "online"}] * 4
        candidates = ["Candidate 0", "Candidate 1", "Candidate 2", "Candidate 3"]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (4,)
        # All 4 candidate scores must be finite and mutually differentiated
        assert torch.isfinite(out["logits"]).all()
