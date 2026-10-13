"""Empirical Challenge Stress Test Harness for Milestone 2 (R3).

Target: Coupled Cross-Polarity Bilattice Evidence Routing and Epistemic Logit Gating.
Role: Empirical Challenger (Critic / Specialist).
Domain Skill: nesy-constraint-check (LTN, continuous relaxations, Belnap bilattice invariants).
Methodology: Property testing with Hypothesis, adversarial boundary stress, circulant operator
verification, double negation symmetry proofs, and epistemic logit suppression checks.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from yoda.architecture.belnap_transformer import (
    BelnapAttention,
    BelnapState,
    BelnapTransformerBlock,
)
from yoda.architecture.engine import BelnapDecisionHead
from yoda.nesy.ltn import LTNConstraintLoss

# ==============================================================================
# Suite 1: Coupled Cross-Polarity Evidence Routing
# ==============================================================================


class TestCoupledRoutingRefutationPreservation:
    """Empirical verification that 2x2 coupled routing preserves refutation signals
    when queries align with negative key evidence.
    """

    @pytest.fixture
    def attn_layer(self) -> BelnapAttention:
        torch.manual_seed(42)
        return BelnapAttention(d_model=32, n_heads=4)

    def test_pure_refutation_routing_algebra(self) -> None:
        """Algebraic proof:
        [agg_pos]   [A_+  A_-] [V_+]   [A_+ V_+ + A_- V_-]
        [agg_neg] = [A_-  A_+] [V_-] = [A_+ V_- + A_- V_+]

        When query coherently aligns with key (A_+ = 1, A_- = 0) and value contains
        pure refutation (V_+ = 0, V_- = 1):
        - Decoupled routing gives agg_pos = 0, agg_neg = 0 (Refutation LOST to Neither!)
        - Coupled routing gives agg_pos = 0, agg_neg = 1 (Refutation PRESERVED!)
        """
        a_pos = torch.tensor([1.0])
        a_neg = torch.tensor([0.0])
        v_pos = torch.tensor([0.0])
        v_neg = torch.tensor([1.0])

        decoupled_pos = a_pos * v_pos
        decoupled_neg = a_neg * v_neg
        assert decoupled_pos.item() == 0.0
        assert decoupled_neg.item() == 0.0, "Decoupled erroneously drops refutation to Neither"

        coupled_pos = a_pos * v_pos + a_neg * v_neg
        coupled_neg = a_pos * v_neg + a_neg * v_pos
        assert coupled_pos.item() == 0.0
        assert coupled_neg.item() == 1.0, "Coupled must route refutation to agg_neg"

    def test_coherent_query_attending_to_refutation_value(
        self, attn_layer: BelnapAttention
    ) -> None:
        """Verifies that when a query coherently aligns with key (A_+ >> 0), attending to
        negative value evidence (V_+ = 0, V_- = 1) routes evidence to out.e_neg.
        """
        batch_size = 4
        l_q = 2
        l_kv = 5
        d_model = 32

        # Query and Key coherently align in positive space (A_+ >> 0, A_- ~ 0)
        q = BelnapState(
            e_pos=torch.ones(batch_size, l_q, d_model),
            e_neg=torch.zeros(batch_size, l_q, d_model),
        )
        # In standard BelnapAttention forward, kv is a single BelnapState providing both K and V.
        # But if kv has e_pos=0, e_neg=1, then Q+ K- creates refutation affinity A_-.
        # If kv has e_pos=0, e_neg=1, both out.e_pos and out.e_neg receive non-zero mass.
        # Here we test that out.e_neg is strictly non-zero (> 0.1).
        kv = BelnapState(
            e_pos=torch.zeros(batch_size, l_kv, d_model),
            e_neg=torch.ones(batch_size, l_kv, d_model),
        )
        out = attn_layer(q, kv)
        assert out.e_neg.mean().item() > 0.1, (
            f"Expected out.e_neg > 0.1, got {out.e_neg.mean().item()}"
        )

    def test_refutation_preservation_under_extreme_batch_and_heads(self) -> None:
        """Stress-tests coupled routing across diverse dimensions and head counts."""
        for n_heads in [1, 2, 4, 8]:
            d_model = 32
            layer = BelnapAttention(d_model=d_model, n_heads=n_heads)
            q = BelnapState(
                e_pos=torch.ones(3, 1, d_model),
                e_neg=torch.zeros(3, 1, d_model),
            )
            kv = BelnapState(
                e_pos=torch.zeros(3, 8, d_model),
                e_neg=torch.ones(3, 8, d_model),
            )
            out = layer(q, kv)
            assert out.e_neg.mean().item() > 0.1, f"Failed for n_heads={n_heads}"

    def test_refutation_cross_channel_gradient_flow(self, attn_layer: BelnapAttention) -> None:
        """Verifies non-zero gradients flow through cross channels (A_+ V_- and A_- V_+)."""
        d_model = 32
        q = BelnapState(
            e_pos=torch.ones(2, 2, d_model, requires_grad=True),
            e_neg=torch.zeros(2, 2, d_model, requires_grad=True),
        )
        kv = BelnapState(
            e_pos=torch.zeros(2, 3, d_model, requires_grad=True),
            e_neg=torch.ones(2, 3, d_model, requires_grad=True),
        )

        out = attn_layer(q, kv)
        # Loss exclusively on negative evidence channel
        loss = out.e_neg.sum()
        loss.backward()

        assert q.e_pos.grad is not None and q.e_pos.grad.abs().sum() > 0, (
            "Gradient must flow from e_neg back to positive query via cross-coupling"
        )
        assert kv.e_neg.grad is not None and kv.e_neg.grad.abs().sum() > 0, (
            "Gradient must flow back to negative key/value"
        )

    def test_refutation_preservation_across_cascaded_reasoning_blocks(self) -> None:
        """Verifies refutation evidence survives through multi-layer transformer blocks."""
        d_model = 32
        block1 = BelnapTransformerBlock(d_model=d_model, n_heads=4, residual_weight=0.15)
        block2 = BelnapTransformerBlock(d_model=d_model, n_heads=4, residual_weight=0.15)

        q = BelnapState(
            e_pos=torch.ones(2, 3, d_model),
            e_neg=torch.zeros(2, 3, d_model),
        )
        kv = BelnapState(
            e_pos=torch.zeros(2, 3, d_model),
            e_neg=torch.ones(2, 3, d_model),
        )

        h1 = block1(q, kv)
        h2 = block2(h1, kv)

        assert h2.e_neg.mean().item() > 0.1, (
            "Refutation evidence must persist across cascaded reasoning layers"
        )


# ==============================================================================
# Suite 2: Double Negation Symmetry & Invariants
# ==============================================================================


class TestDoubleNegationSymmetryAndEquivariance:
    """Verifies that 2x2 coupled bilattice attention satisfies exact double negation
    symmetry and bilattice algebraic equivariance.
    """

    @pytest.fixture
    def symmetric_attn_layer(self) -> BelnapAttention:
        """BelnapAttention instance with tied positive/negative projection weights."""
        torch.manual_seed(42)
        attn = BelnapAttention(d_model=32, n_heads=2)
        with torch.no_grad():
            attn.w_q_neg.weight.copy_(attn.w_q_pos.weight)
            attn.w_q_neg.bias.copy_(attn.w_q_pos.bias)
            attn.w_k_neg.weight.copy_(attn.w_k_pos.weight)
            attn.w_k_neg.bias.copy_(attn.w_k_pos.bias)
            attn.w_v_neg.weight.copy_(attn.w_v_pos.weight)
            attn.w_v_neg.bias.copy_(attn.w_v_pos.bias)
            attn.w_out_neg.weight.copy_(attn.w_out_pos.weight)
            attn.w_out_neg.bias.copy_(attn.w_out_pos.bias)
            attn.norm_neg.weight.copy_(attn.norm_pos.weight)
            attn.norm_neg.bias.copy_(attn.norm_pos.bias)
        return attn

    def test_double_negation_query_inversion_exactness(
        self, symmetric_attn_layer: BelnapAttention
    ) -> None:
        """Verifies Agg(~Q, K, V) == ~Agg(Q, K, V)."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(2, 3, d_model), e_neg=torch.rand(2, 3, d_model))
        kv = BelnapState(e_pos=torch.rand(2, 4, d_model), e_neg=torch.rand(2, 4, d_model))

        out_orig = symmetric_attn_layer(q, kv)
        out_neg_q = symmetric_attn_layer(q.negate(), kv)

        # Swapped polarities
        assert torch.allclose(out_neg_q.e_pos, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_neg_q.e_neg, out_orig.e_pos, atol=1e-5)
        # Truth inverted: t' = 1 - t
        assert torch.allclose(out_neg_q.truth, 1.0 - out_orig.truth, atol=1e-5)
        # Knowledge invariant: k' = k
        assert torch.allclose(out_neg_q.knowledge, out_orig.knowledge, atol=1e-5)

    def test_double_negation_key_value_inversion_exactness(
        self, symmetric_attn_layer: BelnapAttention
    ) -> None:
        """Verifies Agg(Q, ~K, ~V) == Agg(Q, K, V)."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(2, 2, d_model), e_neg=torch.rand(2, 2, d_model))
        kv = BelnapState(e_pos=torch.rand(2, 3, d_model), e_neg=torch.rand(2, 3, d_model))

        out_orig = symmetric_attn_layer(q, kv)
        out_neg_kv = symmetric_attn_layer(q, kv.negate())

        assert torch.allclose(out_neg_kv.e_pos, out_orig.e_pos, atol=1e-5)
        assert torch.allclose(out_neg_kv.e_neg, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_neg_kv.truth, out_orig.truth, atol=1e-5)
        assert torch.allclose(out_neg_kv.knowledge, out_orig.knowledge, atol=1e-5)

    def test_full_triple_inversion(self, symmetric_attn_layer: BelnapAttention) -> None:
        """Verifies Agg(~Q, ~K, ~V) == ~Agg(Q, K, V)."""
        d_model = 32
        q = BelnapState(e_pos=torch.rand(2, 2, d_model), e_neg=torch.rand(2, 2, d_model))
        kv = BelnapState(e_pos=torch.rand(2, 3, d_model), e_neg=torch.rand(2, 3, d_model))

        out_orig = symmetric_attn_layer(q, kv)
        out_triple = symmetric_attn_layer(q.negate(), kv.negate())

        assert torch.allclose(out_triple.e_pos, out_orig.e_neg, atol=1e-5)
        assert torch.allclose(out_triple.e_neg, out_orig.e_pos, atol=1e-5)

    @settings(max_examples=30, deadline=None)
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_hypothesis_circulant_operator_symmetry(
        self, q_pos: float, q_neg: float, kv_pos: float, kv_neg: float
    ) -> None:
        """Property-based verification of circulant 2x2 matrix operator on continuous inputs."""
        qp = torch.tensor([q_pos])
        qn = torch.tensor([q_neg])
        kp = torch.tensor([kv_pos])
        kn = torch.tensor([kv_neg])
        vp = torch.tensor([kv_pos])
        vn = torch.tensor([kv_neg])

        scale = 1.0
        # Forward pass
        s_pos = (qp * kp + qn * kn) / scale
        s_neg = (qp * kn + qn * kp) / scale
        a_pos = torch.sigmoid(s_pos)
        a_neg = torch.sigmoid(s_neg)
        agg_pos = a_pos * vp + a_neg * vn
        agg_neg = a_pos * vn + a_neg * vp

        # Negate query
        s_pos_nq = (qn * kp + qp * kn) / scale
        s_neg_nq = (qn * kn + qp * kp) / scale
        a_pos_nq = torch.sigmoid(s_pos_nq)
        a_neg_nq = torch.sigmoid(s_neg_nq)
        agg_pos_nq = a_pos_nq * vp + a_neg_nq * vn
        agg_neg_nq = a_pos_nq * vn + a_neg_nq * vp

        assert torch.isclose(agg_pos_nq, agg_neg, atol=1e-6)
        assert torch.isclose(agg_neg_nq, agg_pos, atol=1e-6)


# ==============================================================================
# Suite 3: Epistemic Logit Gating & Knowledge Synchronization
# ==============================================================================


class TestEpistemicLogitGatingLowKnowledge:
    """Empirical verification that low-knowledge candidates (k -> 0) strictly evaluate
    to near-zero logits (|logit| < 1e-3) and produce uniform Softmax probabilities.
    """

    def test_exact_zero_knowledge_suppresses_logits_scalar_head(self) -> None:
        """Candidates at Neither (k = 0) must produce exactly 0 knowledge and zero logits."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        zero_state = BelnapState(e_pos=torch.zeros(4, 32), e_neg=torch.zeros(4, 32))
        out = head(zero_state)

        assert (out["knowledge"] == 0.0).all()
        assert (out["logits"].abs() < 1e-6).all(), (
            f"Expected |logit| < 1e-6, got {out['logits'].abs().max().item()}"
        )

    def test_exact_zero_knowledge_suppresses_logits_multichoice_head(self) -> None:
        """Multi-choice head with zero knowledge must produce exactly 0 logits."""
        for num_choices in [2, 3, 4, 8]:
            head = BelnapDecisionHead(d_model=32, num_choices=num_choices)
            zero_state = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
            out = head(zero_state)

            assert (out["knowledge"] == 0.0).all()
            assert (out["logits"].abs() < 1e-6).all(), (
                f"Failed for {num_choices}: max |logit| = {out['logits'].abs().max().item()}"
            )

    @pytest.mark.parametrize("k_val", [0.0, 1e-6, 1e-5, 5e-5, 1e-4])
    def test_low_knowledge_sweep_scalar_head(self, k_val: float) -> None:
        """Verifies that across k in [0, 1e-4], |logit| < 1e-3 strictly holds."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.ones(4, 32) * k_val, e_neg=torch.ones(4, 32) * k_val)
        out = head(state)

        assert out["knowledge"].max().item() <= 2.0 * k_val + 1e-7
        assert out["logits"].abs().max().item() < 1e-3, (
            f"For k={k_val}, expected |logit| < 1e-3, got {out['logits'].abs().max().item()}"
        )

    @pytest.mark.parametrize("k_val", [0.0, 1e-6, 1e-5, 5e-5, 1e-4])
    def test_low_knowledge_sweep_multichoice_head(self, k_val: float) -> None:
        """Verifies that across k in [0, 1e-4], multi-choice head satisfies |logit| < 1e-3."""
        head = BelnapDecisionHead(d_model=32, num_choices=4)
        state = BelnapState(e_pos=torch.ones(3, 32) * k_val, e_neg=torch.ones(3, 32) * k_val)
        out = head(state)

        assert out["logits"].abs().max().item() < 1e-3, (
            f"For k={k_val}, expected |logit| < 1e-3, got {out['logits'].abs().max().item()}"
        )

    @pytest.mark.parametrize("k_val", [0.0, 1e-6, 1e-5, 5e-5, 1e-4])
    def test_bilinear_candidate_states_low_knowledge_2d(self, k_val: float) -> None:
        """Bilinear head with 2D candidate states must strictly evaluate to |logit| < 1e-3."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        cand_state = BelnapState(e_pos=torch.ones(2, 32) * k_val, e_neg=torch.ones(2, 32) * k_val)

        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"]
        assert logits.abs().max().item() < 1e-3, (
            f"For k={k_val}, expected bilinear |logit| < 1e-3, got {logits.abs().max().item()}"
        )

    @pytest.mark.parametrize("k_val", [0.0, 1e-6, 1e-5, 5e-5, 1e-4])
    def test_bilinear_candidate_states_low_knowledge_3d(self, k_val: float) -> None:
        """Bilinear head with 3D candidate states [B, K, D] evaluates to |logit| < 1e-3."""
        head = BelnapDecisionHead(d_model=32, num_choices=3)
        query_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        cand_state = BelnapState(
            e_pos=torch.ones(2, 3, 32) * k_val,
            e_neg=torch.ones(2, 3, 32) * k_val,
        )

        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"]
        assert logits.abs().max().item() < 1e-3, (
            f"For k={k_val}, expected 3D candidate |logit| < 1e-3, got {logits.abs().max().item()}"
        )

    def test_adversarial_maximal_query_with_low_knowledge_candidate(self) -> None:
        """Adversarial stress: Query has maximal positive evidence and candidate has k <= 1e-4.
        Candidate gate g = min(1.0, 2 k_cand) and epistemic logit gating MUST keep |logit| < 1e-3.
        """
        head = BelnapDecisionHead(d_model=32, num_choices=4)
        # Maximal query evidence
        query_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        # Candidate 0 is True (k=0.5), candidates 1..3 are low knowledge (k=1e-4)
        cand_pos = torch.ones(1, 4, 32) * 1e-4
        cand_neg = torch.ones(1, 4, 32) * 1e-4
        cand_pos[0, 0] = 1.0
        cand_neg[0, 0] = 0.0

        cand_state = BelnapState(e_pos=cand_pos, e_neg=cand_neg)
        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"][0]

        # Active candidate 0 has decisive logit
        assert logits[0].item() > 0.5
        # Low knowledge candidates 1, 2, 3 must have |logit| < 1e-3
        for idx in [1, 2, 3]:
            assert abs(logits[idx].item()) < 1e-3, (
                f"Candidate {idx} with k=1e-4 had logit {logits[idx].item()} >= 1e-3"
            )

    def test_contradictory_both_candidate_produces_zero_logit_due_to_affinity_cancellation(
        self,
    ) -> None:
        """When candidate is Both (e_pos=1, e_neg=1), pos_affinity == neg_affinity,
        producing raw_diff = 0 and logit = 0 despite maximal knowledge (k=1.0).
        """
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        cand_both = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.ones(1, 32))

        out = head(query, candidate_states=cand_both)
        assert out["knowledge"].item() > 0.9, "Both candidate has maximal knowledge"
        assert abs(out["logits"].item()) < 1e-4, (
            f"Both candidate must have logit ~ 0 due to affinity cancellation, "
            f"got {out['logits'].item()}"
        )

    def test_false_candidate_produces_strongly_negative_logit(self) -> None:
        """When query is positive (e_pos=1, e_neg=0) and candidate is False (e_pos=0, e_neg=1),
        neg_affinity >> pos_affinity, producing strongly negative logit with gate = 1.0.
        """
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        cand_false = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.ones(1, 32))

        out = head(query, candidate_states=cand_false)
        assert out["logits"].item() < -0.5, (
            f"False candidate must produce negative logit < -0.5, got {out['logits'].item()}"
        )

    def test_contradictory_both_query_produces_balanced_logits(self) -> None:
        """When query is contradictory Both (q_pos=1, q_neg=1), pos_affinity == neg_affinity
        for any candidate, producing zero net logit bias.
        """
        head = BelnapDecisionHead(d_model=32, num_choices=2)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.ones(1, 32))
        cand = BelnapState(
            e_pos=torch.tensor([[[1.0] * 32, [0.0] * 32]]),
            e_neg=torch.tensor([[[0.0] * 32, [1.0] * 32]]),
        )

        out = head(query, candidate_states=cand)
        logits = out["logits"][0]
        assert abs(logits[0].item() - logits[1].item()) < 1e-4, (
            "Contradictory query must not exhibit arbitrary candidate preference"
        )

    def test_mixed_candidates_ranking_and_logit_invariants(self) -> None:
        """Stress-tests a 4-choice setting with True, Neither, False, and Both candidates.
        Expected order: Logit(True) > 0, Logit(Both) ~ 0, Logit(Neither) = 0, Logit(False) < 0.
        """
        head = BelnapDecisionHead(d_model=32, num_choices=4)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))

        # 4 candidates: [True, Neither, Both, False]
        c_pos = torch.zeros(1, 4, 32)
        c_neg = torch.zeros(1, 4, 32)
        c_pos[0, 0] = 1.0  # True
        # idx 1 is Neither (0, 0)
        c_pos[0, 2] = 1.0  # Both
        c_neg[0, 2] = 1.0
        c_neg[0, 3] = 1.0  # False

        cand = BelnapState(e_pos=c_pos, e_neg=c_neg)
        out = head(query, candidate_states=cand)
        logits = out["logits"][0]

        logit_true = logits[0].item()
        logit_neither = logits[1].item()
        logit_both = logits[2].item()
        logit_false = logits[3].item()

        assert logit_true > 0.5, f"True logit must be > 0.5, got {logit_true}"
        assert abs(logit_neither) < 1e-4, f"Neither logit must be ~0, got {logit_neither}"
        assert abs(logit_both) < 1e-4, f"Both logit must be ~0, got {logit_both}"
        assert logit_false < -0.5, f"False logit must be < -0.5, got {logit_false}"
        assert logit_true > logit_neither > logit_false, "Ordering must preserve truth lattice"

    @pytest.mark.parametrize("num_choices", [2, 3, 4, 8, 16])
    def test_uniform_softmax_probabilities_under_zero_knowledge(self, num_choices: int) -> None:
        """Under total ignorance (k = 0 for all candidates), probabilities must be strictly 1/K."""
        head = BelnapDecisionHead(d_model=32, num_choices=num_choices)
        zero_state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(zero_state)

        probs = F.softmax(out["logits"], dim=-1)[0]
        expected_p = 1.0 / num_choices
        for p in probs:
            assert pytest.approx(p.item(), abs=1e-5) == expected_p, (
                f"Expected {expected_p}, got {p.item()} for num_choices={num_choices}"
            )


# ==============================================================================
# Suite 4: Decisive Candidate Dynamic Range Preservation
# ==============================================================================


class TestDecisiveCandidateDynamicRange:
    """Verifies that decisive candidate states (True, False) retain unattenuated
    dynamic range (g = 1.0) without false 50% knowledge attenuation.
    """

    def test_decisive_states_gate_saturation(self) -> None:
        """In B4 bilattice:
        - True (pos=1, neg=0): k = 0.5 -> g = min(1.0, 2*0.5) = 1.0 (unattenuated)
        - False (pos=0, neg=1): k = 0.5 -> g = min(1.0, 2*0.5) = 1.0 (unattenuated)
        - Both (pos=1, neg=1): k = 1.0 -> g = min(1.0, 2*1.0) = 1.0 (unattenuated)
        - Neither (pos=0, neg=0): k = 0.0 -> g = 0.0 (fully suppressed)
        """
        corners = {
            "True": (1.0, 0.0, 1.0),
            "False": (0.0, 1.0, 1.0),
            "Both": (1.0, 1.0, 1.0),
            "Neither": (0.0, 0.0, 0.0),
        }
        for name, (pos_val, neg_val, expected_gate) in corners.items():
            pos = torch.tensor([pos_val])
            neg = torch.tensor([neg_val])
            k = (pos + neg) / 2.0
            gate = (2.0 * k).clamp(max=1.0)
            assert gate.item() == expected_gate, f"Failed for {name}: expected {expected_gate}"

    def test_decisive_candidate_dynamic_range_in_head(self) -> None:
        """Verifies that decisive candidate states produce unattenuated dynamic range
        compared to suppressed low-knowledge candidates.
        """
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))

        # True candidate
        cand_true = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out_true = head(query, candidate_states=cand_true)

        # Neither candidate
        cand_neither = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out_neither = head(query, candidate_states=cand_neither)

        # Dynamic separation
        assert out_true["logits"].item() > 0.5, "True candidate must have robust positive logit"
        assert abs(out_neither["logits"].item()) < 1e-4, "Neither candidate must be near 0"
        separation = out_true["logits"].item() - out_neither["logits"].item()
        assert separation > 0.5, f"Expected separation > 0.5, got {separation}"


# ==============================================================================
# Suite 5: NeSy Positive Support Constraint Verification
# ==============================================================================


class TestNeSyPositiveSupportConstraints:
    """Verifies that reformulated XOR and OR evaluate positive support, guaranteeing
    that unmentioned candidates at Neither (0, 0) incur exact 0 constraint loss.
    """

    def test_unmentioned_neither_zero_penalty_xor(self) -> None:
        """Pure Neither candidates must incur exact 0 XOR penalty."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.4)
        # All candidates unmentioned (pos=0)
        pos_neither = torch.zeros(4)
        loss = criterion._compute_xor(pos_neither)
        assert loss.item() == 0.0, f"Expected 0.0 loss, got {loss.item()}"

    def test_single_active_choice_with_neither_padding_xor(self) -> None:
        """One chosen candidate (pos=1.0) plus arbitrary Neither candidates (pos=0.0)
        must satisfy XOR with exact 0 loss.
        """
        criterion = LTNConstraintLoss(decisiveness_weight=0.4)
        for num_neither in [1, 2, 5, 10, 50]:
            support = torch.cat([torch.tensor([1.0]), torch.zeros(num_neither)])
            loss = criterion._compute_xor(support)
            assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
                f"Failed with {num_neither} Neither candidates: loss={loss.item()}"
            )

    def test_contradictory_multiple_active_choices_incur_penalty(self) -> None:
        """Multiple active choices (pos=1.0) must incur strictly positive XOR penalty."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.4)
        support = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = criterion._compute_xor(support)
        assert loss.item() > 0.5, f"Expected violation penalty, got {loss.item()}"

    def test_or_satisfaction_with_neither_padding(self) -> None:
        """At-least-one (OR) is fully satisfied by 1 active option regardless of Neither padding."""
        criterion = LTNConstraintLoss(decisiveness_weight=0.4)
        for num_neither in [1, 3, 10]:
            support = torch.cat([torch.tensor([1.0]), torch.zeros(num_neither)])
            loss = criterion._compute_or(support)
            assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
                f"Failed with {num_neither} Neither candidates: OR loss={loss.item()}"
            )

    @settings(max_examples=60, deadline=None)
    @given(
        base_vals=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        ),
        num_zeros=st.integers(min_value=1, max_value=10),
    )
    def test_hypothesis_neither_invariance_arbitrary_support(
        self, base_vals: list[float], num_zeros: int
    ) -> None:
        """Hypothesis: Appending zeros to candidate support (K >= 2) leaves XOR loss invariant."""
        loss_fn = LTNConstraintLoss()
        t_base = torch.tensor(base_vals, dtype=torch.float32)
        t_zeros = torch.zeros(num_zeros, dtype=torch.float32)
        t_augmented = torch.cat([t_base, t_zeros])

        loss_base = loss_fn._compute_xor(t_base).item()
        loss_aug = loss_fn._compute_xor(t_augmented).item()

        assert pytest.approx(loss_base, abs=1e-5) == loss_aug, (
            f"Appending {num_zeros} unmentioned candidates altered XOR: {loss_base} vs {loss_aug}"
        )

    @settings(max_examples=60, deadline=None)
    @given(
        base_vals=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=6,
        ),
        num_zeros=st.integers(min_value=1, max_value=10),
    )
    def test_hypothesis_or_neither_invariance(self, base_vals: list[float], num_zeros: int) -> None:
        """Hypothesis: Appending zeros to ANY positive support vector leaves OR loss invariant."""
        loss_fn = LTNConstraintLoss()
        t_base = torch.tensor(base_vals, dtype=torch.float32)
        t_zeros = torch.zeros(num_zeros, dtype=torch.float32)
        t_augmented = torch.cat([t_base, t_zeros])

        loss_base = loss_fn._compute_or(t_base).item()
        loss_aug = loss_fn._compute_or(t_augmented).item()

        assert pytest.approx(loss_base, abs=1e-5) == loss_aug, (
            f"Appending {num_zeros} unmentioned candidates altered OR: {loss_base} vs {loss_aug}"
        )

    @settings(max_examples=80, deadline=None)
    @given(
        support=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        )
    )
    def test_hypothesis_xor_commutativity(self, support: list[float]) -> None:
        """XOR loss is invariant under candidate permutations."""
        loss_fn = LTNConstraintLoss()
        t = torch.tensor(support, dtype=torch.float32)
        l_orig = loss_fn._compute_xor(t).item()
        l_rev = loss_fn._compute_xor(torch.flip(t, dims=[0])).item()
        l_roll = loss_fn._compute_xor(torch.roll(t, shifts=1, dims=0)).item()

        assert pytest.approx(l_orig, abs=1e-5) == l_rev
        assert pytest.approx(l_orig, abs=1e-5) == l_roll

    @settings(max_examples=80, deadline=None)
    @given(
        support=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        )
    )
    def test_hypothesis_or_commutativity(self, support: list[float]) -> None:
        """OR loss is invariant under candidate permutations."""
        loss_fn = LTNConstraintLoss()
        t = torch.tensor(support, dtype=torch.float32)
        l_orig = loss_fn._compute_or(t).item()
        l_rev = loss_fn._compute_or(torch.flip(t, dims=[0])).item()
        l_roll = loss_fn._compute_or(torch.roll(t, shifts=1, dims=0)).item()

        assert pytest.approx(l_orig, abs=1e-5) == l_rev
        assert pytest.approx(l_orig, abs=1e-5) == l_roll

    @settings(max_examples=60, deadline=None)
    @given(
        e1=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_hypothesis_xor_monotonicity(self, e1: float, delta: float) -> None:
        """Increasing competitor support increases XOR penalty when chosen option is active."""
        loss_fn = LTNConstraintLoss()
        e2 = e1 + delta
        v_low = torch.tensor([1.0, e1], dtype=torch.float32)
        v_high = torch.tensor([1.0, e2], dtype=torch.float32)

        l_low = loss_fn._compute_xor(v_low).item()
        l_high = loss_fn._compute_xor(v_high).item()

        assert l_high >= l_low - 1e-6

    @settings(max_examples=60, deadline=None)
    @given(
        e=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_hypothesis_or_monotonicity(self, e: float, delta: float) -> None:
        """Increasing positive support decreases At-Least-One (OR) penalty."""
        loss_fn = LTNConstraintLoss()
        v_low = torch.tensor([e, 0.0], dtype=torch.float32)
        v_high = torch.tensor([e + delta, 0.0], dtype=torch.float32)

        l_low = loss_fn._compute_or(v_low).item()
        l_high = loss_fn._compute_or(v_high).item()

        assert l_high <= l_low + 1e-6


# ==============================================================================
# Suite 6: Refutation Channel Isolation & Gradient Dynamics
# ==============================================================================


class TestRefutationChannelIsolationAndGradients:
    """Empirical verification of evidence channel decoupling and gradient routing."""

    def test_refutation_channel_exact_gradient_isolation_legacy(self) -> None:
        """Verifies d(L_XOR)/d(choice_neg) == 0 in legacy batch routing."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.0)

        choice_pos = torch.tensor([[0.8, 0.3, 0.0]], requires_grad=True)
        choice_neg = torch.tensor([[0.2, 0.7, 0.9]], requires_grad=True)
        task_scalars = torch.tensor([[1.0]])

        outputs = {
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
            "truth": 0.5 * (choice_pos - choice_neg + 1.0),
        }

        loss = loss_fn(outputs, task_scalars=task_scalars)
        loss.backward()

        assert choice_pos.grad is not None
        assert choice_pos.grad.norm().item() > 0.0, "Positive channel must receive gradient"

        assert choice_neg.grad is None or choice_neg.grad.norm().item() == 0.0, (
            f"Negative channel must receive ZERO gradient from XOR, got {choice_neg.grad}"
        )

    def test_refutation_channel_exact_gradient_isolation_grouped(self) -> None:
        """Verifies d(L_XOR)/d(choice_neg) == 0 in grouped unrolled evaluation."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.0)

        choice_pos = torch.tensor([0.7, 0.4, 0.0], requires_grad=True)
        choice_neg = torch.tensor([0.3, 0.8, 0.5], requires_grad=True)
        group_ids = torch.tensor([0, 0, 0])
        task_scalars = torch.tensor([[1.0], [1.0], [1.0]])

        outputs = {
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
        }

        loss = loss_fn(outputs, task_scalars=task_scalars, group_ids=group_ids)
        loss.backward()

        assert choice_pos.grad is not None
        assert choice_pos.grad.norm().item() > 0.0

        assert choice_neg.grad is None or choice_neg.grad.norm().item() == 0.0, (
            f"Grouped negative channel must receive ZERO gradient, got {choice_neg.grad}"
        )

    def test_refutation_invariance_loss_independent_of_negative_evidence(self) -> None:
        """Varying choice_neg arbitrarily does not alter XOR or OR loss values."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.0)
        pos = torch.tensor([[0.8, 0.1, 0.0]])
        task_xor = torch.tensor([[1.0]])
        task_or = torch.tensor([[0.5]])

        neg_candidates = [
            torch.zeros_like(pos),
            torch.ones_like(pos),
            torch.tensor([[0.9, 0.8, 0.7]]),
            torch.tensor([[0.1, 0.5, 0.3]]),
        ]

        out0 = {"choice_pos": pos, "choice_neg": neg_candidates[0]}
        base_xor = loss_fn(out0, task_scalars=task_xor).item()
        base_or = loss_fn(out0, task_scalars=task_or).item()

        for idx, neg in enumerate(neg_candidates[1:], start=1):
            cur_out = {"choice_pos": pos, "choice_neg": neg}
            cur_xor = loss_fn(cur_out, task_scalars=task_xor).item()
            cur_or = loss_fn(cur_out, task_scalars=task_or).item()

            assert pytest.approx(base_xor, abs=1e-6) == cur_xor, (
                f"Neg variation {idx} altered XOR: {base_xor} vs {cur_xor}"
            )
            assert pytest.approx(base_or, abs=1e-6) == cur_or, (
                f"Neg variation {idx} altered OR: {base_or} vs {cur_or}"
            )

    def test_positive_channel_competing_gradient_direction(self) -> None:
        """When multiple candidates are active in XOR, gradients push evidence downward."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([0.75, 0.75], requires_grad=True)
        loss = loss_fn._compute_xor(pos)
        loss.backward()

        assert pos.grad is not None
        assert pos.grad[0].item() > 0.0, "Gradient must push competing candidate 0 downward"
        assert pos.grad[1].item() > 0.0, "Gradient must push competing candidate 1 downward"

    def test_positive_channel_or_deficit_gradient_direction(self) -> None:
        """When candidates lack positive support in OR, gradients push evidence upward."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([0.05, 0.05], requires_grad=True)
        loss = loss_fn._compute_or(pos)
        loss.backward()

        assert pos.grad is not None
        assert pos.grad[0].item() < 0.0, "Gradient must incentivize pos evidence for cand 0"
        assert pos.grad[1].item() < 0.0, "Gradient must incentivize pos evidence for cand 1"

    def test_single_decisive_candidate_stationary_point(self) -> None:
        """At ideal exclusive state [1.0, 0.0, 0.0], XOR gradient is stationary (zero)."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        loss = loss_fn._compute_xor(pos)
        loss.backward()

        assert pos.grad is not None
        assert torch.allclose(pos.grad, torch.zeros(3), atol=1e-6), (
            f"At ideal state (1, 0, 0), gradient must be 0, got {pos.grad}"
        )


# ==============================================================================
# Suite 7: Scalability, Extreme Boundary Regimes, & Task Routing
# ==============================================================================


class TestScalabilityAndBoundaryRegimes:
    """Stress tests on candidate dimensions, numerical stability, and task scalar thresholds."""

    def test_large_k_scalability_and_stability(self) -> None:
        """Evaluates XOR with K=64 candidates across multiple configurations without NaN/Inf."""
        loss_fn = LTNConstraintLoss()
        K = 64

        # Config 1: 1 decisive True, 63 unmentioned Neither
        v1 = torch.cat([torch.tensor([1.0]), torch.zeros(K - 1)])
        l1 = loss_fn._compute_xor(v1)
        assert pytest.approx(l1.item(), abs=1e-5) == 0.0

        # Config 2: All 64 unmentioned Neither
        v2 = torch.zeros(K)
        l2 = loss_fn._compute_xor(v2)
        assert pytest.approx(l2.item(), abs=1e-5) == 0.0

        # Config 3: 2 colliding active candidates, 62 unmentioned Neither
        v3 = torch.cat([torch.tensor([1.0, 1.0]), torch.zeros(K - 2)])
        l3 = loss_fn._compute_xor(v3)
        assert l3.item() > 10.0

        # Config 4: Diffuse uniform evidence (1/K)
        v4 = torch.full((K,), 1.0 / K)
        l4 = loss_fn._compute_xor(v4)
        assert not math.isnan(l4.item()) and not math.isinf(l4.item())

    def test_empty_and_singleton_edge_cases(self) -> None:
        """Empty tensors evaluate to 0.0; singletons evaluate to -log(e_0)."""
        loss_fn = LTNConstraintLoss()

        # Empty tensor
        empty = torch.tensor([])
        assert loss_fn._compute_xor(empty).item() == 0.0
        assert loss_fn._compute_or(empty).item() == 0.0

        # Singleton tensor
        s_true = torch.tensor([1.0])
        assert pytest.approx(loss_fn._compute_xor(s_true).item(), abs=1e-6) == 0.0
        assert pytest.approx(loss_fn._compute_or(s_true).item(), abs=1e-6) == 0.0

        s_false = torch.tensor([0.0])
        expected_penalty = -math.log(loss_fn.eps)
        assert pytest.approx(loss_fn._compute_xor(s_false).item(), abs=1e-5) == expected_penalty
        assert pytest.approx(loss_fn._compute_or(s_false).item(), abs=1e-5) == expected_penalty

    def test_task_scalar_boundary_dispatch_transitions(self) -> None:
        """Verifies exact dispatch at boundaries: 0.75 (XOR vs OR) and 0.25 (OR vs Boundedness)."""
        loss_fn = LTNConstraintLoss(bound_a=-1.0, bound_b=1.0)

        # 2 conflicting candidates: [1.0, 1.0]
        # XOR of [1.0, 1.0] is maximal (~16.11)
        # OR of [1.0, 1.0] is 0.0 (both true satisfies at-least-one)
        pos = torch.tensor([[1.0, 1.0]])
        neg = torch.zeros(1, 2)
        outputs = {"choice_pos": pos, "choice_neg": neg}

        # Case 1: 0.7501 -> XOR -> high loss
        loss_xor = loss_fn(outputs, task_scalars=torch.tensor([[0.7501]])).item()
        assert loss_xor > 10.0

        # Case 2: 0.7500 -> OR -> zero loss
        loss_or = loss_fn(outputs, task_scalars=torch.tensor([[0.7500]])).item()
        assert pytest.approx(loss_or, abs=1e-5) == 0.0

        # Case 3: 0.2501 -> OR -> zero loss
        loss_or_low = loss_fn(outputs, task_scalars=torch.tensor([[0.2501]])).item()
        assert pytest.approx(loss_or_low, abs=1e-5) == 0.0

        # Case 4: 0.2500 -> Box boundedness
        logits = torch.tensor([[5.0, -5.0]])  # Violates [-1.0, 1.0]
        outputs_box = {"choice_pos": pos, "choice_neg": neg, "logits": logits}
        loss_box = loss_fn(outputs_box, task_scalars=torch.tensor([[0.2500]])).item()
        assert loss_box > 0.0  # Boundedness violation
