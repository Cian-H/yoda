"""Comprehensive test suite for R3: Positive Support LTN XOR/OR Reformulation.

Verifies:
1. Four-Valued Belnap corners (True, False, Neither, Both) under positive support XOR/OR.
2. Elimination of false 0.693 penalty per unmentioned Neither candidate compared to truth.
3. Negative evidence gradient isolation (dL/de_neg == 0 for XOR/OR).
4. Positive evidence gradient dynamics (competing options penalized, unmentioned options unpenalized).
5. Grouped multi-choice candidate routing and task scalar dispatch.
6. Mathematical invariants and property-based boundary testing via Hypothesis.
"""

import math
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from yoda.nesy import LTNConstraintLoss


class TestBelnapCornersPositiveSupport:
    """Verifies XOR and OR behavior across all four Belnap-Dunn valuation corners."""

    def test_xor_single_true_and_multiple_neither(self) -> None:
        """Candidate 0 is True (e+=1, e-=0); candidates 1, 2, 3 are Neither (e+=0, e-=0)."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss = loss_fn._compute_xor(pos)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
            "Single True candidate with unmentioned Neither options must incur 0 XOR loss"
        )

    def test_xor_single_true_and_multiple_refuted_false(self) -> None:
        """Candidate 0 is True (e+=1, e-=0); candidates 1, 2 are refuted False (e+=0, e-=1)."""
        loss_fn = LTNConstraintLoss()
        # Both False and Neither have e+=0
        pos = torch.tensor([1.0, 0.0, 0.0])
        loss = loss_fn._compute_xor(pos)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
            "Single True candidate with refuted False options must incur 0 XOR loss"
        )

    def test_xor_mutual_exclusivity_violation(self) -> None:
        """Two candidates are both True (e_0+=1.0, e_1+=1.0) -> severe exclusivity violation."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([1.0, 1.0, 0.0])
        loss = loss_fn._compute_xor(pos)
        # Violating exclusivity drives p_xor to 0, incurring maximal loss (-log(eps))
        assert loss.item() > 10.0, (
            "Violating mutual exclusivity (two True options) must incur maximal loss"
        )

    def test_xor_unmentioned_neither_slice_zero_penalty(self) -> None:
        """Evaluating an unmentioned Neither slice (e+=0, e-=0) incurs 0 penalty."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([0.0, 0.0])
        loss = loss_fn._compute_xor(pos)
        assert loss.item() < 0.05, (
            "Unmentioned Neither slice must not incur false penalty under positive support reformulation"
        )

    def test_or_single_true_and_multiple_neither(self) -> None:
        """Candidate 0 is True (e+=1); candidates 1, 2 are Neither (e+=0)."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([1.0, 0.0, 0.0])
        loss = loss_fn._compute_or(pos)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0, (
            "Active True candidate satisfies OR regardless of unmentioned Neither candidates"
        )

    def test_or_all_neither_penalized(self) -> None:
        """All candidates are Neither (e+=0) -> violates At-Least-One (OR)."""
        loss_fn = LTNConstraintLoss()
        pos = torch.tensor([0.0, 0.0, 0.0])
        loss = loss_fn._compute_or(pos)
        assert loss.item() > 10.0, (
            "Zero evidence across all options must be penalized for violating OR"
        )


class TestTruthVsPositiveSupportComparison:
    """Quantitatively compares truth coordinate vs positive support formulations."""

    def test_xor_elimination_of_half_truth_penalty(self) -> None:
        """Proves exact elimination of -log(1 - 0.5) = 0.69315 per unmentioned option."""
        # For M unmentioned options:
        # Old truth formulation: t_0 = 1.0, t_1 = ... = t_M = 0.5
        # P(XOR_old) = 1.0 * (1 - 0.5)^M + M * 0.5 * (1 - 1.0) * (1 - 0.5)^(M-1)
        #             = 0.5^M
        # Loss_old = -log(0.5^M) = M * log(2) = M * 0.693147...
        loss_fn = LTNConstraintLoss()

        for M in [1, 2, 3, 4]:
            # Old truth simulation:
            old_truth = torch.tensor([1.0] + [0.5] * M)
            # Old XOR manually computed:
            p_not_old = 1.0 - old_truth
            p_xor_old = old_truth[0] * torch.prod(p_not_old[1:])
            old_loss = -torch.log(p_xor_old).item()
            expected_old_loss = M * math.log(2.0)
            assert pytest.approx(old_loss, abs=1e-5) == expected_old_loss

            # New positive support:
            new_pos = torch.tensor([1.0] + [0.0] * M)
            new_loss = loss_fn._compute_xor(new_pos).item()

            assert pytest.approx(new_loss, abs=1e-5) == 0.0
            assert pytest.approx(old_loss - new_loss, abs=1e-5) == expected_old_loss

    def test_or_hallucinated_satisfaction_elimination(self) -> None:
        """Old truth OR hallucinated satisfaction 1 - 0.5^K under zero evidence; new OR requires e+."""
        # Old truth OR on all Neither: P(OR) = 1 - 0.5^K.
        # For K=4, P(OR) = 1 - 0.0625 = 0.9375 (loss = 0.0645) -> model thought OR was satisfied!
        # New positive support OR: P(OR) = 1 - 1^K = 0.0 (loss = 16.11) -> properly penalized!
        loss_fn = LTNConstraintLoss()
        new_pos = torch.tensor([0.0, 0.0, 0.0, 0.0])
        new_loss = loss_fn._compute_or(new_pos).item()
        assert new_loss > 10.0


class TestGradientDynamicsAndIsolation:
    """Verifies gradient highways and evidence channel isolation."""

    def test_negative_evidence_isolated_from_xor(self) -> None:
        """Refutation channel e- receives exactly zero gradient from XOR loss."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.0)  # isolate XOR from contradiction

        pos = torch.tensor([0.9, 0.1, 0.0], requires_grad=True)
        neg = torch.tensor([0.1, 0.0, 0.0], requires_grad=True)
        truth = 0.5 * (pos - neg + 1.0)

        outputs = {
            "choice_pos": pos,
            "choice_neg": neg,
            "truth": truth,
        }
        task_scalars = torch.tensor([[1.0]])

        loss = loss_fn(outputs, task_scalars)
        loss.backward()

        # Positive channel receives gradient pushing competing option (idx 1) down
        assert pos.grad is not None
        assert pos.grad[1] > 0.0  # pushes down competing positive evidence

        # Negative channel must receive ZERO gradient (not forced to fabricate refutation)
        assert neg.grad is None or torch.all(neg.grad == 0.0)

    def test_neither_candidate_receives_zero_xor_gradient_when_single_active(self) -> None:
        """Unmentioned option (pos=0) receives zero gradient when primary option is decisive."""
        loss_fn = LTNConstraintLoss(decisiveness_weight=0.0)

        pos = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        outputs = {
            "choice_pos": pos,
            "choice_neg": torch.zeros(3),
            "truth": torch.tensor([1.0, 0.5, 0.5]),
        }
        task_scalars = torch.tensor([[1.0]])

        loss = loss_fn(outputs, task_scalars)
        assert pytest.approx(loss.item(), abs=1e-5) == 0.0
        loss.backward()

        assert pos.grad is not None
        # All gradients should be zero at the optimal point (1, 0, 0)
        assert torch.allclose(pos.grad, torch.zeros(3), atol=1e-5)


class TestGroupedMultiChoiceRouting:
    """Verifies candidate group dispatch and unrolled evaluation."""

    def test_heterogeneous_groups_routing(self) -> None:
        """Routes Group 0 (Choice, XOR), Group 1 (Multi-Choice, OR), Group 2 (Score, Box)."""
        loss_fn = LTNConstraintLoss(bound_a=-2.0, bound_b=2.0)

        # 3 groups of 2 candidates each
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        task_scalars = torch.tensor([[1.0], [1.0], [0.5], [0.5], [0.0], [0.0]])

        # Group 0: Choice (1.0, 0.0) -> XOR satisfied (loss ~ 0)
        # Group 1: Multi-choice (1.0, 0.0) -> OR satisfied (loss ~ 0)
        # Group 2: Score logits (5.0, 0.0) -> 5.0 violates upper bound 2.0 (loss > 0)
        choice_pos = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5], requires_grad=True)
        choice_neg = torch.zeros(6, requires_grad=True)
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 0.0], requires_grad=True)

        outputs = {
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
            "truth": torch.tensor([1.0, 0.5, 1.0, 0.5, 0.5, 0.5]),
            "logits": logits,
        }

        loss = loss_fn(outputs, task_scalars=task_scalars, group_ids=group_ids)
        assert loss > 0.0
        loss.backward()

        # Group 2 logits must receive gradient from boundedness
        assert logits.grad is not None
        assert logits.grad[4] > 0.0  # penalize 5.0 down toward 2.0


class TestPropertyBasedInvariants:
    """Property-based testing of LTN relaxations via Hypothesis."""

    @settings(max_examples=50)
    @given(
        e0=st.floats(min_value=0.0, max_value=1.0),
        e1=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_xor_commutativity(self, e0: float, e1: float) -> None:
        """XOR loss is symmetric under candidate permutation."""
        loss_fn = LTNConstraintLoss()
        v1 = torch.tensor([e0, e1])
        v2 = torch.tensor([e1, e0])
        l1 = loss_fn._compute_xor(v1)
        l2 = loss_fn._compute_xor(v2)
        assert pytest.approx(l1.item(), abs=1e-5) == l2.item()

    @settings(max_examples=50)
    @given(
        e0=st.floats(min_value=0.0, max_value=1.0),
        e1=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_or_commutativity(self, e0: float, e1: float) -> None:
        """OR loss is symmetric under candidate permutation."""
        loss_fn = LTNConstraintLoss()
        v1 = torch.tensor([e0, e1])
        v2 = torch.tensor([e1, e0])
        l1 = loss_fn._compute_or(v1)
        l2 = loss_fn._compute_or(v2)
        assert pytest.approx(l1.item(), abs=1e-5) == l2.item()

    @settings(max_examples=50)
    @given(
        e_cand=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_xor_monotonicity_in_competing_evidence(self, e_cand: float) -> None:
        """Given a chosen candidate e_0=1.0, increasing competing evidence e_cand increases XOR loss."""
        loss_fn = LTNConstraintLoss()
        v_clean = torch.tensor([1.0, 0.0])
        v_noisy = torch.tensor([1.0, e_cand])
        l_clean = loss_fn._compute_xor(v_clean)
        l_noisy = loss_fn._compute_xor(v_noisy)
        assert l_noisy.item() >= l_clean.item() - 1e-6
