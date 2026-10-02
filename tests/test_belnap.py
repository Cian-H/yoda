"""Tests for Belnap bilattice continuous logic and uncertainty decomposition."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from yoda.probabilistic.belnap import (
    BelnapBilattice,
    BelnapEvidence,
    FuzzyBelnapLoss,
)


def test_canonical_states_truth_tables() -> None:
    """Verifies that classical Belnap 4-valued truth tables hold on canonical vertices."""
    t_true = BelnapEvidence(torch.tensor(1.0), torch.tensor(0.0))
    t_false = BelnapEvidence(torch.tensor(0.0), torch.tensor(1.0))
    t_both = BelnapEvidence(torch.tensor(1.0), torch.tensor(1.0))

    # Negation
    neg_true = BelnapBilattice.negate(t_true)
    assert (neg_true.t == 0.0) and (neg_true.f == 1.0)

    neg_both = BelnapBilattice.negate(t_both)
    assert (neg_both.t == 1.0) and (neg_both.f == 1.0)

    # Truth conjunction (meet_t): True ^ False == False
    conj = BelnapBilattice.meet_t(t_true, t_false)
    assert (conj.t == 0.0) and (conj.f == 1.0)

    # Truth disjunction (join_t): True v False == True
    disj = BelnapBilattice.join_t(t_true, t_false)
    assert (disj.t == 1.0) and (disj.f == 0.0)

    # Knowledge meet (consensus): True ^_k False == Neither (no shared info)
    consensus = BelnapBilattice.meet_k(t_true, t_false)
    assert (consensus.t == 0.0) and (consensus.f == 0.0)

    # Knowledge join (gullibility): True v_k False == Both (combined contradictory info)
    gullible = BelnapBilattice.join_k(t_true, t_false)
    assert (gullible.t == 1.0) and (gullible.f == 1.0)


def test_uncertainty_decomposition() -> None:
    """Verifies orthogonal decomposition into conflict, ignorance, and information mass."""
    t_both = BelnapEvidence(torch.tensor(1.0), torch.tensor(1.0))
    t_neither = BelnapEvidence(torch.tensor(0.0), torch.tensor(0.0))

    decomp_both = BelnapBilattice.decompose_uncertainty(t_both)
    assert decomp_both["conflict"] == 1.0
    assert decomp_both["ignorance"] == 0.0
    assert decomp_both["information"] == 1.0
    assert decomp_both["polarity"] == 0.0

    decomp_neither = BelnapBilattice.decompose_uncertainty(t_neither)
    assert decomp_neither["conflict"] == 0.0
    assert decomp_neither["ignorance"] == 1.0
    assert decomp_neither["information"] == 0.0
    assert decomp_neither["polarity"] == 0.0


def test_fuzzy_belnap_loss_execution() -> None:
    """Verifies that FuzzyBelnapLoss computes correct scalar loss and gradients."""
    pred = BelnapEvidence(
        t=torch.tensor([0.8, 0.2], requires_grad=True),
        f=torch.tensor([0.1, 0.9], requires_grad=True),
    )
    target = BelnapEvidence(
        t=torch.tensor([1.0, 0.0]),
        f=torch.tensor([0.0, 1.0]),
    )

    loss_fn = FuzzyBelnapLoss(reduction="mean")
    loss = loss_fn(pred, target)

    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 1.0

    loss.backward()
    assert pred.t.grad is not None
    assert pred.f.grad is not None


@settings(max_examples=50, deadline=None)
@given(
    t1=st.floats(min_value=0.0, max_value=1.0),
    f1=st.floats(min_value=0.0, max_value=1.0),
    t2=st.floats(min_value=0.0, max_value=1.0),
    f2=st.floats(min_value=0.0, max_value=1.0),
)
def test_lukasiewicz_equivalence_properties(t1: float, f1: float, t2: float, f2: float) -> None:
    """Property test verifying reflexivity, symmetry, and bounds of continuous satisfaction."""
    a = BelnapEvidence(torch.tensor(t1), torch.tensor(f1))
    b = BelnapEvidence(torch.tensor(t2), torch.tensor(f2))

    # Reflexivity: sat(a, a) == 1.0
    sat_self = BelnapBilattice.lukasiewicz_equivalence(a, a)
    assert torch.isclose(sat_self, torch.tensor(1.0), atol=1e-5)

    # Symmetry: sat(a, b) == sat(b, a)
    sat_ab = BelnapBilattice.lukasiewicz_equivalence(a, b)
    sat_ba = BelnapBilattice.lukasiewicz_equivalence(b, a)
    assert torch.isclose(sat_ab, sat_ba, atol=1e-5)

    # Range bound in [0, 1]
    assert 0.0 <= sat_ab.item() <= 1.00001
