"""Tier 4 Acceptance Gate: Paraconsistent Calibration Gate.

Asserts:
1. Inputs containing unmentioned candidates at Neither retain k < 0.20
   without incurring constraint loss penalties.
2. Low-knowledge states (k -> 0) suppress classification logits,
   preventing high-confidence hallucinations under epistemic ignorance.
"""

import torch

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine
from yoda.nesy import LTNConstraintLoss


def test_neither_candidates_retain_low_knowledge_without_penalty() -> None:
    """Acceptance Gate: Unmentioned candidates retain k < 0.20 without constraint penalty."""
    engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_choices=1,
        n_heads=2,
    )
    loss_fn = LTNConstraintLoss(decisiveness_weight=0.1)

    # Question with 1 mentioned candidate and 2 completely unmentioned candidate options
    query = "Select active navigation route."
    state = {"route_a": "clear"}
    # Candidate A is mentioned; Candidate X and Candidate Y are completely unmentioned
    candidates = ["Route A: active", "Route X: unmentioned", "Route Y: unmentioned"]

    out = engine(queries=[query] * 3, states=[state] * 3, candidates=candidates)

    knowledge = out["knowledge"]
    assert knowledge.shape == (3,)

    # Unmentioned candidates (index 1 and 2) must retain low knowledge k < 0.20 (Neither)
    unmentioned_k_1 = knowledge[1].item()
    unmentioned_k_2 = knowledge[2].item()
    assert unmentioned_k_1 < 0.20, (
        f"Unmentioned candidate 1 knowledge must be < 0.20, got {unmentioned_k_1:.4f}"
    )
    assert unmentioned_k_2 < 0.20, (
        f"Unmentioned candidate 2 knowledge must be < 0.20, got {unmentioned_k_2:.4f}"
    )

    # Evaluate constraint loss on the unmentioned candidates
    # Positive-support XOR must not penalize Neither with -log(1 - 0.5) = 0.693
    unmentioned_pos = out["choice_pos"][1:]
    xor_loss = loss_fn._compute_xor(unmentioned_pos)
    assert xor_loss.item() < 0.05, (
        f"Unmentioned Neither candidates incurred false constraint loss of {xor_loss.item():.4f} "
        "(must be near 0 under positive-support reformulation)."
    )


def test_low_knowledge_states_suppress_logits() -> None:
    """Acceptance Gate: Low-knowledge states (k -> 0) must not produce high-confidence logits."""
    head = BelnapDecisionHead(d_model=32, num_choices=1)

    # State with near-zero evidence: raw_pos = -8.0, raw_neg = -12.0
    # In ungated architecture: logit = raw_pos - raw_neg = +4.0 (produces >98% softmax probability!)
    # In knowledge-synchronized architecture: logit = (+4.0) * k ~ 0.0006
    low_k_state = BelnapState(
        e_pos=torch.ones(1, 32) * 1e-4,
        e_neg=torch.ones(1, 32) * 1e-4,
    )

    out = head(low_k_state)
    logit = out["logits"].item()
    knowledge = out["knowledge"].item()

    assert knowledge < 0.01, f"Knowledge must be near 0, got {knowledge:.6f}"
    # Acceptance criterion: Low knowledge must suppress logits (must not produce high confidence)
    assert abs(logit) < 0.25, (
        f"Low-knowledge state (k={knowledge:.4f}) produced high logit {logit:.4f}. "
        "Logits must be gated by knowledge coordinate k."
    )
