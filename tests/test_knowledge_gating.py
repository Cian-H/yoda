"""Unit and property tests for Epistemic Logit Synchronization and Knowledge Gating."""

import pytest
import torch
import torch.nn.functional as F

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import BelnapDecisionHead


class TestEpistemicLogitGating:
    """Verifies that decision head logits are gated by epistemic knowledge coordinates."""

    def test_zero_knowledge_suppression_single_choice(self) -> None:
        """Zero evidence (e_pos=0, e_neg=0) must produce exactly 0 knowledge and zero logits."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        assert out["knowledge"].item() == 0.0
        assert abs(out["logits"].item()) < 1e-4
        assert pytest.approx(out["truth"].item(), abs=1e-4) == 0.5  # Neither truth value is 0.5

    def test_near_zero_knowledge_suppression_low_evidence(self) -> None:
        """Near-zero evidence (k ~ 1e-4) must keep knowledge < 0.01 and logit < 0.25."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.ones(1, 32) * 1e-4, e_neg=torch.ones(1, 32) * 1e-4)
        out = head(state)
        assert out["knowledge"].item() < 0.01
        assert abs(out["logits"].item()) < 0.25

    def test_maximal_knowledge_unsuppressed_dynamic_range(self) -> None:
        """Decisive state (k=0.5, truth=1.0) must transmit unattenuated dynamic range."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state)
        assert out["knowledge"].item() > 0.4
        assert out["logits"].item() > 0.0

    def test_bilinear_candidate_knowledge_gating_scalar(self) -> None:
        """Bilinear candidate head must gate logits by candidate knowledge."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        # Candidate 0 has full evidence, Candidate 1 has zero evidence
        cand_state = BelnapState(
            e_pos=torch.tensor([[1.0] * 32, [0.0] * 32]),
            e_neg=torch.tensor([[0.0] * 32, [0.0] * 32]),
        )
        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"]
        assert abs(logits[1].item()) < 1e-4, "Zero knowledge candidate must be suppressed to 0"
        assert logits[0].item() > logits[1].item()

    def test_bilinear_candidate_knowledge_gating_multi_choice(self) -> None:
        """Multi-choice candidate alignment must suppress zero-knowledge candidate positions."""
        head = BelnapDecisionHead(d_model=32, num_choices=3)
        query_state = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        # Candidate 0 has evidence; Candidate 1 and 2 are pure Neither
        cand_pos = torch.zeros(1, 3, 32)
        cand_pos[0, 0] = 1.0
        cand_neg = torch.zeros(1, 3, 32)
        cand_state = BelnapState(e_pos=cand_pos, e_neg=cand_neg)

        out = head(query_state, candidate_states=cand_state)
        logits = out["logits"][0]
        assert abs(logits[1].item()) < 1e-4
        assert abs(logits[2].item()) < 1e-4
        assert logits[0].item() > logits[1].item()

    def test_uniform_softmax_probabilities_under_zero_knowledge(self) -> None:
        """Zero-knowledge candidates must yield uniform Softmax distributions (1/K)."""
        head = BelnapDecisionHead(d_model=32, num_choices=4)
        zero_state = BelnapState(e_pos=torch.zeros(1, 32), e_neg=torch.zeros(1, 32))
        out = head(zero_state)
        probs = F.softmax(out["logits"], dim=-1)[0]
        for p in probs:
            assert pytest.approx(p.item(), abs=1e-3) == 0.25

    def test_gradient_flow_through_knowledge_gated_head(self) -> None:
        """Backward propagation through knowledge-gated logits must produce valid gradients."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state = BelnapState(
            e_pos=torch.rand(2, 32, requires_grad=True),
            e_neg=torch.rand(2, 32, requires_grad=True),
        )
        out = head(state)
        loss = out["logits"].sum()
        loss.backward()

        assert head.w_pos.weight.grad is not None
        assert not torch.isnan(head.w_pos.weight.grad).any()
        assert head.w_neg.weight.grad is not None
        assert not torch.isnan(head.w_neg.weight.grad).any()
