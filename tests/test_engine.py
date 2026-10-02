"""Tests for top-level YodaDecisionEngine and BelnapDecisionHead."""

from typing import Any

import pytest
import torch

from yoda.architecture.belnap_transformer import BelnapState
from yoda.architecture.engine import BelnapDecisionHead, YodaDecisionEngine


class TestBelnapDecisionHead:
    """Verifies behavior of BelnapDecisionHead projection module."""

    def test_forward_shape_and_bounds(self) -> None:
        """Verifies shape of outputs and coordinate bounds in [0, 1]."""
        batch_size = 3
        num_probes = 4
        d_model = 32
        num_choices = 5

        head = BelnapDecisionHead(d_model=d_model, num_choices=num_choices)

        state = BelnapState(
            e_pos=torch.rand(batch_size, num_probes, d_model),
            e_neg=torch.rand(batch_size, num_probes, d_model),
        )

        out = head(state)

        assert "logits" in out
        assert "truth" in out
        assert "knowledge" in out
        assert "choice" in out

        assert out["logits"].shape == (batch_size, num_choices)
        assert out["truth"].shape == (batch_size, num_choices)
        assert out["knowledge"].shape == (batch_size, num_choices)
        assert out["choice"].shape == (batch_size,)

        assert torch.all(out["truth"] >= 0.0) and torch.all(out["truth"] <= 1.0)
        assert torch.all(out["knowledge"] >= 0.0) and torch.all(out["knowledge"] <= 1.0)
        assert torch.all(out["choice"] >= 0) and torch.all(out["choice"] < num_choices)

    def test_2d_input_support(self) -> None:
        """Verifies support when input state is already 2D (batch_size, d_model)."""
        batch_size = 2
        d_model = 16
        num_choices = 3

        head = BelnapDecisionHead(d_model=d_model, num_choices=num_choices)

        state = BelnapState(
            e_pos=torch.rand(batch_size, d_model),
            e_neg=torch.rand(batch_size, d_model),
        )

        out = head(state)
        assert out["logits"].shape == (batch_size, num_choices)
        assert out["choice"].shape == (batch_size,)


class TestYodaDecisionEngine:
    """Verifies end-to-end integration across Encoders, Belnap MPA, and Reasoning Core."""

    @pytest.fixture
    def dummy_engine(self) -> YodaDecisionEngine:
        torch.manual_seed(42)
        return YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=64,
            num_q_probes=4,
            num_c_probes=8,
            num_k_probes=4,
            num_choices=5,
            n_heads=4,
        )

    def test_default_initialization(self) -> None:
        """Verifies default parameters and module construction."""
        engine = YodaDecisionEngine()
        assert engine.embed_dim == 256
        assert engine.num_choices == 5
        assert isinstance(engine.decision_head, BelnapDecisionHead)

    def test_forward_pass_shapes(self, dummy_engine: YodaDecisionEngine) -> None:
        """Verifies forward pass with single and multi-item batches produces expected shapes."""
        queries: list[str] = ["optimize navigation", "verify constraint boundaries"]
        states: list[dict[str, Any]] = [
            {"speed": 100, "status": "active"},
            {"pressure": 1.05, "valve": "open"},
        ]
        constraints: list[list[str]] = [
            ["speed <= 120", "status != error"],
            ["pressure < 2.0"],
        ]

        out = dummy_engine(queries=queries, states=states, constraints=constraints)

        assert "logits" in out
        assert "truth" in out
        assert "knowledge" in out
        assert "choice" in out

        assert out["logits"].shape == (2, 5)
        assert out["truth"].shape == (2, 5)
        assert out["knowledge"].shape == (2, 5)
        assert out["choice"].shape == (2,)

        assert torch.all(out["truth"] >= 0.0) and torch.all(out["truth"] <= 1.0)
        assert torch.all(out["knowledge"] >= 0.0) and torch.all(out["knowledge"] <= 1.0)
        assert torch.all(out["choice"] >= 0) and torch.all(out["choice"] < 5)

    def test_forward_pass_single_item(self, dummy_engine: YodaDecisionEngine) -> None:
        """Verifies single-item batch inference."""
        queries = ["single query"]
        states = [{"key": "val"}]
        constraints = [["rule1"]]

        out = dummy_engine(queries=queries, states=states, constraints=constraints)
        assert out["logits"].shape == (1, 5)
        assert out["choice"].shape == (1,)

    def test_backward_gradient_flow(self, dummy_engine: YodaDecisionEngine) -> None:
        """Verifies full end-to-end gradient backpropagation through all architectural phases."""
        queries = ["evaluate path", "check logic"]
        states = [{"sensor": 10}, {"sensor": 20}]
        constraints = [["sensor < 50"], ["sensor > 0"]]

        out = dummy_engine(queries=queries, states=states, constraints=constraints)
        logits = out["logits"]
        loss = logits.sum()

        loss.backward()

        # Check Phase 4 Decision Head gradients
        assert dummy_engine.decision_head.w_pos.weight.grad is not None
        assert not torch.isnan(dummy_engine.decision_head.w_pos.weight.grad).any()
        assert dummy_engine.decision_head.w_neg.weight.grad is not None
        assert not torch.isnan(dummy_engine.decision_head.w_neg.weight.grad).any()

        # Check Phase 3 Reasoning Core gradients
        assert dummy_engine.constraint_reasoning.cross_attn.w_q_pos.weight.grad is not None
        assert dummy_engine.context_reasoning.cross_attn.w_q_pos.weight.grad is not None
        assert dummy_engine.context_reasoning.self_attn.w_q_pos.weight.grad is not None

        # Check Phase 2 Belnap MPA pooling probe gradients
        assert dummy_engine.q_mpa.raw_q_pos.grad is not None
        assert not torch.isnan(dummy_engine.q_mpa.raw_q_pos.grad).any()
        assert dummy_engine.c_mpa.raw_q_pos.grad is not None
        assert dummy_engine.k_mpa.raw_q_pos.grad is not None

        # Check Phase 1 Encoders gradients
        assert dummy_engine.text_encoder.dummy_embed.weight.grad is not None
        assert not torch.isnan(dummy_engine.text_encoder.dummy_embed.weight.grad).any()

    def test_engine_diagnostics_trajectory(self, dummy_engine: YodaDecisionEngine) -> None:
        """Verifies intermediate diagnostic probes and attribution trajectories."""
        queries = ["evaluate path", "check logic"]
        states = [{"sensor": 10}, {"sensor": 20}]
        constraints = [["sensor < 50"], ["sensor > 0"]]

        # Default forward should not return diagnostics
        default_out = dummy_engine(queries=queries, states=states, constraints=constraints)
        assert "diagnostics" not in default_out

        # Calling with return_diagnostics=True
        out = dummy_engine(
            queries=queries,
            states=states,
            constraints=constraints,
            return_diagnostics=True,
        )

        assert "diagnostics" in out
        diagnostics = out["diagnostics"]

        assert diagnostics["stage_names"] == ["post_pooling", "post_context", "post_constraint"]

        stage_logits = diagnostics["stage_logits"]
        stage_knowledge = diagnostics["stage_knowledge"]
        stage_truth = diagnostics["stage_truth"]
        attributions = diagnostics["attributions"]

        batch_size = len(queries)
        num_choices = dummy_engine.num_choices
        expected_shape = (3, batch_size, num_choices)

        assert stage_logits.shape == expected_shape
        assert stage_knowledge.shape == expected_shape
        assert stage_truth.shape == expected_shape
        assert attributions.shape == expected_shape

        # Verify final stage matches output logits, truth, knowledge
        assert torch.allclose(stage_logits[2], out["logits"])
        assert torch.allclose(stage_knowledge[2], out["knowledge"])
        assert torch.allclose(stage_truth[2], out["truth"])

        # Verify attribution deltas
        assert torch.allclose(attributions[0], stage_logits[0])
        assert torch.allclose(attributions[1], stage_logits[1] - stage_logits[0])
        assert torch.allclose(attributions[2], stage_logits[2] - stage_logits[1])
        assert torch.allclose(attributions.sum(dim=0), out["logits"])

        # Also test with candidate states active (num_k_probes == num_choices)
        aligned_engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=64,
            num_q_probes=4,
            num_c_probes=4,
            num_k_probes=5,
            num_choices=5,
            n_heads=4,
        )
        aligned_out = aligned_engine(
            queries=queries,
            states=states,
            constraints=constraints,
            return_diagnostics=True,
        )
        assert "diagnostics" in aligned_out
        aligned_diag = aligned_out["diagnostics"]
        assert aligned_diag["stage_logits"].shape == (3, 2, 5)
        assert torch.allclose(
            aligned_diag["attributions"][1],
            aligned_diag["stage_logits"][1] - aligned_diag["stage_logits"][0],
        )
        assert torch.allclose(aligned_diag["attributions"].sum(dim=0), aligned_out["logits"])


def test_architecture_package_exports() -> None:
    """Verifies that YodaDecisionEngine and BelnapDecisionHead are exported from architecture."""
    import yoda.architecture as arch

    assert hasattr(arch, "YodaDecisionEngine")
    assert hasattr(arch, "BelnapDecisionHead")
