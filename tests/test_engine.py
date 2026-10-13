"""Tests for top-level YodaDecisionEngine and BelnapDecisionHead."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from yoda.architecture.belnap_transformer import BelnapState, BelnapTransformerBlock
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

    def test_epistemic_logit_gating_zero_knowledge_suppression(self) -> None:
        """Verifies zero knowledge (k=0) suppresses logits to 0 regardless of pre-activations."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state_zero = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
        out = head(state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_multi_choice_zero_evidence(self) -> None:
        """Verifies multi-choice head suppresses all choice logits when evidence is zero."""
        head = BelnapDecisionHead(d_model=32, num_choices=5)
        state_zero = BelnapState(e_pos=torch.zeros(3, 4, 32), e_neg=torch.zeros(3, 4, 32))
        out = head(state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_with_bilinear_candidate_states(self) -> None:
        """Verifies bilinear candidate scoring suppresses logits for zero-evidence candidates."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        final_state = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        cand_state_zero = BelnapState(e_pos=torch.zeros(2, 32), e_neg=torch.zeros(2, 32))
        out = head(final_state, candidate_states=cand_state_zero)
        assert torch.all(torch.abs(out["logits"]) < 1e-4)
        assert torch.all(out["knowledge"] == 0.0)

    def test_epistemic_logit_gating_unattenuated_decisive_state(self) -> None:
        """Verifies that a decisive state (k=0.5, truth=1.0) transmits non-zero positive logits."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        state_decisive = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        out = head(state_decisive)
        assert out["knowledge"].item() > 0.4
        assert out["logits"].item() > 0.0


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
        assert engine.num_reasoning_blocks == 1
        assert engine.conflation_weight == 0.1
        assert engine.d_hidden_multiplier == 2.0
        assert engine.dropout == 0.0
        assert len(engine.reasoning_layers) == 1
        first_layer = cast(nn.ModuleDict, engine.reasoning_layers[0])
        assert engine.context_reasoning is first_layer["context"]
        assert engine.constraint_reasoning is first_layer["constraint"]
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

    def test_forward_pass_with_variable_criteria_masking(
        self, dummy_engine: YodaDecisionEngine
    ) -> None:
        """Verifies that variable criteria are masked out from logits, truth, and knowledge."""
        queries = ["query 1", "query 2"]
        states = [{"state": 1}, {"state": 2}]
        # Sample 0 has 2 active choices, sample 1 has 3 active choices (padded to 5)
        constraints = [
            [
                "opt_a: A",
                "opt_b: B",
                "none: Unused option",
                "none: Unused option",
                "none: Unused option",
            ],
            ["opt_1: 1", "opt_2: 2", "opt_3: 3", "none: Unused option", "none: Unused option"],
        ]

        out = dummy_engine(queries=queries, states=states, constraints=constraints)

        assert "active_mask" in out
        active_mask = out["active_mask"]
        assert active_mask[0].tolist() == [True, True, False, False, False]
        assert active_mask[1].tolist() == [True, True, True, False, False]

        # Logits for inactive positions should be heavily negative (-1e9)
        assert torch.all(out["logits"][0, 2:] <= -1e8)
        assert torch.all(out["logits"][1, 3:] <= -1e8)

        # Inactive truth and knowledge must be exactly 0.0 (ignorance)
        assert torch.all(out["truth"][0, 2:] == 0.0)
        assert torch.all(out["knowledge"][0, 2:] == 0.0)
        assert torch.all(out["truth"][1, 3:] == 0.0)
        assert torch.all(out["knowledge"][1, 3:] == 0.0)

        # Choice must be within active criteria
        assert out["choice"][0].item() in [0, 1]
        assert out["choice"][1].item() in [0, 1, 2]

    def test_dynamic_candidate_affinity_scoring(self) -> None:
        """Verifies dynamic candidate affinity scoring across variable candidate counts."""
        affinity_engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_q_probes=2,
            num_c_probes=4,
            num_k_probes=4,
            num_choices=5,
            n_heads=2,
            use_candidate_affinity=True,
        )

        queries = ["choose target"]
        states = [{"sensor": 5}]
        constraints = [
            [
                "crit_1: Alpha",
                "crit_2: Beta",
                "crit_3: Gamma",
                "none: Unused option",
                "none: Unused option",
            ]
        ]

        out = affinity_engine(queries=queries, states=states, constraints=constraints)

        assert out["logits"].shape == (1, 5)
        assert out["choice"].shape == (1,)
        assert out["choice"].item() in [0, 1, 2]
        assert torch.all(out["logits"][0, 3:] <= -1e8)

        # Verify gradient flow
        loss = out["logits"].sum()
        loss.backward()
        assert affinity_engine.cand_proj.weight.grad is not None
        assert not torch.isnan(affinity_engine.cand_proj.weight.grad).any()

    def test_native_encoder_dimension_adoption(self) -> None:
        """Verifies engine adopts native full encoder dimension with Identity adapter.

        Applies when embed_dim is None.
        """
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
        ):
            engine = YodaDecisionEngine(text_model_name="mock-transformer", embed_dim=None)
            assert engine.embed_dim == 384
            assert isinstance(engine.text_encoder.adapter, nn.Identity)
            assert engine.q_mpa.input_proj.in_features == 384
            assert engine.c_mpa.input_proj.in_features == 384
            assert engine.k_mpa.input_proj.in_features == 384
            assert engine.context_reasoning.d_model == 384
            assert engine.constraint_reasoning.d_model == 384
            assert engine.decision_head.d_model == 384

    def test_parameterized_topology_initialization(self) -> None:
        """Verifies custom topology parameters and reasoning block construction."""
        engine = YodaDecisionEngine(
            embed_dim=64,
            num_reasoning_blocks=3,
            conflation_weight=0.25,
            d_hidden_multiplier=3.0,
            residual_weight=0.6,
            dropout=0.1,
        )
        assert engine.num_reasoning_blocks == 3
        assert engine.conflation_weight == 0.25
        assert engine.d_hidden_multiplier == 3.0
        assert engine.residual_weight == 0.6
        assert engine.dropout == 0.1
        assert len(engine.reasoning_layers) == 3

        for layer in engine.reasoning_layers:
            layer_dict = cast(nn.ModuleDict, layer)
            ctx_block = cast(BelnapTransformerBlock, layer_dict["context"])
            assert ctx_block.d_model == 64
            assert ctx_block.residual_weight == 0.6
            assert ctx_block.dropout == 0.1
            assert ctx_block.ffn.w_1.out_features == int(64 * 3.0)
            assert ctx_block.ffn.conflation_weight == 0.25
            assert ctx_block.ffn.dropout.p == 0.1

            const_block = cast(BelnapTransformerBlock, layer_dict["constraint"])
            assert const_block.d_model == 64
            assert const_block.residual_weight == 0.6
            assert const_block.dropout == 0.1
            assert const_block.ffn.w_1.out_features == int(64 * 3.0)
            assert const_block.ffn.conflation_weight == 0.25
            assert const_block.ffn.dropout.p == 0.1

    def test_multi_block_reasoning_gradient_flow(self) -> None:
        """Verifies backpropagation through multiple cascaded reasoning blocks."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=64,
            num_q_probes=4,
            num_c_probes=8,
            num_k_probes=4,
            num_choices=5,
            num_reasoning_blocks=2,
            dropout=0.0,
        )
        queries = ["query 1", "query 2"]
        states = [{"key": "val1"}, {"key": "val2"}]
        constraints = [["c1", "c2"], ["c3", "c4"]]

        out = engine(queries=queries, states=states, constraints=constraints)
        loss = out["logits"].sum()
        loss.backward()

        for idx, layer in enumerate(engine.reasoning_layers):
            layer_dict = cast(nn.ModuleDict, layer)
            ctx_block = cast(BelnapTransformerBlock, layer_dict["context"])
            const_block = cast(BelnapTransformerBlock, layer_dict["constraint"])
            assert ctx_block.cross_attn.w_q_pos.weight.grad is not None, (
                f"Block {idx} ctx grad missing"
            )
            assert const_block.cross_attn.w_q_pos.weight.grad is not None, (
                f"Block {idx} const grad missing"
            )
            assert not torch.isnan(ctx_block.cross_attn.w_q_pos.weight.grad).any()
            assert not torch.isnan(const_block.cross_attn.w_q_pos.weight.grad).any()

    def test_multi_block_diagnostics(self) -> None:
        """Verifies diagnostic trajectories and attribution conservation with multi-blocks."""
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=64,
            num_q_probes=4,
            num_c_probes=8,
            num_k_probes=4,
            num_choices=5,
            num_reasoning_blocks=2,
        )
        queries = ["query 1", "query 2"]
        states = [{"key": "val1"}, {"key": "val2"}]
        constraints = [["c1", "c2"], ["c3", "c4"]]

        out = engine(
            queries=queries,
            states=states,
            constraints=constraints,
            return_diagnostics=True,
        )
        assert "diagnostics" in out
        diag = out["diagnostics"]

        expected_stages = [
            "post_pooling",
            "post_context_0",
            "post_constraint_0",
            "post_context_1",
            "post_constraint_1",
        ]
        assert diag["stage_names"] == expected_stages

        stage_logits = diag["stage_logits"]
        attributions = diag["attributions"]
        assert stage_logits.shape == (5, 2, 5)
        assert attributions.shape == (5, 2, 5)

        # Final stage matches output logits
        assert torch.allclose(stage_logits[-1], out["logits"])
        # Sum of attributions telescopes exactly to final logits
        assert torch.allclose(attributions.sum(dim=0), out["logits"])


def test_architecture_package_exports() -> None:
    """Verifies that YodaDecisionEngine and BelnapDecisionHead are exported from architecture."""
    import yoda.architecture as arch

    assert hasattr(arch, "YodaDecisionEngine")
    assert hasattr(arch, "BelnapDecisionHead")


class TestBilinearCandidateAlignment:
    """Verifies R4 bilinear candidate-query alignment and group-relative scoring."""

    def test_zero_affinity_rests_at_neither_evidence(self) -> None:
        """Orthogonal query and candidate evidence produce zero choice evidence."""
        d_model = 32
        head = BelnapDecisionHead(d_model=d_model, num_choices=1)

        q = torch.zeros(1, d_model)
        q[0, : d_model // 2] = 1.0
        c = torch.zeros(1, d_model)
        c[0, d_model // 2 :] = 1.0
        # Centered candidate coordinates are orthogonal to the query evidence
        c_state = BelnapState(e_pos=c, e_neg=torch.zeros(1, d_model))

        out = head(BelnapState(e_pos=q, e_neg=torch.zeros(1, d_model)), candidate_states=c_state)
        assert out["knowledge"].item() < 0.05, (
            f"Orthogonal alignment must rest at Neither, got k={out['knowledge'].item()}"
        )

    def test_candidate_information_mass_gates_logits(self) -> None:
        """Zero-mass candidate evidence suppresses logits despite query evidence."""
        head = BelnapDecisionHead(d_model=32, num_choices=1)
        query = BelnapState(e_pos=torch.ones(2, 32), e_neg=torch.zeros(2, 32))
        # Candidate 0 carries contrast; candidate 1 sits at the evidence midpoint
        cand = BelnapState(
            e_pos=torch.tensor([[1.0] * 32, [0.5] * 32]),
            e_neg=torch.tensor([[0.0] * 32, [0.5] * 32]),
        )

        out = head(query, candidate_states=cand)
        assert abs(out["logits"][1].item()) < 1e-4, "Midpoint candidate must be suppressed"
        assert out["logits"][0].item() > out["logits"][1].item()

    def test_logit_scale_amplifies_ranking_signal(self) -> None:
        """logit_scale multiplies bilinear logits without touching evidence bounds."""
        head = BelnapDecisionHead(d_model=32, num_choices=1, logit_scale=4.0)
        query = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))
        cand = BelnapState(e_pos=torch.ones(1, 32), e_neg=torch.zeros(1, 32))

        out = head(query, candidate_states=cand)
        assert out["logits"].item() > 0.5
        assert out["choice_pos"].max().item() <= 1.0
        assert out["choice_neg"].max().item() <= 1.0

    def test_engine_independent_mode_ranks_within_group(self) -> None:
        """Engine scores candidates group-relative: distinct candidates get distinct logits."""
        torch.manual_seed(7)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        queries = ["Which action is safest?"] * 2
        states = [{"system": "boiler"}] * 2
        candidates = ["Vent valve", "Close valve"]

        out = engine(queries=queries, states=states, candidates=candidates)
        assert out["logits"].shape == (2,)
        assert not torch.allclose(out["logits"][0:1], out["logits"][1:2]), (
            "Candidates sharing a context must receive group-relative distinct scores"
        )

    def test_engine_single_candidate_group_avoids_collapse(self) -> None:
        """Single-candidate groups keep raw evidence (no dimensional collapse)."""
        torch.manual_seed(8)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        out = engine(queries=["Solo query"], states=[{"id": 1}], candidates=["Only option"])
        assert out["logits"].shape == (1,)
        assert torch.isfinite(out["logits"]).all()
        assert out["knowledge"].shape == (1,)

    def test_engine_unmentioned_candidates_stay_near_neither(self) -> None:
        """Unmentioned candidates retain k < 0.20 at initialization."""
        torch.manual_seed(11)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )

        out = engine(
            queries=["Select active navigation route."] * 3,
            states=[{"route_a": "clear"}] * 3,
            candidates=["Route A: active", "Route X: unmentioned", "Route Y: unmentioned"],
        )
        for idx in (1, 2):
            assert out["knowledge"][idx].item() < 0.20, (
                f"Unmentioned candidate {idx} knowledge must stay < 0.20, "
                f"got {out['knowledge'][idx].item():.4f}"
            )


class TestContextEncodingDeduplication:
    """Verifies R5 context encoding deduplication and cross-call memoization."""

    def test_deduplicated_outputs_match_uncached_engine(self) -> None:
        """Cached and uncached engines must produce identical logits (to float noise)."""
        torch.manual_seed(31)
        uncached = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
            use_context_cache=False,
        )
        cached = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )
        cached.load_state_dict(uncached.state_dict())
        uncached.eval()
        cached.eval()

        queries = ["Shared question about pressure"] * 3
        states = [{"system": "boiler"}] * 3
        candidates = ["Vent valve", "Close valve", "Ignore alarm"]

        with torch.no_grad():
            out_uncached = uncached(queries=queries, states=states, candidates=candidates)
            out_cached = cached(queries=queries, states=states, candidates=candidates)
            out_warm = cached(queries=queries, states=states, candidates=candidates)

        assert torch.allclose(out_uncached["logits"], out_cached["logits"], atol=1e-5)
        assert torch.allclose(out_cached["logits"], out_warm["logits"], atol=1e-6), (
            "Warm cache must return identical outputs"
        )

    def test_context_states_are_memoized_under_no_grad(self) -> None:
        """Gradient-free forwards populate the context cache; repeated calls hit it."""
        torch.manual_seed(32)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )
        engine.eval()
        queries = ["Dedup question"] * 2
        states = [{"sensor": "active"}] * 2
        candidates = ["Option A", "Option B"]

        assert engine._context_cache == {}
        with torch.no_grad():
            engine(queries=queries, states=states, candidates=candidates)
            assert len(engine._context_cache) == 1, "One unique context must be memoized"
            engine(queries=queries, states=states, candidates=candidates)

    def test_gradient_enabled_forward_invalidates_cache(self) -> None:
        """A training (grad-enabled) forward must clear memoized context states."""
        torch.manual_seed(33)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )
        engine.eval()
        queries = ["Invalidate question"] * 2
        states = [{"mode": "test"}] * 2
        candidates = ["Choice one", "Choice two"]

        with torch.no_grad():
            engine(queries=queries, states=states, candidates=candidates)
        assert len(engine._context_cache) == 1

        engine(queries=queries, states=states, candidates=candidates)  # grad enabled
        assert engine._context_cache == {}, "Grad-enabled forward must invalidate the cache"

    def test_independent_mode_encodes_contexts_once_per_batch(self) -> None:
        """Unique (query, state) pairs share a single encoder pass per batch."""
        from unittest.mock import MagicMock

        torch.manual_seed(34)
        engine = YodaDecisionEngine(
            text_model_name="dummy",
            embed_dim=32,
            num_choices=1,
            n_heads=2,
        )
        engine.eval()
        original_forward = engine.text_encoder.forward
        mock_forward = MagicMock(side_effect=original_forward)
        engine.text_encoder.forward = mock_forward

        queries = ["Common query"] * 4
        states: list[dict[str, Any]] = [{"env": "sim"}] * 4
        candidates = ["Option 1", "Option 2", "Option 3", "Option 4"]

        engine(queries=queries, states=states, candidates=candidates)

        assert mock_forward.call_count <= 2, (
            f"Query and state contexts should be encoded in one pass, "
            f"called {mock_forward.call_count} times"
        )
