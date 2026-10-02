"""Tests for decision payload schema and architecture models."""

from yoda.architecture.schema import DecisionPayload, QueryContext


def test_decision_payload_instantiation() -> None:
    """Verify that a standard DecisionPayload validates and instantiates correctly."""
    payload = DecisionPayload(
        query="evaluate_state",
        context=QueryContext(
            semantic_embedding=[0.1, -0.4, 0.8],
            symbolic_state={"location": "node_a", "active": True},
            history=[{"action": "init"}],
        ),
        constraints=["non_negative", "bounded"],
        metadata={"priority": 1},
    )

    assert payload.query == "evaluate_state"
    assert len(payload.context.semantic_embedding) == 3
    assert payload.context.symbolic_state["location"] == "node_a"
    assert "non_negative" in payload.constraints


def test_decision_payload_defaults() -> None:
    """Verify minimal instantiation with defaults."""
    payload = DecisionPayload(query="test_query")
    assert payload.context.semantic_embedding == []
    assert payload.constraints == []
    assert payload.metadata == {}
