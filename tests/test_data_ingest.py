"""Tests for DatasetIngestionPipeline and schema transformation."""

from __future__ import annotations

from pathlib import Path

from yoda.architecture.schema import DecisionPayload
from yoda.data.ingest import DatasetIngestionPipeline


def test_parse_nimble_record() -> None:
    """Verifies transformation of raw Nimble record into valid DecisionPayload."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_nimble = {
        "id": "nimble-test-1",
        "domain": "commerce",
        "family": "commerce-01",
        "input": {
            "state": "Customer wants to return an unopened item 35 days after purchase.",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Determine the routing action under the policy.",
                    "criteria": {
                        "deny": "Deny return over 45 days.",
                        "manager_review": "Review allowed for 31-45 days if unopened.",
                    },
                }
            },
        },
        "teacher": {
            "answers": {
                "decision": {
                    "choice": "manager_review",
                    "confidence": 0.98,
                }
            }
        },
    }

    payloads = pipeline.parse_nimble_record(raw_nimble)
    assert len(payloads) == 1
    p = payloads[0]

    assert isinstance(p, DecisionPayload)
    assert p.query == "Determine the routing action under the policy."
    assert p.context.symbolic_state == {
        "text": "Customer wants to return an unopened item 35 days after purchase."
    }
    assert "deny: Deny return over 45 days." in p.constraints
    assert "manager_review: Review allowed for 31-45 days if unopened." in p.constraints
    assert p.metadata["source"] == "nimble"
    assert p.metadata["target"] == "manager_review"


def test_parse_kev_record() -> None:
    """Verifies transformation of raw Kev record into valid DecisionPayload."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_kev = {
        "state": {"subject": "Duplicate charge", "body": "I was charged twice."},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Billing and refund inquiries", "tech": "Technical bugs"},
                "label": "billing",
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this an urgent issue?",
                "label": False,
            },
        },
    }

    payloads = pipeline.parse_kev_record(raw_kev, file_tag="v1_test")
    assert len(payloads) == 2

    # Verify first question
    p0 = payloads[0]
    assert p0.query == "Which team should handle this?"
    assert p0.context.symbolic_state["subject"] == "Duplicate charge"
    assert p0.metadata["source"] == "kev"
    assert p0.metadata["target"] == "billing"

    # Verify second question
    p1 = payloads[1]
    assert p1.query == "Is this an urgent issue?"
    assert p1.metadata["target"] is False


def test_parse_dwidlee_row() -> None:
    """Verifies transformation of tabular dwidlee record into valid DecisionPayload."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    row = {
        "task": "debate.winner",
        "state": '{"side_a": {"label": "support"}, "side_b": {"label": "oppose"}}',
        "instructions": "Given the claims, who should win the debate?",
        "criteria_keys": ["A", "B"],
        "criteria_values": ["oppose: Side B", "support: Side A"],
        "label_alias": "A",
        "label_key": "oppose",
        "meta": '{"gym": "debate_judge"}',
    }

    payload = pipeline.parse_dwidlee_row(row, source_tag="dwidlee_gen")
    assert isinstance(payload, DecisionPayload)
    assert payload.query == "Given the claims, who should win the debate?"
    assert payload.context.symbolic_state == {
        "side_a": {"label": "support"},
        "side_b": {"label": "oppose"},
    }
    assert payload.constraints == ["oppose: Side B", "support: Side A"]
    assert payload.metadata["source"] == "dwidlee_gen"
    assert payload.metadata["target"] == "oppose"
    assert payload.metadata["label_alias"] == "A"


def test_parse_n4ze3m_record() -> None:
    """Verifies transformation of n4ze3m synthetic review record."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_row = {
        "state_id": "synth-001",
        "domain": "product reviews",
        "state": '{"review": "Great pack but strap broke."}',
        "questions": '{"kept_item": {"type": "noul", "instructions": "Did user keep item?"}}',
        "gold": '{"kept_item": false}',
    }

    payloads = pipeline.parse_n4ze3m_record(raw_row)
    assert len(payloads) == 1
    p = payloads[0]
    assert isinstance(p, DecisionPayload)
    assert p.query == "Did user keep item?"
    assert p.metadata["source"] == "n4ze3m_synth"
    assert p.metadata["target"] is False


def test_parse_mghafiri_record() -> None:
    """Verifies transformation of mghafiri scenario record."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_row = {
        "id": "scenario-001",
        "domain": "permits",
        "pattern": "verification",
        "state": '{"permit": "residential"}',
        "questions": '{"need_permit": {"type": "noul", "instructions": "Is permit needed?"}}',
        "targets": '{"need_permit": {"noul": 0.95}}',
    }

    payloads = pipeline.parse_mghafiri_record(raw_row)
    assert len(payloads) == 1
    p = payloads[0]
    assert isinstance(p, DecisionPayload)
    assert p.query == "Is permit needed?"
    assert p.metadata["source"] == "mghafiri_scenarios"
    assert p.metadata["target"] == 0.95


def test_dataset_audit_postures(tmp_path: Path) -> None:
    """Verifies accepted datasets have documented postures and rejected datasets are flagged."""
    pipeline = DatasetIngestionPipeline(data_root=tmp_path)
    accepted = pipeline.ACCEPTED_DATASETS
    rejected = pipeline.REJECTED_DATASETS

    assert "kev" in accepted
    assert accepted["kev"]["license"] == "Apache-2.0"
    assert "nimble" in accepted
    assert "research-fair-use" in accepted["nimble"]["posture"]
    assert "dwidlee_phase2" in accepted
    assert accepted["dwidlee_phase2"]["license"] == "Apache-2.0"
    assert "dwidlee_general" in accepted
    assert accepted["dwidlee_general"]["license"] == "MIT"
    assert "n4ze3m_synth" in accepted
    assert "mghafiri_scenarios" in accepted

    assert "tasksource" in rejected
    assert "roskosmos19" in rejected
    assert "pngwn" in rejected
    assert "FaroukMoc2_image_beans" in rejected
