"""Tests for DatasetIngestionPipeline and schema transformation."""

from pathlib import Path

from yoda.architecture.schema import DecisionPayload
from yoda.data.ingest import DatasetIngestionPipeline


def test_parse_nimble_record() -> None:
    """Verifies transformation of a real-schema Nimble record into a DecisionPayload.

    Real Nimble records encode gold decisions under ``reference.target`` (a
    criteria key), carry dialogue-event states, and identify the
    base/counterfactual variant via the record id suffix.
    """
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_nimble = {
        "id": "scale-diverse-013-002-base",
        "domain": "commerce",
        "family": "commerce-01",
        "variant": "base",
        "input": {
            "state": [
                {
                    "speaker": "Account reviewer",
                    "text": "The subscriber's account records a $24 retention offer.",
                },
            ],
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose the authorized resolution.",
                    "criteria": {
                        "apply_credit_keep_annual": "Add the $24 credit and preserve the plan.",
                        "deny": "Deny the correction request.",
                    },
                }
            },
        },
        "reference": {
            "human_reviewed": False,
            "source": "policy",
            "target": "apply_credit_keep_annual",
        },
    }

    payloads = pipeline.parse_nimble_record(raw_nimble)
    assert len(payloads) == 1
    p = payloads[0]

    assert isinstance(p, DecisionPayload)
    assert p.query == "Choose the authorized resolution."
    assert p.context.symbolic_state == {
        "dialogue_or_list": [
            {
                "speaker": "Account reviewer",
                "text": "The subscriber's account records a $24 retention offer.",
            },
        ]
    }
    assert "apply_credit_keep_annual: Add the $24 credit and preserve the plan." in p.constraints
    assert "deny: Deny the correction request." in p.constraints
    assert p.metadata["source"] == "nimble"
    assert p.metadata["target"] == "apply_credit_keep_annual", (
        "Gold decision must be read from reference.target, not teacher.answers"
    )


def test_parse_nimble_record_extracts_pair_metadata() -> None:
    """Base and counterfactual twins must share a pair_id with distinct variants."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    def make_record(record_id: str) -> dict[str, object]:
        return {
            "id": record_id,
            "input": {
                "state": "Policy context.",
                "questions": {
                    "decision": {
                        "type": "choice",
                        "instructions": "Choose the resolution.",
                        "criteria": {"refund": "Issue refund.", "deny": "Deny request."},
                    }
                },
            },
            "reference": {"target": "refund"},
        }

    base = pipeline.parse_nimble_record(make_record("scale-diverse-014-001-base"))[0]
    counter = pipeline.parse_nimble_record(make_record("scale-diverse-014-001-counterfactual"))[0]

    assert base.metadata["pair_id"] == "scale-diverse-014-001"
    assert base.metadata["variant"] == "base"
    assert counter.metadata["pair_id"] == "scale-diverse-014-001"
    assert counter.metadata["variant"] == "counterfactual", (
        "Counterfactual twins must be identifiable for paired contrastive evaluation"
    )


def test_parse_nimble_record_without_gold_target_yields_none() -> None:
    """Records lacking a reference target must still produce payloads with target None."""
    pipeline = DatasetIngestionPipeline(data_root="data")

    raw_nimble = {
        "id": "scale-diverse-015-001-base",
        "input": {
            "state": "Context.",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"a": "Option A.", "b": "Option B."},
                }
            },
        },
    }

    payloads = pipeline.parse_nimble_record(raw_nimble)
    assert len(payloads) == 1
    assert payloads[0].metadata["target"] is None


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
