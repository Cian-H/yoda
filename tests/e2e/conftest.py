"""Shared fixtures and test utilities for Yoda E2E test suites."""

from typing import Any

import pytest
import torch

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data import collate_decision_batch


@pytest.fixture
def dummy_engine_independent() -> YodaDecisionEngine:
    """Creates a lightweight YodaDecisionEngine in independent candidate mode (num_choices=1)."""
    torch.manual_seed(42)
    return YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=1,
        n_heads=2,
        num_reasoning_blocks=1,
    )


@pytest.fixture
def dummy_engine_multichoice() -> YodaDecisionEngine:
    """Creates a lightweight YodaDecisionEngine in multi-choice mode (num_choices=5)."""
    torch.manual_seed(42)
    return YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=5,
        n_heads=2,
        num_reasoning_blocks=1,
    )


@pytest.fixture
def synthetic_unrolled_batch() -> dict[str, Any]:
    """Generates an unrolled independent evaluation batch containing two groups."""
    raw_samples = [
        {
            "query": "Which safety action is required?",
            "state": {"pressure": "high", "system": "nominal"},
            "constraints": [
                "Vent valve",
                "Increase feed",
                "Scram reactor",
                "none: Unused",
                "none: Unused",
            ],
            "target_idx": 0,
            "num_active": 3,
            "task_scalar": 1.0,
        },
        {
            "query": "Select routing path",
            "state": {"congestion": "severe"},
            "constraints": [
                "Route North",
                "Route South",
                "none: Unused",
                "none: Unused",
                "none: Unused",
            ],
            "target_idx": 1,
            "num_active": 2,
            "task_scalar": 1.0,
        },
    ]
    return collate_decision_batch(raw_samples)


@pytest.fixture
def synthetic_multichoice_batch() -> dict[str, Any]:
    """Generates a legacy fixed-dimension multi-choice batch."""
    return {
        "queries": ["Determine protocol", "Verify parameters"],
        "states": [{"level": 3, "mode": "auto"}, {"level": 1, "mode": "manual"}],
        "constraints": [
            ["Option A", "Option B", "Option C", "Option D", "none: Unused"],
            ["Choice 1", "Choice 2", "Choice 3", "none: Unused", "none: Unused"],
        ],
        "target_indices": torch.tensor([1, 0], dtype=torch.long),
        "task_scalars": torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        "active_mask": torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, True, False, False],
            ],
            dtype=torch.bool,
        ),
    }
