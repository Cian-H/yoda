"""Tests for Direct Logit Attribution (DLA) diagnostics in yoda.xai."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from yoda.xai.dla import DirectLogitAttribution, run_dla_evaluation


class MockDLAEngine(nn.Module):
    """Mock engine outputting diagnostics for DLA testing."""

    def __init__(self, num_choices: int = 5, num_stages: int = 3) -> None:
        super().__init__()
        self.num_choices = num_choices
        self.num_stages = num_stages

    def forward(
        self,
        queries: list[str],
        states: list[dict[str, Any]],
        constraints: list[list[str]],
        task_scalars: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        batch_size = len(queries)
        stage_logits = torch.randn(self.num_stages, batch_size, self.num_choices)
        stage_knowledge = torch.full((self.num_stages, batch_size, self.num_choices), 0.5)
        attributions = torch.randn(self.num_stages, batch_size, self.num_choices)

        res: dict[str, Any] = {
            "logits": stage_logits[-1],
            "truth": torch.sigmoid(stage_logits[-1]),
            "knowledge": stage_knowledge[-1],
        }
        if return_diagnostics:
            res["diagnostics"] = {
                "stage_logits": stage_logits,
                "stage_knowledge": stage_knowledge,
                "attributions": attributions,
            }
        return res


def test_dla_evaluator_computes_metrics() -> None:
    model: Any = MockDLAEngine()
    batch = {
        "queries": ["q1", "q2"],
        "states": [{}, {}],
        "constraints": [[], []],
        "target_indices": torch.tensor([0, 1]),
        "task_scalars": torch.tensor([[1.0], [1.0]]),
    }
    loader = DataLoader(
        [batch],  # type: ignore
        batch_size=1,
        collate_fn=lambda x: x[0],
    )

    evaluator = DirectLogitAttribution()
    report = evaluator.evaluate(model=model, eval_loader=loader, device="cpu")

    assert report["total_samples"] == 2
    assert len(report["stage_accuracies"]) == 3
    assert len(report["avg_target_logits"]) == 3
    assert len(report["avg_knowledge"]) == 3
    assert len(report["avg_attributions"]) == 3

    formatted = evaluator.format_report(report)
    assert "DIRECT LOGIT ATTRIBUTION" in formatted
    assert "Stage 0 (Post-Pooling)" in formatted
    assert "Stage 2 (Post-Constraint)" in formatted


def test_dla_evaluator_handles_empty_loader() -> None:
    model: Any = MockDLAEngine()
    loader: DataLoader[dict[str, Any]] = DataLoader(
        [],  # type: ignore
    )
    evaluator = DirectLogitAttribution()
    report = evaluator.evaluate(model=model, eval_loader=loader, device="cpu")
    assert report == {}
    assert evaluator.format_report(report) == "No DLA metrics available."


def test_run_dla_evaluation_convenience_function() -> None:
    model: Any = MockDLAEngine()
    batch = {
        "queries": ["q1"],
        "states": [{}],
        "constraints": [[]],
        "target_indices": torch.tensor([0]),
    }
    loader = DataLoader(
        [batch],  # type: ignore
        batch_size=1,
        collate_fn=lambda x: x[0],
    )
    report = run_dla_evaluation(model=model, eval_loader=loader, device="cpu", print_report=False)
    assert report["total_samples"] == 1
