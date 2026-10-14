"""Unit tests for macro-averaged per-source eval accuracy plumbing."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data.dataset import YodaDecisionDataset, collate_decision_batch
from yoda.data.parquet_dataset import YodaParquetDataset
from yoda.training.trainer import YodaTrainer


@pytest.fixture
def dummy_engine() -> YodaDecisionEngine:
    """Creates a lightweight YodaDecisionEngine for fast testing."""
    torch.manual_seed(42)
    return YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=32,
        num_q_probes=2,
        num_c_probes=4,
        num_k_probes=2,
        num_choices=5,
        n_heads=2,
    )


@pytest.fixture
def synthetic_loader() -> DataLoader[dict[str, Any]]:
    samples = [
        {
            "query": f"Query {i}",
            "state": {"sensor": i * 10},
            "constraints": [f"c{j}: Option {j}" for j in range(5)],
            "target_idx": i % 5,
        }
        for i in range(10)
    ]
    return DataLoader(samples, batch_size=4, shuffle=False, collate_fn=collate_decision_batch)  # type: ignore


def make_parquet(tmp_path: Path) -> Path:
    rows = [
        {
            "query": f"Q{i}",
            "state_json": "{}",
            "choices": ["a", "b", "c"],
            "num_active": 3,
            "target_idx": 0,
            "task_scalar": 1.0,
            "question_type": "choice",
            "source": "src_a" if i % 2 == 0 else "src_b",
        }
        for i in range(4)
    ]
    path = tmp_path / "data.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def make_jsonl(tmp_path: Path) -> Path:
    records = [
        {
            "query": f"Q{i}",
            "context": {"symbolic_state": {}},
            "constraints": ["a", "b", "c"],
            "metadata": {
                "target": "a",
                "source": "src_a" if i % 2 == 0 else "src_b",
                "question_type": "choice",
            },
        }
        for i in range(4)
    ]
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def test_parquet_dataset_includes_source(tmp_path: Path) -> None:
    ds = YodaParquetDataset(make_parquet(tmp_path), shuffle_choices=False)
    sources = {ds[i]["source"] for i in range(len(ds))}
    assert sources == {"src_a", "src_b"}


def test_jsonl_dataset_includes_source(tmp_path: Path) -> None:
    # source=None disables the dataset's default "n4ze3m_synth" source filter.
    ds = YodaDecisionDataset(
        make_jsonl(tmp_path), source=None, max_choices=3, shuffle_choices=False
    )
    assert len(ds) == 4
    sources = {ds[i]["source"] for i in range(len(ds))}
    assert sources == {"src_a", "src_b"}


def test_collate_carries_sources() -> None:
    samples = [
        {
            "query": f"Q{i}",
            "state": {},
            "constraints": ["a", "b"],
            "target_idx": 0,
            "source": f"src_{i}",
        }
        for i in range(2)
    ]
    batch = collate_decision_batch(samples)
    assert batch["sources"] == ["src_0", "src_1"]


def _batch_metrics(
    correct: float,
    total: float,
    source_correct: dict[str, float] | None = None,
    source_total: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "loss": 1.0,
        "ce_loss": 0.5,
        "focal_loss": 0.5,
        "margin_loss": 0.1,
        "belnap_loss": 0.2,
        "ltn_loss": 0.0,
        "assertion_loss": 0.0,
        "correct": correct,
        "total": total,
        "knowledge_sum": 0.0,
        "source_correct": source_correct or {},
        "source_total": source_total or {},
    }


def test_evaluate_macro_accuracy_averages_per_source(
    dummy_engine: YodaDecisionEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Macro accuracy is the unweighted mean of per-source accuracies."""
    samples = [
        {
            "query": "Q0",
            "state": {},
            "constraints": ["a", "b"],
            "target_idx": 0,
            "source": "src_a",
        }
    ] + [
        {
            "query": f"Q{i}",
            "state": {},
            "constraints": ["a", "b"],
            "target_idx": 0,
            "source": "src_b",
        }
        for i in range(1, 4)
    ]
    loader = DataLoader(samples, batch_size=4, shuffle=False, collate_fn=collate_decision_batch)  # type: ignore

    trainer = YodaTrainer(model=dummy_engine, device="cpu")
    monkeypatch.setattr(
        trainer,
        "_compute_loss_and_metrics",
        lambda batch: (
            torch.tensor(0.0),
            _batch_metrics(
                correct=1.0,
                total=4.0,
                source_correct={"src_a": 1.0, "src_b": 0.0},
                source_total={"src_a": 1.0, "src_b": 3.0},
            ),
        ),
    )
    metrics = trainer.evaluate(loader)
    assert metrics["accuracy"] == pytest.approx(0.25)  # pooled: 1/4
    assert metrics["macro_accuracy"] == pytest.approx(0.5)  # macro: (1.0 + 0.0) / 2


def test_evaluate_macro_falls_back_to_pooled_without_source_stats(
    dummy_engine: YodaDecisionEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [
        {
            "query": "Q0",
            "state": {},
            "constraints": ["a", "b"],
            "target_idx": 0,
        }
    ]
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_decision_batch)  # type: ignore

    trainer = YodaTrainer(model=dummy_engine, device="cpu")
    monkeypatch.setattr(
        trainer,
        "_compute_loss_and_metrics",
        lambda batch: (torch.tensor(0.0), _batch_metrics(correct=1.0, total=2.0)),
    )
    metrics = trainer.evaluate(loader)
    assert metrics["macro_accuracy"] == pytest.approx(metrics["accuracy"])


def test_fit_history_includes_macro_accuracy(
    dummy_engine: YodaDecisionEngine,
    synthetic_loader: DataLoader[dict[str, Any]],
) -> None:
    trainer = YodaTrainer(model=dummy_engine, device="cpu")
    history = trainer.fit(train_loader=synthetic_loader, eval_loader=synthetic_loader, epochs=1)
    assert len(history) == 1
    assert "macro_accuracy" in history[0]
    # Single "unknown" source: macro equals pooled.
    assert history[0]["macro_accuracy"] == pytest.approx(history[0]["eval_accuracy"])


def test_independent_path_source_stats(dummy_engine: YodaDecisionEngine) -> None:
    """The candidate-unrolled (independent) path attributes groups to sources correctly.

    Regression test: sources are per candidate row in this path (collated as
    ``candidate_sources``); indexing the per-sample list crashed with IndexError.
    """
    samples = [
        {
            "query": f"Q{i}",
            "state": {},
            "constraints": [f"c{j}" for j in range(3)],
            "target_idx": i % 3,
            "source": "src_a" if i % 2 == 0 else "src_b",
        }
        for i in range(6)
    ]
    batch = collate_decision_batch(samples)
    assert len(batch["candidate_sources"]) == 18  # 6 samples x 3 candidate rows

    trainer = YodaTrainer(model=dummy_engine, device="cpu", independent_eval=True)
    _, metrics = trainer._compute_loss_and_metrics(batch)

    assert metrics["source_total"] == {"src_a": 3.0, "src_b": 3.0}
    assert sum(metrics["source_correct"].values()) == metrics["correct"]
    assert metrics["total"] == 6.0


def test_collapse_pruning_keys_on_macro_accuracy(
    dummy_engine: YodaDecisionEngine,
    synthetic_loader: DataLoader[dict[str, Any]],
) -> None:
    """Collapse pruning fires on low macro accuracy even when pooled accuracy is high."""
    trainer = YodaTrainer(
        model=dummy_engine,
        device="cpu",
        collapse_threshold=0.4,
        collapse_patience_epochs=1,
    )

    def mock_eval(loader: Any) -> dict[str, float]:
        return {
            "loss": 1.0,
            "ce_loss": 0.5,
            "focal_loss": 0.5,
            "margin_loss": 0.1,
            "belnap_loss": 0.2,
            "ltn_loss": 0.0,
            "assertion_loss": 0.0,
            "accuracy": 0.9,
            "eval_accuracy": 0.9,
            "macro_accuracy": 0.3,
            "mean_knowledge": 0.0,
        }

    trainer.evaluate = mock_eval  # type: ignore
    history = trainer.fit(train_loader=synthetic_loader, eval_loader=synthetic_loader, epochs=5)
    assert len(history) == 1  # pruned at epoch 1 despite pooled accuracy 0.9
