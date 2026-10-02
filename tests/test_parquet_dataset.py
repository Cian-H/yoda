"""Tests for ParquetETLPipeline and YodaParquetDataset."""

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from yoda.data import (
    ParquetETLPipeline,
    YodaParquetDataset,
    collate_decision_batch,
    match_target_to_constraints,
    run_etl,
)


@pytest.fixture
def mock_jsonl(tmp_path: Path) -> Path:
    jsonl_file = tmp_path / "test_samples.jsonl"
    samples = [
        {
            "query": "Is this action safe?",
            "constraints": ["A1: Proceed", "A2: Abort", "A3: Escalate"],
            "metadata": {"target": "A2", "question_type": "choice", "source": "unit_test"},
            "context": {"symbolic_state": {"flag": True}},
        },
        {
            "query": "Select appropriate tier",
            "constraints": ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"],
            "metadata": {"target": "Tier 1", "question_type": "choice", "source": "unit_test"},
            "state": {"env": "prod"},
        },
        {
            "query": "Unmatched query",
            "constraints": ["Opt A", "Opt B"],
            "metadata": {"target": "Missing", "question_type": "null"},
        },
    ]

    with jsonl_file.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return jsonl_file


def test_target_matching() -> None:
    raw = ["A1: Proceed", "A2: Abort", "A3: Escalate"]
    assert match_target_to_constraints("A2", raw) == 1
    assert match_target_to_constraints("a1", raw) == 0
    assert match_target_to_constraints("A3: Escalate", raw) == 2
    assert match_target_to_constraints("NonExistent", raw) is None


def test_parquet_etl_pipeline(mock_jsonl: Path, tmp_path: Path) -> None:
    out_parquet = tmp_path / "out.parquet"
    etl = ParquetETLPipeline(max_choices=5)
    df = etl.transform_jsonl_to_parquet(mock_jsonl, out_parquet)

    assert isinstance(df, pl.DataFrame)
    # Third sample is unmatched and should be skipped
    assert len(df) == 2
    assert out_parquet.exists()

    # Check columns
    assert "query" in df.columns
    assert "choices" in df.columns
    assert "target_idx" in df.columns
    assert "task_scalar" in df.columns
    assert "state_json" in df.columns

    # Verify first row standardization
    row0 = df.row(0, named=True)
    assert row0["query"] == "Is this action safe?"
    assert row0["target_idx"] == 1
    assert row0["task_scalar"] == 1.0  # "choice" task
    assert row0["num_active"] == 3
    assert len(row0["choices"]) == 5
    assert row0["choices"][:3] == ["A1: Proceed", "A2: Abort", "A3: Escalate"]
    assert row0["choices"][3] == "none: Unused option"


def test_yoda_parquet_dataset_loading_and_shuffling(mock_jsonl: Path, tmp_path: Path) -> None:
    out_parquet = tmp_path / "out.parquet"
    etl = ParquetETLPipeline(max_choices=5)
    etl.transform_jsonl_to_parquet(mock_jsonl, out_parquet)

    # 1. Unshuffled dataset
    ds = YodaParquetDataset(out_parquet, shuffle_choices=False)
    assert len(ds) == 2

    item0 = ds[0]
    assert item0["query"] == "Is this action safe?"
    assert item0["target_idx"] == 1
    assert item0["task_scalar"] == 1.0
    assert item0["state"] == {"flag": True}
    assert len(item0["constraints"]) == 5

    # 2. Shuffled dataset
    torch.manual_seed(123)
    ds_shuffled = YodaParquetDataset(out_parquet, shuffle_choices=True)
    shuffled_item0 = ds_shuffled[0]

    # Target label string must match target index
    target_str = shuffled_item0["constraints"][shuffled_item0["target_idx"]]
    assert "A2: Abort" in target_str
    # Unused choices remain at the tail
    assert shuffled_item0["constraints"][-1] == "none: Unused option"


def test_collate_decision_batch_with_parquet_dataset(mock_jsonl: Path, tmp_path: Path) -> None:
    out_parquet = tmp_path / "out.parquet"
    etl = ParquetETLPipeline(max_choices=5)
    etl.transform_jsonl_to_parquet(mock_jsonl, out_parquet)

    ds = YodaParquetDataset(out_parquet)
    batch = collate_decision_batch([ds[0], ds[1]])

    assert len(batch["queries"]) == 2
    assert len(batch["states"]) == 2
    assert len(batch["constraints"]) == 2
    assert batch["target_indices"].shape == (2,)
    assert batch["task_scalars"].shape == (2, 1)


def test_run_etl_caching_and_invalidation(mock_jsonl: Path, tmp_path: Path) -> None:
    out_parquet = tmp_path / "cached.parquet"
    meta_file = tmp_path / ".cached.parquet.meta.json"

    # 1. Initial run: cache miss, produces parquet and meta
    res1 = run_etl(input_path=mock_jsonl, output_path=out_parquet, max_choices=5)
    assert res1 == out_parquet
    assert out_parquet.exists()
    assert meta_file.exists()
    initial_mtime = out_parquet.stat().st_mtime_ns

    # 2. Second run with same args: cache hit, file untouched
    res2 = run_etl(input_path=mock_jsonl, output_path=out_parquet, max_choices=5)
    assert res2 == out_parquet
    assert out_parquet.stat().st_mtime_ns == initial_mtime

    # 3. Argument mismatch (e.g. max_choices changed): cache invalidated and re-executed
    res3 = run_etl(input_path=mock_jsonl, output_path=out_parquet, max_choices=4)
    assert res3 == out_parquet
    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["max_choices"] == 4

    # 4. Input file modification: cache invalidated and re-executed
    import time

    time.sleep(0.01)
    mock_jsonl.touch()
    res4 = run_etl(input_path=mock_jsonl, output_path=out_parquet, max_choices=4)
    assert res4 == out_parquet

    # 5. Missing input file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        run_etl(input_path=tmp_path / "nonexistent.jsonl")
