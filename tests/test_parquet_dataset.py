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
    compute_split_boundaries,
    match_target_to_constraints,
    run_etl,
    split_parquet_dataset,
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


def test_compute_split_boundaries_standard() -> None:
    # 20,000 samples with 80/10/10 split
    tr, val, te = compute_split_boundaries(20000, 0.8, 0.1, 0.1)
    assert tr == (0, 16000)
    assert val == (16000, 2000)
    assert te == (18000, 2000)
    assert tr[1] + val[1] + te[1] == 20000

    # 291,470 samples
    tr, val, te = compute_split_boundaries(291470, 0.8, 0.1, 0.1)
    assert tr == (0, 233176)
    assert val == (233176, 29147)
    assert te == (262323, 29147)
    assert tr[1] + val[1] + te[1] == 291470


def test_compute_split_boundaries_edge_cases() -> None:
    # 0 samples
    assert compute_split_boundaries(0) == ((0, 0), (0, 0), (0, 0))

    # Small sample counts
    tr1, v1, te1 = compute_split_boundaries(1)
    assert tr1[1] + v1[1] + te1[1] == 1

    tr2, v2, te2 = compute_split_boundaries(2)
    assert tr2[1] + v2[1] + te2[1] == 2

    tr3, v3, te3 = compute_split_boundaries(3)
    assert tr3 == (0, 1)
    assert v3 == (1, 1)
    assert te3 == (2, 1)
    assert tr3[1] + v3[1] + te3[1] == 3

    # Invalid ratios sum
    with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
        compute_split_boundaries(100, 0.5, 0.1, 0.1)


def test_split_parquet_dataset_non_overlapping(tmp_path: Path) -> None:
    # Generate 100 mock samples
    records = []
    for i in range(100):
        records.append({
            "query": f"Query {i}",
            "choices": [
                f"Choice {i}_A",
                f"Choice {i}_B",
                f"Choice {i}_C",
                "none: Unused",
                "none: Unused",
            ],
            "target_idx": 0,
            "task_scalar": 1.0,
            "state_json": json.dumps({"idx": i}),
            "num_active": 3,
        })
    df = pl.DataFrame(records)
    parquet_path = tmp_path / "mock_100.parquet"
    df.write_parquet(parquet_path)

    # Split 80/10/10
    train_ds, val_ds, test_ds = split_parquet_dataset(
        parquet_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    assert len(train_ds) == 80
    assert len(val_ds) == 10
    assert len(test_ds) == 10

    # Ensure non-overlapping queries
    train_queries = {train_ds[i]["query"] for i in range(len(train_ds))}
    val_queries = {val_ds[i]["query"] for i in range(len(val_ds))}
    test_queries = {test_ds[i]["query"] for i in range(len(test_ds))}

    assert len(train_queries.intersection(val_queries)) == 0
    assert len(train_queries.intersection(test_queries)) == 0
    assert len(val_queries.intersection(test_queries)) == 0

    # Test with max_samples cap and per-split overrides
    tr2, val2, te2 = split_parquet_dataset(
        parquet_path,
        max_samples=50,
        max_train_samples=30,
        max_eval_samples=4,
        max_test_samples=4,
    )
    assert len(tr2) == 30
    assert len(val2) == 4
    assert len(te2) == 4

