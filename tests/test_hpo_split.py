"""Unit tests for the v2 HPO split builder (experiments/prepare_hpo_split.py)."""

import polars as pl
import pytest

from experiments.prepare_hpo_split import (
    bucket_for_key,
    build_splits,
    compute_metadata,
    group_key_for,
    normalize_text,
)

CHOICE_SEP = "\x1e"


def make_pool(n_groups: int = 400) -> pl.DataFrame:
    """Synthetic pool: unique groups across two sources, one exact dup, one state variant."""
    rows = []
    for i in range(n_groups):
        rows.append(
            {
                "query": f"Query {i}",
                "state_json": "{}",
                "choices": [f"c{j} for {i}" for j in range(3)],
                "num_active": 3,
                "target_idx": i % 3,
                "task_scalar": 1.0,
                "question_type": "choice",
                "source": "src_a" if i % 2 == 0 else "src_b",
            }
        )
    rows.append(dict(rows[0]))  # exact duplicate row
    rows.append({**rows[1], "state_json": '{"x": 1}'})  # same group, different state
    return pl.DataFrame(rows)


def group_of(query: str, choices: list[str]) -> str:
    """Recomputes the split group key the same way the builder does."""
    return group_key_for(query, CHOICE_SEP.join(choices))


def test_normalize_text_collapses_whitespace_but_keeps_case() -> None:
    assert normalize_text("  Hello   World \n") == "Hello World"
    assert normalize_text("Query 7") == "Query 7"


def test_bucket_for_key_is_deterministic_and_bounded() -> None:
    key = group_of("Query 1", ["a", "b"])
    assert bucket_for_key(key) == bucket_for_key(key)
    assert 0 <= bucket_for_key(key) < 10_000
    assert bucket_for_key(key, modulus=100) < 100


def test_exact_duplicate_rows_removed() -> None:
    pool = make_pool()
    splits = build_splits(pool)
    assert sum(df.height for df in splits.values()) == pool.height - 1


def test_groups_are_disjoint_and_complete() -> None:
    pool = make_pool()
    splits = build_splits(pool)
    split_groups = {
        name: {group_of(r["query"], r["choices"]) for r in df.iter_rows(named=True)}
        for name, df in splits.items()
    }
    names = list(split_groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (split_groups[a] & split_groups[b]), f"leak between {a} and {b}"
    expected = {group_of(r["query"], r["choices"]) for r in pool.iter_rows(named=True)}
    assert set().union(*split_groups.values()) == expected


def test_state_variant_stays_in_same_split() -> None:
    pool = make_pool()
    splits = build_splits(pool)
    variant_query = "Query 1"  # rows[1] and its state variant share this group
    owners = [
        name
        for name, df in splits.items()
        if any(r["state_json"] == '{"x": 1}' for r in df.iter_rows(named=True))
    ]
    assert len(owners) == 1
    owner = owners[0]
    base_rows = [r for r in splits[owner].iter_rows(named=True) if r["query"] == variant_query]
    assert len(base_rows) == 2  # original row + state variant, never split apart


def test_assignment_is_deterministic() -> None:
    pool = make_pool()
    first = build_splits(pool)
    second = build_splits(pool)
    for name in first:
        assert first[name]["query"].to_list() == second[name]["query"].to_list()


def test_ratios_are_approximately_honored() -> None:
    pool = make_pool(n_groups=400)
    splits = build_splits(pool)
    assert abs(splits["train"].height - 0.8 * 401) < 40
    assert abs(splits["optuna_eval"].height - 0.1 * 401) < 30
    assert abs(splits["test"].height - 0.1 * 401) < 30


def test_invalid_ratios_raise() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        build_splits(make_pool(), train_ratio=0.5, eval_ratio=0.1, test_ratio=0.1)


def test_missing_columns_raise() -> None:
    pool = make_pool().drop("source")
    with pytest.raises(ValueError, match="missing required columns"):
        build_splits(pool)


def test_metadata_reports_const_rates_and_collapse_threshold() -> None:
    pool = make_pool()
    splits = build_splits(pool)
    metadata = compute_metadata(
        splits,
        ratios=(0.8, 0.1, 0.1),
        modulus=10_000,
        inputs=["a.parquet", "b.parquet"],
        pool_rows_before=pool.height,
        pool_rows_after=sum(df.height for df in splits.values()),
    )
    eval_stats = metadata["splits"]["optuna_eval"]
    assert metadata["collapse_threshold"] == round(eval_stats["macro_const"] + 0.02, 4)
    assert metadata["objective"]["metric"] == "macro_accuracy"
    assert metadata["objective"]["aggregation"] == "mean_last2"
    assert metadata["objective"]["eval_split"] == "optuna_eval"
    assert set(metadata["splits"]) == {"train", "optuna_eval", "test"}
    for stats in metadata["splits"].values():
        assert set(stats["per_source"]) == {"src_a", "src_b"}
        assert 0.0 <= stats["macro_const"] <= 1.0
        assert 0.0 <= stats["macro_chance"] <= 1.0
    assert metadata["pool_rows_after_dedup"] == pool.height - 1
