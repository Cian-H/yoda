"""Builds the v2 HPO data protocol: group-disjoint, source-stratified splits.

Gate 0 of the post-hardening Optuna study (council verdict, 2026-10-14).

Why this exists:
- ``data/processed/eval.parquet`` is ~54% contaminated: 6,998/13,002 rows are exact
  ``(query, choices, target_idx)`` duplicates of ``train.parquet`` rows (95.4%
  label agreement), concentrated in dwidlee_gen / dwidlee_p2 / kev.
- ``train.parquet`` has 5.04x within-file duplication on ``(query, choices)``
  groups (299,423 rows -> 59,366 unique groups).
- The legacy contiguous-slice split (``split_parquet_dataset``) therefore produced
  single-source eval slices and a memorization-shaped objective.

Protocol:
- Pool = train.parquet + eval.parquet (both ETL-filtered).
- Exact-row dedup on (query, state_json, choices, target_idx).
- Split unit = (whitespace-normalized query, whitespace-normalized choices) group:
  every row of a group lands in exactly one split. This kills within-pool
  duplication leakage and cross-file contamination simultaneously.
- Hash assignment (sha256 of the group key mod 10,000) is reproducible from the
  data alone and source-stratified in expectation.
- Writes ``train.parquet`` / ``optuna_eval.parquet`` / ``test.parquet`` plus
  ``split_metadata.json`` with per-source counts, const-predictor rates, chance
  rates, and the derived collapse threshold (macro const-predictor + 0.02).

Usage:
    uv run python experiments/prepare_hpo_split.py
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

BUCKET_MODULUS = 10_000
GROUP_SEP = "\x1f"
CHOICE_SEP = "\x1e"
COLLAPSE_MARGIN = 0.02

REQUIRED_COLUMNS = ["query", "state_json", "choices", "num_active", "target_idx", "source"]
SPLIT_NAMES = ("train", "optuna_eval", "test")


def normalize_text(text: Any) -> str:
    """Collapses whitespace and strips; keeps case (case folding merges distinct items)."""
    return " ".join(str(text).split())


def group_key_for(query: str, choices_key: str) -> str:
    """Builds the group key from normalized query and choices."""
    return f"{normalize_text(query)}{GROUP_SEP}{choices_key}"


def bucket_for_key(key: str, modulus: int = BUCKET_MODULUS) -> int:
    """Deterministic hash bucket in [0, modulus) from the group key alone."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _add_group_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Adds normalized query/choices helper columns and the group key."""
    return df.with_columns(
        [
            pl.col("query").str.replace_all(r"\s+", " ").str.strip_chars().alias("_q"),
            pl.col("choices")
            .list.eval(pl.element().str.replace_all(r"\s+", " ").str.strip_chars())
            .list.join(CHOICE_SEP)
            .alias("_c"),
        ]
    ).with_columns((pl.col("_q") + GROUP_SEP + pl.col("_c")).alias("_group"))


def build_splits(
    pool: pl.DataFrame,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    modulus: int = BUCKET_MODULUS,
) -> dict[str, pl.DataFrame]:
    """Dedups and hash-assigns pool rows to group-disjoint train/eval/test splits.

    Raises:
        ValueError: If ratios are invalid or required columns are missing.
        AssertionError: If any group key appears in more than one split.
    """
    ratios = (train_ratio, eval_ratio, test_ratio)
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Ratios must be non-negative and sum to 1.0, got {ratios}")
    missing = [c for c in REQUIRED_COLUMNS if c not in pool.columns]
    if missing:
        raise ValueError(f"Pool is missing required columns: {missing}")

    work = _add_group_columns(pool)
    # Exact-row dedup (stringified choices avoid list-column hashing pitfalls).
    work = work.unique(
        subset=["query", "state_json", "_c", "target_idx"], keep="first", maintain_order=True
    )
    work = work.with_columns(
        pl.col("_group")
        .map_elements(lambda k: bucket_for_key(k, modulus), return_dtype=pl.Int64)
        .alias("_bucket")
    )

    train_cut = round(train_ratio * modulus)
    eval_cut = train_cut + round(eval_ratio * modulus)
    work = work.with_columns(
        pl.when(pl.col("_bucket") < train_cut)
        .then(pl.lit("train"))
        .when(pl.col("_bucket") < eval_cut)
        .then(pl.lit("optuna_eval"))
        .otherwise(pl.lit("test"))
        .alias("_split")
    )

    # Group-disjointness is the load-bearing invariant: verify before writing.
    split_groups: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for split, grp in work.select(["_split", "_group"]).rows():
        split_groups[split].add(grp)
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1 :]:
            overlap = split_groups[a] & split_groups[b]
            if overlap:
                raise AssertionError(
                    f"Group leakage between {a} and {b}: {len(overlap)} shared groups"
                )

    drop_cols = ["_q", "_c", "_group", "_bucket", "_split"]
    return {
        name: work.filter(pl.col("_split") == name).drop(drop_cols).sort("source", "query")
        for name in SPLIT_NAMES
    }


def _split_stats(df: pl.DataFrame) -> dict[str, Any]:
    """Per-source row counts, const-predictor rate P(target==0), and chance rate."""
    agg = df.group_by("source").agg(
        [
            pl.len().alias("rows"),
            (pl.col("target_idx") == 0).mean().alias("const_predictor"),
            (1.0 / pl.col("num_active")).mean().alias("chance"),
        ]
    )
    per_source = {
        row["source"]: {
            "rows": int(row["rows"]),
            "const_predictor": round(float(row["const_predictor"]), 6),
            "chance": round(float(row["chance"]), 6),
        }
        for row in agg.iter_rows(named=True)
    }
    return {
        "rows": int(df.height),
        "unique_groups": int(_add_group_columns(df)["_group"].n_unique()),
        "per_source": per_source,
        "pooled_const": round(float((df["target_idx"] == 0).mean()), 6),
        "macro_const": round(float(agg["const_predictor"].mean()), 6),
        "macro_chance": round(float(agg["chance"].mean()), 6),
    }


def compute_metadata(
    splits: dict[str, pl.DataFrame],
    *,
    ratios: tuple[float, float, float],
    modulus: int,
    inputs: list[str],
    pool_rows_before: int,
    pool_rows_after: int,
) -> dict[str, Any]:
    """Builds the split metadata record consumed by the v2 study."""
    per_split = {name: _split_stats(df) for name, df in splits.items()}
    collapse_threshold = round(per_split["optuna_eval"]["macro_const"] + COLLAPSE_MARGIN, 4)
    return {
        "protocol": "v2 group-disjoint hash split",
        "objective": {
            "metric": "macro_accuracy",
            "aggregation": "mean_last2",
            "eval_split": "optuna_eval",
            "test_split": "test (Phase D only)",
        },
        "collapse_threshold": collapse_threshold,
        "collapse_threshold_basis": "macro const-predictor rate of optuna_eval + 0.02",
        "group_key": "(whitespace-normalized query, whitespace-normalized choices)",
        "bucket_modulus": modulus,
        "ratios": {"train": ratios[0], "optuna_eval": ratios[1], "test": ratios[2]},
        "inputs": inputs,
        "pool_rows_before_dedup": pool_rows_before,
        "pool_rows_after_dedup": pool_rows_after,
        "splits": per_split,
    }


def main() -> None:
    """Builds and writes the v2 splits plus split metadata."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-input", default="data/processed/train.parquet")
    parser.add_argument("--eval-input", default="data/processed/eval.parquet")
    parser.add_argument("--output-dir", default="data/processed/v2")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--modulus", type=int, default=BUCKET_MODULUS)
    args = parser.parse_args()

    frames = []
    inputs = []
    for raw in (args.train_input, args.eval_input):
        path = Path(raw)
        if not path.exists():
            raise SystemExit(
                f"Input not found: {path}. The processed parquet inputs are "
                "gitignored derived data — transfer them from a prepared host (or "
                "run the ETL from data/raw where available)."
            )
        frames.append(pl.read_parquet(path))
        inputs.append(str(path))
    pool = pl.concat(frames)
    before = int(pool.height)

    splits = build_splits(
        pool,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        modulus=args.modulus,
    )
    after = sum(int(df.height) for df in splits.values())
    metadata = compute_metadata(
        splits,
        ratios=(args.train_ratio, args.eval_ratio, args.test_ratio),
        modulus=args.modulus,
        inputs=inputs,
        pool_rows_before=before,
        pool_rows_after=after,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.write_parquet(out / f"{name}.parquet")
    metadata_path = out / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Pool: {before:,} rows -> {after:,} rows after exact-row dedup ({len(inputs)} inputs)")
    for name in SPLIT_NAMES:
        stats = metadata["splits"][name]
        print(
            f"  {name:12s} rows={stats['rows']:>8,}  groups={stats['unique_groups']:>7,}  "
            f"macro_const={stats['macro_const']:.4f}  macro_chance={stats['macro_chance']:.4f}"
        )
        for src, s in sorted(stats["per_source"].items()):
            print(f"      {src:18s} rows={s['rows']:>7,}  const={s['const_predictor']:.4f}")
    print(
        f"Derived collapse_threshold = {metadata['collapse_threshold']} "
        f"({metadata['collapse_threshold_basis']})"
    )
    print(f"Wrote splits and metadata to {out}/")


if __name__ == "__main__":
    main()
