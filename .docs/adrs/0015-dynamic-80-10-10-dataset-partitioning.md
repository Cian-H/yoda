# ADR 0015: Dynamic 80/10/10 Dataset Partitioning and Slice Management

## Status
Accepted

## Date
2026-10-05

## Context
Prior iterations of the training pipeline loaded `--train-path` fully for training and carved out evaluation and test sets from `--eval-path` using hardcoded offsets (`offset=args.max_eval_samples` when `test_path == eval_path`):
1. **Coupled File Assumptions**: The pipeline assumed two separate files existed (`train.parquet` and `eval.parquet`), but then artificially subdivided `eval.parquet` into eval and test partitions.
2. **Data Exhaustion Vulnerability**: If `eval.parquet` had fewer samples than `max_eval_samples + max_test_samples`, the test split would silently be starved or empty, leading to missing post-training held-out evaluations.
3. **Deviation from Standard ML Practices**: Standard practice in deep learning is a clean 80/10/10 Train:Val:Test split on the primary dataset to ensure representative data distributions across all stages without partition leakage.

## Decision
1. **Dynamic Split Boundaries**:
   - Implemented `compute_split_boundaries(total_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)` in `src/yoda/data/parquet_dataset.py`.
   - Computes deterministic, non-overlapping `(offset, count)` tuples that strictly cover the total sample budget without rounding loss ($N_{\text{train}} + N_{\text{eval}} + N_{\text{test}} = N_{\text{total}}$).
2. **Columnar In-Memory Dataset Partitioning**:
   - Extended `YodaParquetDataset` to accept pre-loaded Polars `DataFrame` instances in addition to filesystem paths.
   - Implemented `split_parquet_dataset` in `src/yoda/data/parquet_dataset.py`, which reads the Parquet table once and produces zero-copy columnar slices for train, eval, and test datasets.
   - Automatically enables candidate choice shuffling (`shuffle_choices=True`) only for the training partition while keeping eval and test partitions deterministic (`shuffle_choices=False`).
3. **Training Script Integration**:
   - Updated `experiments/train_yoda.py` to partition the primary dataset (`--train-path`) dynamically using 80/10/10 by default when explicit eval/test paths are omitted.
   - Exposed `--max-samples`, `--train-ratio`, `--val-ratio`, and `--test-ratio` flags.
   - Preserved support for explicit separate evaluation and test paths while eliminating brittle offset carve-out hacks.

## Consequences
- **Positive**: Guarantees zero overlap between training, evaluation, and test DataLoaders.
- **Positive**: Eliminates risk of empty test splits caused by multi-file offset mismatches.
- **Positive**: Accelerates data loading by reading the Parquet file into memory once and taking zero-copy slices.
- **Positive**: Enables single-dataset training workflows from a single unified `.parquet` or `.jsonl` file.
- **Neutral**: When training on the full 291k sample table, 80% yields ~233k training samples, 10% yields ~29k validation samples, and 10% yields ~29k test samples.
