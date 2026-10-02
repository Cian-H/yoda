# ADR 0010: Decoupled Columnar Parquet ETL and Dataset Pipeline

## Status
Accepted

## Date
2026-10-02

## Context
Prior iterations of the training pipeline loaded raw dataset records directly from JSONL files (`aggregated_train.jsonl`, 280MB) line-by-line during runtime:
1. **CPU & I/O Overhead**: Calling `json.loads` on hundreds of thousands of lines on every run incurred significant Python GIL overhead and slow disk seeks.
2. **String Matching Bottleneck**: For every sample, fuzzy prefix matching and colon splitting was performed repeatedly to resolve target labels to choice indices.
3. **Decoupled Architecture Constraint**: While caching precomputed embeddings into disk tables was considered, baking dense embedding vectors into dataset files violates model decoupling, permanently locking the dataset to a specific text encoder backbone, tokenizer, and embedding dimension.

## Decision
1. **Offline Columnar Parquet ETL**:
   - Implemented `ParquetETLPipeline` in `src/yoda/data/etl.py` using `polars`.
   - Performs offline JSON parsing, target label matching (`target_idx: Int64`), choice standardization/padding to 5 choices (`choices: List[Utf8]`), and task scalar projection (`task_scalar: Float32`).
   - Compresses data into native columnar Parquet using ZSTD (`train.parquet`, `eval.parquet`).
   - Explicitly does **not** embed text into vectors during ETL, keeping the data completely backbone-agnostic.
2. **High-Throughput Parquet Dataset**:
   - Implemented `YodaParquetDataset` in `src/yoda/data/parquet_dataset.py`, exported in `yoda.data`.
   - Uses `polars.read_parquet()` for zero-copy memory-mapped columnar loading.
   - Preserves runtime candidate choice permutation (`shuffle_choices: bool`) by permuting choice list slices in memory.
3. **Training Integration**:
   - Added `experiments/run_etl.py` CLI script to execute the offline transformation.
   - Updated `experiments/train_yoda.py` to seamlessly auto-detect `.parquet` or `.jsonl` file extensions, and added `--num-workers` and `--pin-memory` flags.

## Consequences
- **Positive**: 17.5× storage compression (from 280MB JSONL down to 16MB Parquet for 291k samples).
- **Positive**: Dataset loading time dropped from seconds to ~0.08s via Polars columnar memory-mapping.
- **Positive**: 100% text backbone decoupling preserved: any embedding model (`all-MiniLM-L6-v2`, `bge`, `nomic`) can be swapped via CLI without re-generating the dataset.
- **Positive**: DataLoader can utilize `num_workers > 0` and pinned memory without GIL-bound file read contention.
- **Negative**: Requires a one-time ETL step before training on newly ingested datasets.
