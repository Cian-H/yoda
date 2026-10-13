# Nimble Paired Contrastive Evaluation

- **Area:** Losses, ETL Schema & Training Pipeline (ADR-0013, ADR-0019)
- **Refs:** `../adrs/0013-independent-choice-assessment-architecture.md`, `../adrs/0019-centered-bilinear-candidate-alignment-and-group-relative-scoring.md`, `../../src/yoda/data/etl.py`, `../../src/yoda/data/dataset.py`, `../../src/yoda/training/losses.py`

## Context

Nimble records are constructed as paired base/counterfactual scenarios: each
`<pair_id>-base` record shares its resolution policy with a
`<pair_id>-counterfactual` twin whose evidence flips exactly one factual span,
and the gold decision changes accordingly. Ingestion now preserves the pairing
(`pair_id` and `variant` in payload metadata), but the pipeline still treats the
twins as isolated rows: the ETL drops payload metadata when writing Parquet, and
no loss term or evaluation protocol exploits the pairing. A counterfactual
consistency penalty (the engine's decision distribution should flip between
paired twins in step with the flipped evidence) and a paired flip evaluation
metric would give the engine explicit counterfactual invariance supervision and a
sensitivity measure the benchmark suite currently lacks.

## Deferred because

Implementing paired evaluation requires: extending the Parquet ETL schema with
`pair_id`/`variant` columns (and `YodaParquetDataset`/`YodaDecisionDataset` to
carry them), a collation mode that keeps twins adjacent across batches, a
counterfactual consistency loss in the trainer, and flip-accuracy metrics in
`evaluate()`. The ingestion-side prerequisite (correct targets and preserved
pairing) landed first to restore Nimble to the corpus; the modeling work was
scoped out of that fix.

## Revisit when

Next commit that extends the ETL/Parquet schema or the training losses, or
before a training sweep that reports counterfactual sensitivity.
