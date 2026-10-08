# Fix Nimble Target Ingestion and Implement Paired Contrastive Evaluation

- **Area:** Ingest and Dataset Pipeline (ADR-0003, ADR-0010)
- **Refs:** `../adrs/0003-dataset-ingestion-and-normalization.md`, `../adrs/0010-decoupled-columnar-parquet-dataset.md`, `../adrs/0013-independent-choice-assessment-architecture.md`, `../../src/yoda/data/ingest.py`, `../../src/yoda/data/etl.py`, `../../tests/test_data_ingest.py`

## Context

An audit of the aggregated dataset revealed that the Nimble dataset (`bespokelabsai/nimble`) is currently not contributing any samples to training or evaluation Parquet tables. In `src/yoda/data/ingest.py`, `parse_nimble_record` attempts to extract ground-truth targets from `raw_row.get("teacher", {}).get("answers", {})`, but real raw Nimble records encode gold decisions under `raw_row["reference"]["target"]`. Because targets default to `None` in `aggregated_train.jsonl` and `aggregated_eval.jsonl`, `ParquetETLPipeline` fails to match choice indices and silently skips all Nimble records during ETL. Furthermore, Nimble is constructed from paired base/counterfactual scenarios (`[id]-base` vs `[id]-counterfactual`) with fine-grained evidence certificates, which the pipeline currently treats as isolated, unpaired rows without counterfactual consistency loss or paired flip evaluation. Addressing this will restore Nimble's data to the training corpus and equip the decision engine with explicit counterfactual invariance supervision.

## Deferred because

The immediate focus was on inspecting the dataset balance and verifying contrastive data existence in the active corpus. Fixing the ingestion logic requires updating `DatasetIngestionPipeline.parse_nimble_record`, correcting the synthetic mock in `tests/test_data_ingest.py`, re-running dataset streaming and Parquet ETL, and designing paired batch collation that was scoped out of the initial exploration.

## Revisit when

Next commit that touches `src/yoda/data/ingest.py` or before the next multi-dataset training sweep requiring counterfactual sensitivity evaluation.
