# 3. Dataset Ingestion, Licensing Audit, and Schema Normalization

- **Status**: Accepted
- **Date**: 2026-10-01

## Context

Training and evaluating the System 1 decision engine requires high-quality, typed decision datasets that model fast intuitive decision-making under structured context, criteria, and symbolic constraints.

Candidate datasets were identified across three main hubs:
1. Nimble dataset (`https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md`)
2. Kev model dataset (`https://github.com/jaredpalmer/kev`)
3. The `awesome-system-one-models` census (`https://github.com/pozapas/awesome-system-one-models#-datasets-and-benchmarks`)

To prevent legal contamination and maintain architectural consistency:
1. All candidate datasets must be strictly audited for permissive open-source licensing (e.g. Apache-2.0, MIT). Non-commercial, unstated, or copyleft share-alike (e.g. CC-BY-SA 4.0) licenses must be rejected and purged.
2. All records must be normalized into the canonical `DecisionPayload` schema (`query`, `context` [semantic_embedding, symbolic_state, history], `constraints`, `metadata`).

## Decision

We implement a dedicated, reproducible ingestion and normalization pipeline in `yoda.data.ingest.DatasetIngestionPipeline` and establish the vetted corpus:

1. **Licensing and Suitability Audit**:
   - **Nimble (`bespokelabsai/nimble`)**: **Accepted** (Apache-2.0). 11,000+ domain decision records with state, questions, criteria, and teacher annotations.
   - **Kev (`jaredpalmer/kev`)**: **Accepted** (Apache-2.0). Evaluation and training suites (`decision-v1`, `decision-v2`) with rubric criteria and target labels.
   - **Tasksource Jev Decisions (`tasksource/tasksource-jev-typed-decisions`)**: **Rejected & Purged** (License: `other`). Contains heterogeneous upstream academic datasets with mixed non-commercial conditions.
   - **SystemOne (`roskosmos19/SystemOne`)**: **Rejected & Purged** (No license stated). Unlicensed repository poses IP ambiguity.
   - **Typed Decisions (`pngwn/typed-decisions`)**: **Rejected & Purged** (License: CC-BY-SA-4.0). Share-alike copyleft condition poses risk to downstream model weights.
   - **Image Beans Pilot (`FaroukMoc2/jev-stage2-image-beans-pilot`)**: **Rejected & Purged** (Domain mismatch). Pure vision dataset incompatible with text `DecisionPayload` architecture.

2. **Schema Normalization**:
   - `query`: String extracted from question instructions or rubrics, normalized across string, dict, or list variants via `_normalize_instructions`.
   - `context`: `QueryContext` containing `symbolic_state` (dictionary representation of scenario/dialogue/timeline), empty initial `semantic_embedding`, and `history`.
   - `constraints`: List of formatted criteria strings (`key: description` or explicit rules).
   - `metadata`: Source provenance (`nimble` or `kev`), record ID, domain/family, question type, and ground-truth target label.

3. **Aggregation Outputs**:
   - `data/processed/aggregated_train.jsonl`: 9,108 canonical training records.
   - `data/processed/aggregated_eval.jsonl`: 2,616 canonical evaluation records.
   - `data/processed/manifest.json`: Ingestion provenance manifest detailing accepted/rejected sources and record counts.

## Consequences

- The training corpus is 100% Apache-2.0 compliant and free from copyleft or non-commercial restrictions.
- All 11,724 aggregated records validate strictly against Pydantic `DecisionPayload`.
- Raw and processed datasets are stored in `data/` and excluded from version control via `.gitignore`, preserving lightweight repository size while remaining reproducible via `DatasetIngestionPipeline().process_and_aggregate()`.
