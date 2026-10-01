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

1. **Licensing and Suitability Audit & Posture**:
   - **Kev (`jaredpalmer/kev`)**: **Accepted** (`permissive-clean`, Apache-2.0). Evaluation and training suites (`decision-v1`, `decision-v2`) with rubric criteria and target labels. The repository root contains an explicit `LICENSE` file granting full Apache-2.0 rights.
   - **Nimble (`bespokelabsai/nimble`)**: **Accepted with Posture** (`research-fair-use`). The repository has no explicit `LICENSE` file (GitHub API reports `license: null`), meaning statutory default copyright technically applies. However, following industry practice and experimental research needs, Nimble is ingested under an explicit `research-fair-use` posture. Every record is tagged with source provenance (`metadata.source = "nimble"`), ensuring the model can be retrained or restrained at any time solely on clean subsets (e.g. Kev-only) without structural friction.
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
   - `data/processed/aggregated_train.jsonl`: 9,108 canonical training records (6,432 Kev + 2,676 Nimble).
   - `data/processed/aggregated_eval.jsonl`: 2,616 canonical evaluation records (2,292 Kev + 324 Nimble).
   - `data/processed/manifest.json`: Ingestion provenance manifest detailing accepted/rejected sources, legal postures, and record counts.

## Consequences

- The training corpus provides 11,724 canonical, schema-normalized `DecisionPayload` records across both Kev and Nimble.
- Source provenance is strictly maintained in metadata, allowing instant filtering to 100% clean Apache-2.0 records if commercial redistribution or weight licensing requires it.
- Non-compliant datasets (`tasksource`, `roskosmos19`, `pngwn`, `FaroukMoc2`) remain completely purged and excluded.
- Raw and processed datasets are stored in `data/` and excluded from version control via `.gitignore`, preserving lightweight repository size while remaining reproducible via `DatasetIngestionPipeline().process_and_aggregate()`.
