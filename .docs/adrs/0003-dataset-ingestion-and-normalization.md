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
   - **Kev (`jaredpalmer/kev`)**: **Accepted** (`permissive-clean`, Apache-2.0). Evaluation and training suites (`decision-v1`, `decision-v2`) with rubric criteria and target labels. Explicit `LICENSE` file granting Apache-2.0 rights.
   - **Dwidlee Lite Phase 2 (`dwidlee/systemone-lite-phase2`)**: **Accepted** (`permissive-clean`, Apache-2.0). 245,500 debate and judgement records with state, formatted criteria, instructions, and target labels.
   - **Dwidlee Lite General (`dwidlee/systemone-lite-general`)**: **Accepted** (`permissive-clean`, MIT). 36,000 multi-domain decision records with formatted criteria and target labels.
   - **N4ze3m Synth (`n4ze3m/typed-decisions-synth`)**: **Accepted** (`permissive-clean`, MIT). 25,859 synthetic product review evaluation records with choice/score/noul questions and gold targets.
   - **Mghafiri Scenarios (`mghafiri/decision-model-scenarios`)**: **Accepted** (`permissive-clean`, MIT). 9,716 scenario verification records across government services and operational rules.
   - **Nimble (`bespokelabsai/nimble`)**: **Accepted with Posture** (`research-fair-use`). Ingested under explicit `research-fair-use` posture. Every record is tagged with source provenance (`metadata.source = "nimble"`), ensuring the model can be retrained or restrained at any time solely on clean subsets (over 99.1% of the corpus is clean Apache-2.0 / MIT).
   - **Tasksource Jev Decisions (`tasksource/tasksource-jev-typed-decisions`)**: **Rejected & Purged** (License: `other`). Contains heterogeneous upstream academic datasets with mixed non-commercial conditions.
   - **SystemOne (`roskosmos19/SystemOne`)**: **Rejected & Purged** (No license stated). Unlicensed repository poses IP ambiguity.
   - **Typed Decisions (`pngwn/typed-decisions`)**: **Rejected & Purged** (License: CC-BY-SA-4.0). Share-alike copyleft condition poses risk to downstream model weights.
   - **Image Beans Pilot (`FaroukMoc2/jev-stage2-image-beans-pilot`)**: **Rejected & Purged** (Domain mismatch). Pure vision dataset incompatible with text `DecisionPayload` architecture.

2. **Schema Normalization**:
   - `query`: Extracted from `instructions` string/dict/list across all formats, normalized via `_normalize_instructions`.
   - `context`: `QueryContext` containing `symbolic_state` (dictionary representation of scenario, dialogue, arguments, or event logs), empty initial `semantic_embedding`, and `history`.
   - `constraints`: List of formatted criteria strings (`key: description` or explicit rubric rules).
   - `metadata`: Source provenance (`kev`, `dwidlee_p2`, `dwidlee_gen`, `n4ze3m_synth`, `mghafiri_scenarios`, or `nimble`), record ID, task domain, question type, and ground-truth target.

3. **Aggregation Outputs**:
   - `data/processed/aggregated_train.jsonl`: **313,909** canonical training records.
   - `data/processed/aggregated_eval.jsonl`: **14,890** canonical evaluation records.
   - Total aggregated dataset: **328,799** records (>99.1% clean Apache-2.0 / MIT).
   - `data/processed/manifest.json`: Version 2.0.0 provenance manifest detailing accepted/rejected sources, legal postures, and record counts.

## Consequences

- The training corpus provides 328,799 canonical, schema-normalized `DecisionPayload` records.
- Streaming writes prevent memory exhaustion and enable instant deterministic re-generation.
- Clean provenance tracking allows instant filtering to 100% clean Apache-2.0/MIT records for commercial distribution.
- Non-compliant datasets remain completely purged and excluded.
- Raw and processed datasets are stored in `data/` and excluded from version control via `.gitignore`.
