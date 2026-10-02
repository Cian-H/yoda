# Architecture Decision Records

This directory holds the [Architecture Decision Records (ADRs)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) for the project. Each ADR is a short markdown file capturing one structural decision: the context that prompted it, what was decided, and the consequences. Read these before making structural changes.

## Index

| # | Title | Status | Date |
| --- | --- | --- | --- |
| [0000](./0000-adr-template.md) | ADR Template (do not cite) | Template | — |
| [0001](./0001-architecture-foundations.md) | Architecture Foundations & System 1 Decision Engine | Accepted | 2026-10-01 |
| [0002](./0002-building-blocks-attention-and-belnap.md) | Core Building Blocks: Multihead Pooled Attention and Fuzzy Belnap Bilattice | Accepted | 2026-10-01 |
| [0003](./0003-dataset-ingestion-and-normalization.md) | Dataset Ingestion, Licensing Audit, and Schema Normalization | Accepted | 2026-10-01 |
| [0004](./0004-belnap-bilattice-native-transformer.md) | Belnap Bilattice Native Transformer & Decoupled Ingestion | Accepted | 2026-10-01 |
| [0005](./0005-belnap-multihead-pooled-attention.md) | Belnap Multihead Pooled Attention (Belnap-MPA) | Accepted | 2026-10-01 |
| [0006](./0006-softexp-and-mish-activation-architecture.md) | SoftExp and Mish Activation Upgrades for Gradient Flow and Evidence Modeling | Accepted | 2026-10-02 |
| [0007](./0007-convex-combination-residual-joins.md) | Convex Combination Residual Joins for Epistemic Bilattices | Accepted | 2026-10-02 |
| [0008](./0008-modular-submodule-realignment.md) | Modular Submodule Realignment (NeSy, Data, and XAI) | Accepted | 2026-10-02 |
| [0009](./0009-decoupled-lightning-adapter.md) | Decoupled PyTorch Lightning Adapter and Frozen Backbone Architecture | Accepted | 2026-10-02 |
| [0010](./0010-decoupled-columnar-parquet-dataset.md) | Decoupled Columnar Parquet ETL and Dataset Pipeline | Accepted | 2026-10-02 |
| [0011](./0011-active-criteria-masking-and-candidate-affinity.md) | Active Criteria Masking and Dynamic Candidate Epistemic Affinity Scoring | Accepted | 2026-10-02 |
| [0012](./0012-focal-margin-loss-and-onecycle-lr-scheduling.md) | Focal-Margin Loss Hybrid and OneCycleLR Annealing | Accepted | 2026-10-02 |

(Append new ADRs as `NNNN-<kebab-slug>.md` and add a row here in the same commit. See `.agents/rules/workflow.md` for when an ADR is required.)
