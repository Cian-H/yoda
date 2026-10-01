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

(Append new ADRs as `NNNN-<kebab-slug>.md` and add a row here in the same commit. See `.agents/rules/workflow.md` for when an ADR is required.)
