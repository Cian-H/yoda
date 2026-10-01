---
name: xai-decision-audit
description: Audit and explain System 1 heuristic decisions using attribution scores, decision path provenance, and prune rationale traces. Use when debugging why an intuitive heuristic was selected or rejected, or generating explainable decision artifacts.
---

# Explainability & Decision Auditing (XAI)

Procedure for auditing System 1 decisions and extracting explainable attribution traces.

## Purpose

System 1 reasoning operates quickly and instinctually. To ensure reliability and debug unexpected outputs, the XAI module provides post-hoc and intrinsic explanations of decision pathways.

## Workflow

1. **Trace Capture**:
   Inspect the decision output payload for provenance metadata:
   - Evaluated candidate paths.
   - Symbolic constraints applied vs satisfied.
   - Epistemic uncertainty estimate.
   - Attention/attribution weights across `QueryContext.semantic_embedding`.

2. **Audit Verification**:
   - Check if the decision prune rationale is justified by the symbolic state.
   - Ensure sensitive attributes are not leaked in explanation logs.

3. **Reporting**:
   Generate structured audit summaries when investigating decision edge cases.
