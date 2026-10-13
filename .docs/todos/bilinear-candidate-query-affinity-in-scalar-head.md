# Eliminate Query-Context Cancellation via Bilinear Alignment in Scalar Head

- **Area:** Decision Head & Independent Choice Evaluation (ADR-0011, ADR-0013)
- **Refs:** `../plan.md`, `../adrs/0011-active-criteria-masking-and-candidate-affinity.md`, `../adrs/0013-independent-choice-assessment-architecture.md`, `../../src/yoda/architecture/engine.py`, `../../tests/test_independent_eval.py`

## Context

In `src/yoda/architecture/engine.py`, during independent choice evaluation (`is_independent = True`), candidate options are unrolled into flat samples and scored using `self.scalar_head` (`num_choices == 1`). However, `BelnapDecisionHead.forward` executes a simple linear projection `w_pos(pooled_pos)` whenever `num_choices == 1`, ignoring candidate probes entirely. Because `final_state` is computed via convex residual joins between context-enriched queries $q_{\text{ctx}}$ and candidate representations $x_{\text{cand}, i}$, linear scoring evaluates $z_i = (1 - \alpha) w^\top q_{\text{ctx}} + \alpha w^\top x_{\text{cand}, i}$. When `FocalLoss` or `MarginLoss` compares candidates within the same question group ($z_y - z_j$), the query-context term $(1 - \alpha) w^\top q_{\text{ctx}}$ is identical across candidates and cancels out completely. Consequently, the model cannot match query context to candidates and ranks options solely on generic candidate plausibility, creating a hard 50% accuracy ceiling. Passing pooled candidate states `k_state` into `head(final_state, candidate_states=k_state)` and computing explicit bilinear Belnap inner products breaks this ceiling.

## Deferred because

This change requires adapting `BelnapDecisionHead.forward` to accept 3D probe tensors in independent mode, updating `YodaDecisionEngine.forward` to pass `k_state` to `self.scalar_head`, and validating against `tests/test_independent_eval.py`. Staged as a P0 milestone in Phase 3 of the roadmap.

## Revisit when

Next commit modifying `src/yoda/architecture/engine.py` or before benchmarking independent choice evaluation accuracy.
