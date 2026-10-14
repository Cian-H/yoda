# Candidate Knowledge Gate Lower Bound

- **Area:** Architecture — decision head (ADR-0019, ADR-0022)
- **Refs:** `../adrs/0019-centered-bilinear-candidate-alignment-and-group-relative-scoring.md`, `../adrs/0022-v2-hpo-data-protocol-and-macro-objective.md`, `../../src/yoda/architecture/engine.py`, `../../.docs/prompts/1791937530.implement_hpo_council_verdict.md`

## Context

In the bilinear candidate-alignment head, the epistemic gate is a multiplicative
hard clamp with no lower bound: `cand_gate = (2.0 * cand_k).clamp(max=1.0)`
followed by `logits = raw_diff * cand_gate * logit_scale` (`engine.py`). When
candidate knowledge collapses toward zero, every logit becomes exactly zero,
`argmax` breaks ties at index 0, and accuracy equals the const-predictor rate
`P(target==0)` of the eval slice — which is precisely the 36.5% plateau the
pre-hardening studies recorded. This is a representational dead end: no
hyperparameter regime recovers a signal that has been multiplied to zero. The
council flagged it as the architectural stop-condition for the v2 study: if
Phase-0 trials pin at the const-predictor baseline, the fix belongs here (floor
`cand_gate`, or route the alignment residual around the gate), not in the search
space.

## Deferred because

Fixing the gate is an architectural change to the decision head with its own
regression surface (knowledge gating is load-bearing for calibration and for
the epistemic-uncertainty story), and the v2 study's Phase 0 is explicitly
designed to detect the symptom first: five anchor-config trials with blocking
gates (mean macro accuracy must exceed const-predictor + 0.05, seed sigma <
0.02). Investing in the gate fix before measuring whether the hardened
architecture actually hits the dead end would be speculative.

## Revisit when

Phase 0 sanity results land: if mean macro accuracy across the five anchor
trials is at or below const-predictor + 0.05 (i.e. ~0.3934 on the current
v2 split), implement the gate floor or residual bypass and re-run Phase 0.
