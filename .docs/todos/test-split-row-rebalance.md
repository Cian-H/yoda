# Rebalance v2 test Split by Rows

- **Area:** Data protocol (`experiments/prepare_hpo_split.py`, ADR-0022)
- **Refs:** `../../experiments/prepare_hpo_split.py`, `../adrs/0022-v2-hpo-data-protocol-and-macro-objective.md`, `../../.docs/prompts/1791937530.implement_hpo_council_verdict.md`

## Context

The v2 split assigns whole `(query, choices)` groups to splits by hash, which is
the correct unit for leakage, but rows-per-group varies sharply by source
(nimble ~13.9, dwidlee_p2 ~6.4, n4ze3m_synth ~1.0). The result: `test.parquet`
is dwidlee_p2-heavy by rows (55,360 rows, ~85% p2) even though group counts are
balanced (80/10/10), while `optuna_eval.parquet` is more evenly mixed (18,611
rows). Group-disjointness — the load-bearing property — is intact; only row
proportions skew, and the test split is touched exclusively by Phase-D
finalist validation.

## Deferred because

Rebalancing would require row-weighted group assignment (or source-stratified
bucketing) in the split builder, adding complexity for a set that only Phase D
touches and whose skew is cosmetic for a final readout. The macro-averaged
objective that actually drives the search uses `optuna_eval.parquet`, which is
well mixed.

## Revisit when

Phase D validation readouts look source-skewed and per-source test accuracy is
needed for the final report — then add row-weighted (or explicitly
source-stratified) group assignment to `prepare_hpo_split.py` and regenerate.
