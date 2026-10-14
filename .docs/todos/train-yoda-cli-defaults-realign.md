# Realign train_yoda.py CLI Defaults with the M4 Caps

- **Area:** Training CLI (`experiments/train_yoda.py`)
- **Refs:** `../../experiments/train_yoda.py`, `../adrs/0022-v2-hpo-data-protocol-and-macro-objective.md`, `../../.docs/prompts/1791937530.implement_hpo_council_verdict.md`

## Context

Several `train_yoda.py` CLI defaults exceed the ranges the M4 sweep realignment
established as stable: `--lr` defaults to 1e-3 (above the 5e-4 cap),
`--decisiveness-weight` defaults to 0.1 (2x the 0.05 cap), and `--min-lr`
defaults to 1e-4, which inverts the cosine floor for any `lr < 1e-4` because
the scheduler's single scalar `eta_min` is shared across the backbone and head
param groups. The v2 study is immune — it passes every value explicitly — but
direct (non-study) training invocations silently inherit the unsafe defaults,
and the silent-default failure mode is exactly what produced the stale narrow
study (its trials ran at decisiveness 0.1, outside its own realigned range).

## Deferred because

Changing the defaults alters the behavior of existing direct training
invocations and of e2e gates that train with parser defaults, so it needs its
own pass with the gate suite re-run, not a drive-by inside the HPO-protocol
change. The study harness — the consumer that matters for the next experiment —
already passes everything explicitly.

## Revisit when

The next direct (non-study) training run is planned, or any commit touches
`build_parser()` in `experiments/train_yoda.py` — align the defaults with the
M4-realigned ranges and re-run the e2e gates.
