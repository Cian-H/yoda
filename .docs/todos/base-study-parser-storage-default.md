# Fix build_base_study_parser --storage Default

- **Area:** Hyperopt harness (`src/yoda/hyperopt/optuna.py`, ADR-0016)
- **Refs:** `../../src/yoda/hyperopt/optuna.py`, `../adrs/0016-hyperopt-module-and-optuna-harness.md`, `../adrs/0022-v2-hpo-data-protocol-and-macro-objective.md`, `../../.docs/prompts/1791937530.implement_hpo_council_verdict.md`

## Context

`build_base_study_parser` accepts a `default_storage` parameter but wires it
only into the `--storage` flag's help text — the argparse default is hardcoded
to `None`. A study that omits `--storage` therefore falls through to
`init_study_and_metadata`'s fallback `sqlite:///data/optuna.db`, silently
ignoring the study's declared default (the narrow study's metadata records
`data/optuna_narrow.db` only because `--storage` was passed explicitly). The v2
study is immune: it resolves storage explicitly via `resolve_storage` (flag >
`YODA_OPTUNA_STORAGE` env > local sqlite default). The quirk remains in the
shared parser for any future study built on it.

## Deferred because

Fixing the base parser changes `args.storage` semantics for the remaining broad
study (`experiments/optuna_study.py`), which currently relies on the
fallthrough; that is a separate behavioral change with its own verification
pass, not a drive-by inside the v2-protocol change. The v2 study does not need
the fix.

## Revisit when

The next new study script is written on `build_base_study_parser`, or the broad
study is superseded by `optuna_study_v2.py` — then make the parser default real
(`default=default_storage`) and re-verify the affected studies.
