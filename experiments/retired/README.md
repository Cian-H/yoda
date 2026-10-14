# Retired Optuna studies

`optuna_study_focused.py` and `optuna_study_narrow.py` are retired (2026-10-14,
council verdict). Both encode pre-M4-hardening search spaces that are no longer
valid:

- `optuna_study_focused.py` locks `residual_weight=0.6356` and scans
  `conflation_weight` in [0.48, 0.65] — the regime the M4 realignment excluded.
- `optuna_study_narrow.py` enqueues a champion with `conflation_weight=0.4256`,
  outside its own realigned [0.01, 0.25] range (Optuna >= 5 only warns and
  evaluates the out-of-range value anyway), and omits `decisiveness_weight`, so
  its trials silently use the CLI default 0.1 (2x the realigned cap of 0.05).

They are kept here (not deleted) for provenance and to keep the old study
databases (`data/optuna_narrow.db`, the hosted `yoda_focused` study)
interpretable. Do not run them. Use `experiments/optuna_study_v2.py` instead.
