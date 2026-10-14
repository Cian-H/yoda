# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Calendar Versioning](https://calver.org/) (`YYYY.MM.MICRO`).

## [Unreleased]

### Added
- Local SQLite backup (`data/optuna_v2.db`) and credential-free JSON export
  (`data/optuna_v2_backup.json`) preserving all 7 Optuna studies and 672 trials
  from the hosted Neon database.
- Deterministic dummy text encoder (stable crc32 token hashing) for reproducible tests.
- Bilinear candidate-query alignment in the decision head: centered truth-order affinity,
  Neither-at-zero evidence, information-mass knowledge gating, and a `logit_scale`
  ranking temperature (ADR-0019).
- Group-relative candidate scoring in the engine's independent path: candidates are
  scored on their deviation from the `(query, state, task)` group mean.
- Dual-channel negation-equivariant `BelnapFFN` with tied output projection and
  normalization (ADR-0020).
- Context encoding deduplication and cross-call memoization with a `use_context_cache`
  flag; caches are cleared automatically by any gradient-enabled forward (ADR-0021).
- Requirements-driven Tier-4 e2e acceptance suite (`tests/e2e/`) covering R1-R5.
- `--decisiveness-weight` and `--decisiveness-warmup-epochs` CLI flags in
  `experiments/train_yoda.py`; `eval_accuracy` metric alias in `YodaTrainer.evaluate`.
- Skip telemetry: the Parquet ETL now counts and warns on records dropped for
  unmatched targets or targets beyond `max_choices`.
- Nimble ingestion preserves base/counterfactual pairing (`pair_id`, `variant`) in
  payload metadata.
- `experiments/prepare_hpo_split.py`: v2 data protocol — group-disjoint,
  source-stratified `train`/`optuna_eval`/`test` splits with reproducible
  hash-based assignment, exact-row dedup, and split metadata (per-source
  const-predictor rates, derived collapse threshold).
- `experiments/optuna_study_v2.py`: phased post-hardening study
  (`sanity`/`broad`/`focused`/`validate`) implementing the council-verdict
  conditions; local sqlite storage by default, `YODA_OPTUNA_STORAGE` override
  for distributed runs.
- `experiments/analyze_study.py`: study analysis — top trials, fANOVA parameter
  importances, Phase-B lock and top-config exports.
- `--logit-scale` CLI flag in `experiments/train_yoda.py`;
  `YodaDecisionEngine` accepts `logit_scale` and forwards it to both decision
  heads (default 8.0).
- `--eval-path`/`--test-path` flags in the shared Optuna study parser.
- Macro-averaged per-source eval accuracy (`macro_accuracy`) in
  `YodaTrainer.evaluate` and per-epoch history; `source` plumbed through both
  datasets and `collate_decision_batch`.
- Metric aggregation (`final`/`best`/`mean_last2`) in `run_training_trial` with
  final/best metric user attrs; `aggregate_metric_values`/`extract_metric_value`
  helpers.
- `scikit-learn` dependency (fANOVA importance analysis).
### Changed
- Switched project versioning to CalVer (`YYYY.MM.MICRO`).
- Collapse pruning compares the macro-averaged per-source accuracy against the
  threshold, so per-source positional bias can no longer mask collapse.
- Optuna search space realigned to the hardened architecture: `conflation_weight`
  [0.01, 0.25] log, `residual_weight` [0.10, 0.25], `lr` [1e-5, 5e-4],
  `decisiveness_weight` [0.001, 0.05] with 1-3 warmup epochs.
- `YodaTrainer` auto-dispatches grouped candidate batches for `num_choices == 1`
  engines; `fit()` accepts a `val_loader` alias.
- `YodaTrainer.fit` residual weight default 0.5 -> 0.15 in `experiments/train_yoda.py`,
  matching the hardened architecture.
- `.agents/teamwork/` excluded from ruff (archival agentic session artifacts).
### Deprecated
### Removed
- Purged legacy pre-hardening databases and metadata (`data/optuna.db`,
  `data/optuna_narrow.db`, `data/yoda_sweep_1791233751_metadata.json`,
  `data/yoda_narrow_sweep_1791462045_metadata.json`).
- Removed application-centric `SECURITY.md` and `.docs/security/` methodology.
- Removed satisfied `.docs/todos/` entries from the architectural hardening roadmap.
- Retired `experiments/optuna_study_focused.py` and
  `experiments/optuna_study_narrow.py` to `experiments/retired/` (stale
  pre-M4 search spaces; superseded by `optuna_study_v2.py`).
### Fixed
- Attention mask propagation and sigmoid attention scale normalization by active
  sequence mass (R1); paraconsistent contradiction loss with linear warmup replacing
  the decisiveness Jacobian penalty (R2); NeSy XOR/OR positive-support reformulation,
  coupled cross-polarity routing, and epistemic logit gating (R3).
- Nimble dataset contributed zero training rows: gold targets are read from
  `reference.target` instead of a nonexistent `teacher.answers` block, and integer
  gold targets are matched as ordinals into the constraint list — restoring 6,335
  dropped records across the nimble, kev, and n4ze3m_synth sources.
- `init_study_and_metadata` redacts the password component of the storage DSN in
  study metadata JSON and log output, so credentialed Postgres URLs can no longer
  leak into committable metadata files or logs.
- `optuna_study_v2.py` fails fast at startup when the v2 split files are missing
  (validate phase likewise requires the locked test split) instead of pruning
  every trial against non-existent paths.
- `build_base_study_parser` escapes literal `%` in the storage help string
  (argparse %-formatting crashed on storage URLs with percent-encoded
  characters), and the v2 study passes a static, credential-free
  default-storage help string.
- `experiments/analyze_study.py` reports the available study names instead of a
  raw `KeyError` when `--study-name` is not found in the storage.
- The v2 study fail-fast message and `prepare_hpo_split.py`'s missing-input error
  now name the gitignored-data transfer step (transfer `data/processed/v2/` or
  copy the processed parquet inputs) instead of only "run the ETL first".
### Security

## [2026.10.0] — 2026-10-01

### Added
- Initial project bootstrap. See `AGENTS.md` and `.agents/rules/` for conventions.
