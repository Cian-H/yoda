# ADR 0016: Reusable Hyperparameter Optimization Module (yoda.hyperopt)

## Status
Accepted

## Date
2026-10-08

## Context
Hyperparameter optimization for Yoda requires evaluating configurations under varied computational constraints and topological setups (e.g. initial broad search vs. narrowed champion refinement sweeps). Previously:
1. **Subprocess Boilerplate Duplication**: Trial execution, CUDA OOM isolation via subprocesses, return-code verification, trial pruning, and `history.json` metric extraction were duplicated verbatim across experiment scripts (`optuna_study.py` and `optuna_study_narrow.py`).
2. **Missing Study Early Stopping Abstraction**: When a sweep plateaued early (such as the initial 275-trial run plateauing at Trial #21), Optuna lacked an integrated early-stopping mechanism without manual intervention.
3. **Inconsistent CLI Configurations**: Command-line arguments for study sizing, storage locations, patience budgets, and sample caps were redefined independently in each experiment entry point.
4. **Namespace Cleanliness**: To avoid colliding with the popular third-party PyPI library `hyperopt`, hyperparameter optimization utilities must reside firmly within the `yoda.*` first-party package namespace.

## Decision
1. **Created `src/yoda/hyperopt` Submodule**:
   - `StudyEarlyStoppingCallback`: Callback monitoring trial completion to stop `study.optimize()` after $N$ consecutive trials without validation gain.
   - `build_training_command`: Maps typed hyperparameter dictionaries and boolean flags cleanly to `train_yoda.py` CLI arguments.
   - `run_training_trial`: Executes training within an isolated subprocess to protect PyTorch from CUDA memory fragmentation, raises `optuna.TrialPruned` on failure or missing telemetry, and extracts the target metric from `history.json`.
   - `init_study_and_metadata`: Standardizes study creation with a multivariate `TPESampler`, cryptographic non-42 seed selection, SQLite storage initialization, and metadata JSON serialization.
   - `build_base_study_parser`: Provides a reusable argparse parser populated with standard hyperparameter sweep flags and customizable defaults.
2. **Refactored Experiment Scripts**:
   - Updated both `experiments/optuna_study.py` and `experiments/optuna_study_narrow.py` to import from `yoda.hyperopt.optuna`.
   - Experiment scripts now focus exclusively on declaring parameter search spaces and invoking `run_training_trial`.

## Consequences
- **Positive**: Eliminates ~100 lines of duplicated execution code, standardizes trial error handling and metrics extraction, and enables automated study-level early stopping.
- **Positive**: First-party `yoda.hyperopt` namespace keeps the architecture modular and avoids third-party library naming collisions.
- **Testing**: Dedicated unit tests in `tests/test_hyperopt.py` verify command generation, subprocess execution isolation, study callback triggers, and metadata generation.
