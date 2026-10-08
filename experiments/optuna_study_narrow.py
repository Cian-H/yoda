"""Focused hyperparameter optimization study for YodaDecisionEngine using Optuna.

Narrows the search space based on empirical results from the initial sweep (yoda_sweep_1791233751):
- Fixes low-importance parameters (<5% importance) to their optimal values.
- Locks text backbone to 'BAAI/bge-large-en-v1.5'.
- Locks num_reasoning_blocks to 1.
- Narrows search bounds for parameters with high empirical leverage.
- Increases training capacity: 16,000 max samples, 6 epochs, patience 2.
- Adds study-level early stopping to terminate automatically when no improvement is found.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from yoda.hyperopt.optuna import (
    StudyEarlyStoppingCallback,
    build_base_study_parser,
    init_study_and_metadata,
    run_training_trial,
)

# Optimal fixed values identified from champion Trial #21 (<5% parameter importance)
FIXED_TEXT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
FIXED_NUM_REASONING_BLOCKS = 1
FIXED_N_HEADS = 2
FIXED_T0_EPOCHS = 3
FIXED_T_MULT = 2
FIXED_RESIDUAL_WEIGHT = 0.6356
FIXED_DROPOUT = 0.2857
FIXED_MARGIN_WEIGHT = 0.6180

# Champion Trial #21 baseline parameters to enqueue as initial evaluation point
CHAMPION_TRIAL_21_PARAMS = {
    "lr": 0.00027843,
    "backbone_lr": 8.919e-06,
    "lr_decay": 0.5825,
    "num_c_probes": 13,
    "num_q_probes": 6,
    "d_hidden_multiplier": 1.9025,
    "belnap_weight": 1.3556,
    "conflation_weight": 0.4256,
}


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function evaluating narrowed YodaDecisionEngine configurations.

    Args:
        trial: An Optuna trial instance.
        args: Command-line arguments.

    Returns:
        Final epoch evaluation accuracy (eval_accuracy) to maximize.
    """
    params: dict[str, Any] = {
        # Locked backbone and topological blocks
        "text_model_name": FIXED_TEXT_MODEL_NAME,
        "num_reasoning_blocks": FIXED_NUM_REASONING_BLOCKS,
        "n_heads": FIXED_N_HEADS,
        # Fixed regularizers and weights (<5% importance)
        "dropout": FIXED_DROPOUT,
        "residual_weight": FIXED_RESIDUAL_WEIGHT,
        "margin_weight": FIXED_MARGIN_WEIGHT,
        "t0_epochs": FIXED_T0_EPOCHS,
        "t_mult": FIXED_T_MULT,
        # Sampled narrowed parameters
        "lr": trial.suggest_float("lr", 1.0e-4, 3.5e-4, log=True),
        "backbone_lr": trial.suggest_float("backbone_lr", 5.0e-6, 4.5e-5, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.55, 0.75),
        "num_c_probes": trial.suggest_int("num_c_probes", 12, 20),
        "num_q_probes": trial.suggest_int("num_q_probes", 4, 7),
        "d_hidden_multiplier": trial.suggest_float("d_hidden_multiplier", 1.4, 3.9),
        "belnap_weight": trial.suggest_float("belnap_weight", 0.75, 1.45),
        "conflation_weight": trial.suggest_float("conflation_weight", 0.35, 0.50),
        # Runtime settings
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stopping_patience,
    }

    if args.train_path:
        params["train_path"] = args.train_path

    trial_dir = Path("models/optuna_narrow") / f"trial_{trial.number}"
    return run_training_trial(trial, trial_dir, params)


def main() -> None:
    """Creates the narrowed Optuna study and executes optimization."""
    parser = build_base_study_parser(
        description="Run narrowed Optuna study for YodaDecisionEngine.",
        default_n_trials=150,
        default_max_samples=16000,
        default_epochs=6,
        default_batch_size=8,
        default_early_stopping_patience=2,
        default_study_patience=35,
        default_storage="sqlite:///data/optuna_narrow.db",
    )
    args = parser.parse_args()

    study_name = args.study_name or f"yoda_narrow_sweep_{int(time.time())}"
    callbacks = []
    if args.study_patience and args.study_patience > 0:
        callbacks.append(StudyEarlyStoppingCallback(patience=args.study_patience))

    study, _ = init_study_and_metadata(
        study_name=study_name,
        storage=args.storage,
        seed=args.seed,
        extra_metadata={
            "n_trials": args.n_trials,
            "study_patience": args.study_patience,
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "fixed_parameters": {
                "text_model_name": FIXED_TEXT_MODEL_NAME,
                "num_reasoning_blocks": FIXED_NUM_REASONING_BLOCKS,
                "n_heads": FIXED_N_HEADS,
                "t0_epochs": FIXED_T0_EPOCHS,
                "t_mult": FIXED_T_MULT,
                "residual_weight": FIXED_RESIDUAL_WEIGHT,
                "dropout": FIXED_DROPOUT,
                "margin_weight": FIXED_MARGIN_WEIGHT,
            },
        },
    )

    # Enqueue champion Trial #21 parameters as baseline seed trial
    study.enqueue_trial(CHAMPION_TRIAL_21_PARAMS)

    study.optimize(
        lambda t: objective(t, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=callbacks,
    )

    try:
        logger.info("Optuna study completed.")
        logger.info("Best trial: #{}", study.best_trial.number)
        logger.info("Best eval_accuracy: {:.4f}", study.best_value)
        logger.info("Best parameters: {}", study.best_params)
    except ValueError:
        logger.warning("No trials completed successfully.")


if __name__ == "__main__":
    main()
