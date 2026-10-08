"""Hyperparameter optimization study for YodaDecisionEngine using Optuna."""

import argparse
import time
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from yoda.hyperopt.optuna import (
    build_base_study_parser,
    init_study_and_metadata,
    run_training_trial,
)


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function evaluating YodaDecisionEngine configurations.

    Args:
        trial: An Optuna trial instance.
        args: Command-line arguments.

    Returns:
        Final epoch evaluation accuracy (eval_accuracy) to maximize.
    """
    params: dict[str, Any] = {
        # 1. Text model backbone selection
        "text_model_name": trial.suggest_categorical(
            "text_model_name",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-large-en-v1.5",
                "intfloat/e5-large-v2",
                "BAAI/bge-base-en-v1.5",
            ],
        ),
        # 2. Topological parameters
        "num_reasoning_blocks": trial.suggest_int("num_reasoning_blocks", 1, 3),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "d_hidden_multiplier": trial.suggest_float("d_hidden_multiplier", 1.0, 4.0),
        "num_q_probes": trial.suggest_int("num_q_probes", 2, 8),
        "num_c_probes": trial.suggest_int("num_c_probes", 8, 32),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "conflation_weight": trial.suggest_float("conflation_weight", 0.0, 0.5),
        "residual_weight": trial.suggest_float("residual_weight", 0.1, 0.9),
        # 3. Training parameters
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-6, 1e-4, log=True),
        "t0_epochs": trial.suggest_int("t0_epochs", 1, 3),
        "t_mult": trial.suggest_int("t_mult", 1, 2),
        "lr_decay": trial.suggest_float("lr_decay", 0.5, 1.0),
        # 4. Loss Weights
        "belnap_weight": trial.suggest_float("belnap_weight", 0.5, 2.0),
        "margin_weight": trial.suggest_float("margin_weight", 0.1, 1.0),
        # 5. Fixed study runtime parameters
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stopping_patience,
    }

    if args.train_path:
        params["train_path"] = args.train_path

    trial_dir = Path("models/optuna") / f"trial_{trial.number}"
    return run_training_trial(trial, trial_dir, params)


def main() -> None:
    """Creates the Optuna study and runs the optimization sweep."""
    parser = build_base_study_parser(
        description="Run Optuna study for YodaDecisionEngine.",
        default_n_trials=2500,
        default_max_samples=8000,
        default_epochs=4,
        default_batch_size=8,
        default_early_stopping_patience=1,
        default_study_patience=0,
        default_storage="sqlite:///data/optuna.db",
    )
    args = parser.parse_args()

    study_name = args.study_name or f"yoda_sweep_{int(time.time())}"

    study, _ = init_study_and_metadata(
        study_name=study_name,
        storage=args.storage,
        seed=args.seed,
        extra_metadata={
            "n_trials": args.n_trials,
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
        },
    )

    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials, n_jobs=args.n_jobs)

    try:
        logger.info("Optuna study completed.")
        logger.info("Best trial: #{}", study.best_trial.number)
        logger.info("Best eval_accuracy: {:.4f}", study.best_value)
        logger.info("Best parameters: {}", study.best_params)
    except ValueError:
        logger.warning("No trials completed successfully.")


if __name__ == "__main__":
    main()
