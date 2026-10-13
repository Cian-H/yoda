"""Focused hyperparameter optimization study for YodaDecisionEngine using Optuna.

Isolates and scans epistemic query probe capacity (`num_q_probes`) and contradiction
pruning weight (`conflation_weight`) while locking in all empirical champion parameters:
- Locks text backbone to 'BAAI/bge-large-en-v1.5'.
- Locks num_reasoning_blocks to 1, n_heads to 2.
- Locks lr to 3.0e-4, backbone_lr to 6.0e-6, lr_decay to 0.60.
- Locks num_c_probes to 14, belnap_weight to 1.25.
- Locks d_hidden_multiplier to 1.4 (Pareto optimal).
- Scans num_q_probes in [7, 12] (expanding past upper bound 7).
- Scans conflation_weight in [0.48, 0.65] (expanding past upper bound 0.50).
- Enqueues Champion Trial #40 baseline point (num_q_probes=7, conflation_weight=0.49).
- Defaults to 8 epochs and 16,000 samples to allow complete cosine scheduler annealing.
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

# Optimal fixed values identified from broad (#21) and narrow (#40) sweeps
FIXED_TEXT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
FIXED_NUM_REASONING_BLOCKS = 1
FIXED_N_HEADS = 2
FIXED_T0_EPOCHS = 3
FIXED_T_MULT = 2
FIXED_RESIDUAL_WEIGHT = 0.6356
FIXED_DROPOUT = 0.2857
FIXED_MARGIN_WEIGHT = 0.6180

# Newly locked parameters from narrow champion Trial #40
FIXED_LR = 3.0e-4
FIXED_BACKBONE_LR = 6.0e-6
FIXED_LR_DECAY = 0.60
FIXED_NUM_C_PROBES = 14
FIXED_BELNAP_WEIGHT = 1.25
FIXED_D_HIDDEN_MULTIPLIER = 1.4

# Champion Trial #40 baseline point to enqueue as initial evaluation point
CHAMPION_TRIAL_40_PARAMS = {
    "num_q_probes": 7,
    "conflation_weight": 0.49,
}


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective evaluating query probe capacity and contradiction pruning.

    Args:
        trial: An Optuna trial instance.
        args: Command-line arguments.

    Returns:
        Final epoch evaluation accuracy (eval_accuracy) to maximize.
    """
    params: dict[str, Any] = {
        # Locked backbone, topological, and training parameters
        "text_model_name": FIXED_TEXT_MODEL_NAME,
        "num_reasoning_blocks": FIXED_NUM_REASONING_BLOCKS,
        "n_heads": FIXED_N_HEADS,
        "dropout": FIXED_DROPOUT,
        "residual_weight": FIXED_RESIDUAL_WEIGHT,
        "margin_weight": FIXED_MARGIN_WEIGHT,
        "t0_epochs": FIXED_T0_EPOCHS,
        "t_mult": FIXED_T_MULT,
        "lr": FIXED_LR,
        "backbone_lr": FIXED_BACKBONE_LR,
        "lr_decay": FIXED_LR_DECAY,
        "num_c_probes": FIXED_NUM_C_PROBES,
        "belnap_weight": FIXED_BELNAP_WEIGHT,
        "d_hidden_multiplier": FIXED_D_HIDDEN_MULTIPLIER,
        # Active scanned parameters
        "num_q_probes": trial.suggest_int("num_q_probes", 7, 12),
        "conflation_weight": trial.suggest_float("conflation_weight", 0.48, 0.65),
        # Runtime settings
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "collapse_threshold": args.collapse_threshold,
        "collapse_patience_epochs": args.collapse_patience_epochs,
    }

    if args.train_path:
        params["train_path"] = args.train_path

    trial_dir = Path("models/optuna_focused") / f"trial_{trial.number}"
    return run_training_trial(trial, trial_dir, params)


def main() -> None:
    """Creates the focused Optuna study and executes optimization."""
    parser = build_base_study_parser(
        description="Run focused Optuna study scanning num_q_probes and conflation_weight.",
        default_n_trials=50,
        default_max_samples=16000,
        default_epochs=8,
        default_batch_size=8,
        default_early_stopping_patience=2,
        default_study_patience=20,
        default_storage="sqlite:///data/optuna_focused.db",
    )
    args = parser.parse_args()

    study_name = args.study_name or f"yoda_focused_sweep_{int(time.time())}"
    callbacks = []
    if args.study_patience and args.study_patience > 0:
        callbacks.append(StudyEarlyStoppingCallback(patience=args.study_patience))

    study, _ = init_study_and_metadata(
        study_name=study_name,
        storage=args.storage,
        seed=args.seed,
        heartbeat_interval=args.heartbeat_interval if args.heartbeat_interval > 0 else None,
        grace_period=args.grace_period,
        extra_metadata={
            "n_trials": args.n_trials,
            "study_patience": args.study_patience,
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "heartbeat_interval": args.heartbeat_interval,
            "grace_period": args.grace_period,
            "fixed_parameters": {
                "text_model_name": FIXED_TEXT_MODEL_NAME,
                "num_reasoning_blocks": FIXED_NUM_REASONING_BLOCKS,
                "n_heads": FIXED_N_HEADS,
                "t0_epochs": FIXED_T0_EPOCHS,
                "t_mult": FIXED_T_MULT,
                "residual_weight": FIXED_RESIDUAL_WEIGHT,
                "dropout": FIXED_DROPOUT,
                "margin_weight": FIXED_MARGIN_WEIGHT,
                "lr": FIXED_LR,
                "backbone_lr": FIXED_BACKBONE_LR,
                "lr_decay": FIXED_LR_DECAY,
                "num_c_probes": FIXED_NUM_C_PROBES,
                "belnap_weight": FIXED_BELNAP_WEIGHT,
                "d_hidden_multiplier": FIXED_D_HIDDEN_MULTIPLIER,
            },
        },
    )

    # Enqueue champion Trial #40 baseline point idempotently for distributed workers
    champ_q = CHAMPION_TRIAL_40_PARAMS["num_q_probes"]
    champ_conf = CHAMPION_TRIAL_40_PARAMS["conflation_weight"]
    already_enqueued = any(
        t.params.get("num_q_probes") == champ_q
        and abs(t.params.get("conflation_weight", 0.0) - champ_conf) < 1e-4
        for t in study.get_trials(deepcopy=False)
    )
    if not already_enqueued:
        study.enqueue_trial(CHAMPION_TRIAL_40_PARAMS)

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
