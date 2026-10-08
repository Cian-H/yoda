"""Focused hyperparameter optimization study for YodaDecisionEngine using Optuna.

Narrows the search space based on empirical results from the initial sweep (yoda_sweep_1791233751):
- Fixes low-importance parameters (<5% importance) to their optimal values.
- Locks text backbone to 'BAAI/bge-large-en-v1.5'.
- Locks num_reasoning_blocks to 1.
- Narrows the search bounds for parameters with high empirical leverage.
- Increases training capacity: 16,000 max samples, 6 epochs, patience 2.
- Adds study-level early stopping to terminate automatically when no improvement is found.
"""

import argparse
import json
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

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


class StudyEarlyStoppingCallback:
    """Callback to terminate an Optuna study after N trials without improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_trial_number: int | None = None
        self.best_value: float | None = None

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if self.best_value is None or (trial.value is not None and trial.value > self.best_value):
            self.best_value = trial.value
            self.best_trial_number = trial.number
        elif self.best_trial_number is not None and (
            trial.number - self.best_trial_number >= self.patience
        ):
            logger.info(
                "Study early stopping triggered: No improvement for {} trials "
                "(best trial #{}, value={:.4f})",
                self.patience,
                self.best_trial_number,
                self.best_value if self.best_value is not None else 0.0,
            )
            study.stop()


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective function evaluating narrowed YodaDecisionEngine configurations.

    Args:
        trial: An Optuna trial instance.
        args: Command-line arguments.

    Returns:
        Final epoch evaluation accuracy (eval_accuracy) to maximize.
    """
    # 1. Narrowed High-Importance Parameters
    lr = trial.suggest_float("lr", 1.0e-4, 3.5e-4, log=True)
    backbone_lr = trial.suggest_float("backbone_lr", 5.0e-6, 4.5e-5, log=True)
    lr_decay = trial.suggest_float("lr_decay", 0.55, 0.75)

    num_c_probes = trial.suggest_int("num_c_probes", 12, 20)
    num_q_probes = trial.suggest_int("num_q_probes", 4, 7)
    d_hidden_multiplier = trial.suggest_float("d_hidden_multiplier", 1.4, 3.9)

    belnap_weight = trial.suggest_float("belnap_weight", 0.75, 1.45)
    conflation_weight = trial.suggest_float("conflation_weight", 0.35, 0.50)

    # 2. Generate trial checkpoint directory
    trial_dir = Path("models/optuna_narrow") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/train_yoda.py",
        # Locked backbone and topological blocks
        "--text-model-name",
        FIXED_TEXT_MODEL_NAME,
        "--num-reasoning-blocks",
        str(FIXED_NUM_REASONING_BLOCKS),
        "--n-heads",
        str(FIXED_N_HEADS),
        # Sampled topological parameters
        "--d-hidden-multiplier",
        str(d_hidden_multiplier),
        "--num-q-probes",
        str(num_q_probes),
        "--num-c-probes",
        str(num_c_probes),
        # Fixed regularizers and weights (<5% importance)
        "--dropout",
        str(FIXED_DROPOUT),
        "--residual-weight",
        str(FIXED_RESIDUAL_WEIGHT),
        "--margin-weight",
        str(FIXED_MARGIN_WEIGHT),
        "--t0-epochs",
        str(FIXED_T0_EPOCHS),
        "--t-mult",
        str(FIXED_T_MULT),
        # Sampled learning and loss dynamics
        "--conflation-weight",
        str(conflation_weight),
        "--lr",
        str(lr),
        "--backbone-lr",
        str(backbone_lr),
        "--lr-decay",
        str(lr_decay),
        "--belnap-weight",
        str(belnap_weight),
    ]

    if args.train_path:
        cmd.extend(["--train-path", args.train_path])

    # Expanded training settings: 16k samples, 6 epochs, early-stopping-patience 2
    cmd.extend(
        [
            "--checkpoint-dir",
            str(trial_dir),
            "--output-model",
            str(trial_dir / "model.pt"),
            "--max-samples",
            str(args.max_samples),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--precision",
            "bfloat16",
            "--early-stopping-patience",
            str(args.early_stopping_patience),
        ]
    )

    logger.info(
        "Starting Trial #{} [lr={:.2e}, bb_lr={:.2e}, decay={:.2f}, "
        "c_probes={}, q_probes={}, d_mult={:.2f}, belnap={:.2f}, conflation={:.2f}]",
        trial.number,
        lr,
        backbone_lr,
        lr_decay,
        num_c_probes,
        num_q_probes,
        d_hidden_multiplier,
        belnap_weight,
        conflation_weight,
    )

    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            logger.warning(
                "Trial #{} failed: subprocess returned exit code {}",
                trial.number,
                proc.returncode,
            )
            raise optuna.TrialPruned(f"Subprocess failed with exit code {proc.returncode}")
    except Exception as exc:
        if isinstance(exc, optuna.TrialPruned):
            raise
        logger.error("Exception during trial #{} execution: {}", trial.number, exc)
        raise optuna.TrialPruned(str(exc)) from exc

    history_file = trial_dir / "history.json"
    if not history_file.exists():
        logger.warning("history.json not found in {}", trial_dir)
        raise optuna.TrialPruned(f"history.json not found in {trial_dir}")

    try:
        with history_file.open("r", encoding="utf-8") as f:
            history = json.load(f)

        if not history or not isinstance(history, list):
            logger.warning("history.json in {} is empty or not a list", trial_dir)
            return 0.0

        final_epoch = history[-1]
        eval_accuracy = final_epoch.get("eval_accuracy")
        if eval_accuracy is None:
            logger.warning("Final epoch record missing 'eval_accuracy'")
            return 0.0

        logger.info(
            "Trial #{} finished with final eval_accuracy: {:.4f}",
            trial.number,
            float(eval_accuracy),
        )
        return float(eval_accuracy)
    except Exception as exc:
        logger.error("Error reading history.json for trial #{}: {}", trial.number, exc)
        return 0.0


def main() -> None:
    """Creates the narrowed Optuna study and executes optimization."""
    parser = argparse.ArgumentParser(
        description="Run narrowed Optuna study for YodaDecisionEngine."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=150,
        help="Number of Optuna optimization trials (default: 150)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of trials to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16000,
        help="Maximum training samples from dataset (default: 16000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs per trial (default: 6)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Epochs without validation improvement before early stopping (default: 2)",
    )
    parser.add_argument(
        "--study-patience",
        type=int,
        default=35,
        help="Number of trials without improvement before stopping the study (default: 35)",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to the primary dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Optuna sampler (default: randomly generated)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study (default: auto-generated timestamped name)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database storage URL (default: sqlite:///data/optuna_narrow.db)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = secrets.randbits(32)
        while seed == 42:
            seed = secrets.randbits(32)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

    Path("data").mkdir(exist_ok=True)
    Path("models/optuna_narrow").mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///data/optuna_narrow.db"
    if args.storage:
        storage = args.storage
    study_name = f"yoda_narrow_sweep_{int(time.time())}"
    if args.study_name:
        study_name = args.study_name

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    # Enqueue champion Trial #21 parameters as baseline seed trial
    study.enqueue_trial(CHAMPION_TRIAL_21_PARAMS)

    callbacks: list[Any] = []
    if args.study_patience and args.study_patience > 0:
        callbacks.append(StudyEarlyStoppingCallback(patience=args.study_patience))

    metadata = {
        "study_name": study_name,
        "seed": seed,
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
        "storage": storage,
        "direction": "maximize",
        "sampler": type(sampler).__name__,
        "created_at": time.time(),
    }
    metadata_path = Path("data") / f"{study_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Initialized study '{}' with seed={} and storage={}", study_name, seed, storage)
    logger.info("Enqueued Trial #21 champion baseline: {}", CHAMPION_TRIAL_21_PARAMS)
    logger.info("Wrote study metadata to {}", metadata_path)

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
