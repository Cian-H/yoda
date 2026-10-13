"""Optuna study management, trial runner, and early stopping harness for Yoda.

Provides reusable abstractions for hyperparameter optimization sweeps:
- StudyEarlyStoppingCallback: patience-based study termination.
- run_training_trial: isolated subprocess execution, error handling, pruning, and metric
  extraction.
- init_study_and_metadata: reproducible study creation with TPESampler, storage setup,
  and metadata serialization.
- build_base_study_parser: standardized CLI argument parser for Optuna sweeps.
"""

import argparse
import json
import secrets
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import optuna
from loguru import logger
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback, fail_stale_trials


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


def build_training_command(
    trial_dir: Path,
    training_args: dict[str, Any],
    script_path: str = "experiments/train_yoda.py",
) -> list[str]:
    """Builds a CLI command line list for running train_yoda.py in a subprocess.

    Args:
        trial_dir: Directory where trial artifacts and history.json are stored.
        training_args: Dictionary of parameter names and values. Keys can use underscores
            or hyphens (e.g. 'num_reasoning_blocks' becomes '--num-reasoning-blocks').
        script_path: Path to the training entrypoint script.

    Returns:
        List of command-line arguments suitable for subprocess.run.
    """
    cmd = ["uv", "run", "python", script_path]

    # Flags that may be passed explicitly or defaulted
    default_flags = {
        "checkpoint-dir": str(trial_dir),
        "output-model": str(trial_dir / "model.pt"),
        "precision": "bfloat16",
    }

    merged_args: dict[str, Any] = {**default_flags}
    for k, v in training_args.items():
        normalized_key = k.replace("_", "-")
        merged_args[normalized_key] = v

    for k, v in merged_args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            else:
                cmd.append(f"--no-{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    return cmd


def run_training_trial(
    trial: optuna.Trial,
    trial_dir: Path,
    training_args: dict[str, Any],
    metric_name: str = "eval_accuracy",
    script_path: str = "experiments/train_yoda.py",
) -> float:
    """Executes a single training trial in an isolated subprocess and extracts target metric.

    Isolating execution in a subprocess prevents PyTorch CUDA memory fragmentation and leaks
    across Optuna iterations.

    Args:
        trial: Optuna trial instance.
        trial_dir: Directory to save checkpoint, telemetry, and history.json.
        training_args: Hyperparameters and flags passed to the training script.
        metric_name: Metric key in history.json to extract and maximize.
        script_path: Training script entry point.

    Returns:
        Extracted metric value for the final training epoch.

    Raises:
        optuna.TrialPruned: If the subprocess exits with non-zero status or history.json is missing.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_training_command(trial_dir, training_args, script_path=script_path)

    logger.info("Starting Trial #{} with {} parameters", trial.number, len(training_args))

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
        metric_value = final_epoch.get(metric_name)
        if metric_value is None:
            logger.warning("Final epoch record missing '{}'", metric_name)
            return 0.0

        logger.info(
            "Trial #{} finished with final {}: {:.4f}",
            trial.number,
            metric_name,
            float(metric_value),
        )
        return float(metric_value)
    except Exception as exc:
        logger.error("Error reading history.json for trial #{}: {}", trial.number, exc)
        return 0.0


def init_study_and_metadata(
    study_name: str | None = None,
    storage: str | None = None,
    direction: str = "maximize",
    seed: int | None = None,
    metadata_dir: Path | str = "data",
    extra_metadata: dict[str, Any] | None = None,
    load_if_exists: bool = True,
    heartbeat_interval: int | None = 30,
    grace_period: int | None = None,
) -> tuple[optuna.Study, dict[str, Any]]:
    """Initializes an Optuna study with multivariate TPESampler and records metadata JSON.

    Args:
        study_name: Unique study name. If None, auto-generated with timestamp.
        storage: Storage URL. If None, defaults to 'sqlite:///data/optuna.db'.
        direction: Optimization direction ('maximize' or 'minimize').
        seed: Random seed for sampler. If None, cryptographically generated (excluding 42).
        metadata_dir: Directory where metadata JSON is persisted.
        extra_metadata: Additional configuration or study parameters to record.
        load_if_exists: Whether to reload an existing study with the same name.
        heartbeat_interval: Heartbeat interval in seconds for distributed execution.
        grace_period: Grace period in seconds before failing stale trials (default: 2 * interval).

    Returns:
        Tuple of (optuna.Study, metadata_dict).
    """
    if seed is not None:
        resolved_seed = seed
    else:
        resolved_seed = secrets.randbits(32)
        while resolved_seed == 42:
            resolved_seed = secrets.randbits(32)

    sampler = optuna.samplers.TPESampler(seed=resolved_seed, multivariate=True)

    target_metadata_dir = Path(metadata_dir)
    target_metadata_dir.mkdir(parents=True, exist_ok=True)

    resolved_storage_url = storage if storage else "sqlite:///data/optuna.db"
    resolved_study_name = study_name if study_name else f"yoda_sweep_{int(time.time())}"

    # Setup RDBStorage with heartbeat monitoring for distributed workers
    if heartbeat_interval and heartbeat_interval > 0 and ":memory:" not in resolved_storage_url:
        connect_args = {"timeout": 60.0} if resolved_storage_url.startswith("sqlite") else {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            study_storage = RDBStorage(
                url=resolved_storage_url,
                heartbeat_interval=heartbeat_interval,
                grace_period=grace_period,
                heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=3),
                engine_kwargs={"connect_args": connect_args} if connect_args else None,
            )
    else:
        study_storage = resolved_storage_url

    study = optuna.create_study(
        study_name=resolved_study_name,
        storage=study_storage,
        sampler=sampler,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    # Clean up any stale trials from earlier aborted distributed workers
    if isinstance(study_storage, RDBStorage):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            fail_stale_trials(study)

    metadata: dict[str, Any] = {
        "study_name": resolved_study_name,
        "seed": resolved_seed,
        "storage": resolved_storage_url,
        "direction": direction,
        "sampler": type(sampler).__name__,
        "heartbeat_interval": heartbeat_interval,
        "grace_period": grace_period,
        "created_at": time.time(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = target_metadata_dir / f"{resolved_study_name}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Initialized study '{}' with seed={} and storage={}",
        resolved_study_name,
        resolved_seed,
        resolved_storage_url,
    )
    logger.info("Wrote study metadata to {}", metadata_path)

    return study, metadata


def build_base_study_parser(
    description: str,
    default_n_trials: int = 150,
    default_max_samples: int = 16000,
    default_epochs: int = 6,
    default_batch_size: int = 8,
    default_early_stopping_patience: int = 2,
    default_study_patience: int = 35,
    default_storage: str = "sqlite:///data/optuna.db",
) -> argparse.ArgumentParser:
    """Creates a base argument parser populated with standard hyperparameter study flags.

    Args:
        description: Description text for the argument parser.
        default_n_trials: Default trial count.
        default_max_samples: Default dataset sample cap.
        default_epochs: Default epoch count per trial.
        default_batch_size: Default training batch size.
        default_early_stopping_patience: Default per-trial early stopping patience.
        default_study_patience: Default study-level early stopping patience.
        default_storage: Default database storage URL.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=default_n_trials,
        help=f"Number of Optuna optimization trials (default: {default_n_trials})",
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
        default=default_max_samples,
        help=f"Maximum training samples from dataset (default: {default_max_samples})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_epochs,
        help=f"Number of training epochs per trial (default: {default_epochs})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help=f"Training batch size (default: {default_batch_size})",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=default_early_stopping_patience,
        help="Epochs without validation improvement before trial early stopping "
        f"(default: {default_early_stopping_patience})",
    )
    parser.add_argument(
        "--study-patience",
        type=int,
        default=default_study_patience,
        help="Number of trials without improvement before stopping the study "
        f"(default: {default_study_patience})",
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
        help=f"Database storage URL (default: {default_storage})",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat recording interval in seconds for distributed trials (default: 30)",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=None,
        help="Grace period before failing stale trials (default: 2 * heartbeat_interval)",
    )
    return parser
