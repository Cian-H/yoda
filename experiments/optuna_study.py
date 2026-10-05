"""Hyperparameter optimization study for YodaDecisionEngine using Optuna."""

import argparse
import json
import subprocess
from pathlib import Path

import optuna
from loguru import logger


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function evaluating YodaDecisionEngine configurations.

    Args:
        trial: An Optuna trial instance.

    Returns:
        Final epoch evaluation accuracy (eval_accuracy) to maximize.
    """
    # 1. Text model backbone selection
    text_model_name = trial.suggest_categorical(
        "text_model_name",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "EmbeddingGemma300M",
            "harrier-oss-v1-0.6b",
            "Qwen3-Embedding-0.6B",
        ],
    )

    # 2. Topological parameters
    num_reasoning_blocks = trial.suggest_int("num_reasoning_blocks", 1, 3)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_hidden_multiplier = trial.suggest_float("d_hidden_multiplier", 1.0, 4.0)
    num_q_probes = trial.suggest_int("num_q_probes", 2, 8)
    num_c_probes = trial.suggest_int("num_c_probes", 8, 32)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    conflation_weight = trial.suggest_float("conflation_weight", 0.0, 0.5)
    residual_weight = trial.suggest_float("residual_weight", 0.1, 0.9)

    # 3. Training parameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 1e-4, log=True)
    t0_epochs = trial.suggest_int("t0_epochs", 1, 3)

    # 4. Generate trial checkpoint directory
    trial_dir = Path("models/optuna") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/train_yoda.py",
        "--text-model-name",
        str(text_model_name),
        "--num-reasoning-blocks",
        str(num_reasoning_blocks),
        "--n-heads",
        str(n_heads),
        "--d-hidden-multiplier",
        str(d_hidden_multiplier),
        "--num-q-probes",
        str(num_q_probes),
        "--num-c-probes",
        str(num_c_probes),
        "--dropout",
        str(dropout),
        "--conflation-weight",
        str(conflation_weight),
        "--residual-weight",
        str(residual_weight),
        "--lr",
        str(lr),
        "--backbone-lr",
        str(backbone_lr),
        "--t0-epochs",
        str(t0_epochs),
        "--checkpoint-dir",
        str(trial_dir),
        "--output-model",
        str(trial_dir / "model.pt"),
        "--epochs",
        "4",
        "--max-samples",
        "2000",
        "--batch-size",
        "16",
    ]

    logger.info(
        "Starting Trial #{} with backbone={} lr={:.2e} backbone_lr={:.2e} blocks={}",
        trial.number,
        text_model_name,
        lr,
        backbone_lr,
        num_reasoning_blocks,
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
    """Creates the Optuna study and runs the optimization sweep."""
    parser = argparse.ArgumentParser(description="Run Optuna study for YodaDecisionEngine.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna optimization trials (default: 50)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="yoda_hpo_study",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional database storage URL (e.g., sqlite:///optuna.db)",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.n_trials)

    try:
        logger.info("Optuna study completed.")
        logger.info("Best trial: #{}", study.best_trial.number)
        logger.info("Best eval_accuracy: {:.4f}", study.best_value)
        logger.info("Best parameters: {}", study.best_params)
    except ValueError:
        logger.warning("No trials completed successfully.")


if __name__ == "__main__":
    main()
