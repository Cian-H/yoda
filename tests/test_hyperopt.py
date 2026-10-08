"""Unit tests for yoda.hyperopt.optuna module."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

from yoda.hyperopt.optuna import (
    StudyEarlyStoppingCallback,
    build_base_study_parser,
    build_training_command,
    init_study_and_metadata,
    run_training_trial,
)


def test_build_training_command(tmp_path: Path) -> None:
    """Verify build_training_command converts argument dictionaries to CLI flags."""
    trial_dir = tmp_path / "trial_0"
    args = {
        "text_model_name": "BAAI/bge-large-en-v1.5",
        "num_reasoning_blocks": 2,
        "lr": 1e-4,
        "freeze_backbone": True,
        "shuffle_choices": False,
        "ignored_none": None,
    }

    cmd = build_training_command(trial_dir, args)

    assert cmd[0:3] == ["uv", "run", "python"]
    assert cmd[3] == "experiments/train_yoda.py"
    assert "--text-model-name" in cmd
    assert cmd[cmd.index("--text-model-name") + 1] == "BAAI/bge-large-en-v1.5"
    assert "--num-reasoning-blocks" in cmd
    assert cmd[cmd.index("--num-reasoning-blocks") + 1] == "2"
    assert "--lr" in cmd
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert "--freeze-backbone" in cmd
    assert "--no-shuffle-choices" in cmd
    assert "--ignored-none" not in cmd
    assert "--checkpoint-dir" in cmd
    assert cmd[cmd.index("--checkpoint-dir") + 1] == str(trial_dir)


def test_study_early_stopping_callback() -> None:
    """Verify StudyEarlyStoppingCallback halts study when patience is exhausted."""
    callback = StudyEarlyStoppingCallback(patience=3)
    study = optuna.create_study(direction="maximize")

    def dummy_objective(trial: optuna.Trial) -> float:
        if trial.number == 0:
            return 10.0
        return float(5.0 - trial.number)

    study.optimize(dummy_objective, n_trials=10, callbacks=[callback])

    assert len(study.trials) == 4
    assert study.best_trial.number == 0
    assert callback.best_trial_number == 0
    assert callback.best_value == 10.0


def test_run_training_trial_success(tmp_path: Path) -> None:
    """Verify run_training_trial succeeds and parses target metric."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial_dir = tmp_path / "trial_success"

    history_data = [
        {"epoch": 1, "eval_accuracy": 0.42},
        {"epoch": 2, "eval_accuracy": 0.51},
    ]

    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open"),
        patch("json.load", return_value=history_data),
    ):
        result = run_training_trial(trial, trial_dir, {"lr": 1e-4}, metric_name="eval_accuracy")
        assert result == 0.51


def test_run_training_trial_subprocess_failure(tmp_path: Path) -> None:
    """Verify run_training_trial prunes trial on subprocess error."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial_dir = tmp_path / "trial_fail"

    mock_run = MagicMock()
    mock_run.returncode = 1

    with (
        patch("subprocess.run", return_value=mock_run),
        pytest.raises(optuna.TrialPruned, match="Subprocess failed"),
    ):
        run_training_trial(trial, trial_dir, {})


def test_run_training_trial_missing_history(tmp_path: Path) -> None:
    """Verify run_training_trial prunes trial when history.json is missing."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial_dir = tmp_path / "trial_no_hist"

    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run),
        patch("pathlib.Path.exists", return_value=False),
        pytest.raises(optuna.TrialPruned, match=r"history\.json not found"),
    ):
        run_training_trial(trial, trial_dir, {})


def test_init_study_and_metadata(tmp_path: Path) -> None:
    """Verify init_study_and_metadata creates storage, study, and metadata JSON."""
    storage = f"sqlite:///{tmp_path}/test_optuna.db"
    study_name = "test_sweep_123"

    study, metadata = init_study_and_metadata(
        study_name=study_name,
        storage=storage,
        seed=12345,
        metadata_dir=tmp_path,
        extra_metadata={"n_trials": 10},
    )

    assert study.study_name == study_name
    assert metadata["study_name"] == study_name
    assert metadata["seed"] == 12345
    assert metadata["n_trials"] == 10

    metadata_file = tmp_path / f"{study_name}_metadata.json"
    assert metadata_file.exists()
    with metadata_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["seed"] == 12345


def test_build_base_study_parser() -> None:
    """Verify build_base_study_parser configures expected CLI flags."""
    parser = build_base_study_parser("Test study parser")
    args = parser.parse_args(["--n-trials", "20", "--epochs", "5"])
    assert args.n_trials == 20
    assert args.epochs == 5
    assert args.batch_size == 8


def test_optuna_study_objective(tmp_path: Path) -> None:
    """Verify experiments.optuna_study.objective runs with yoda.hyperopt abstractions."""
    from experiments.optuna_study import objective as broad_objective

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    args = argparse.Namespace(
        train_path=None,
        max_samples=8000,
        epochs=4,
        batch_size=8,
        early_stopping_patience=1,
    )

    history_content = [{"epoch": 1, "eval_accuracy": 0.45}]
    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run) as patched_run,
        patch("pathlib.Path.open"),
        patch("json.load", return_value=history_content),
        patch("pathlib.Path.exists", return_value=True),
    ):
        acc = broad_objective(trial, args)
        assert acc == 0.45
        patched_run.assert_called_once()
