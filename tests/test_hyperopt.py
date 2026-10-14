"""Unit tests for yoda.hyperopt.optuna module."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

from yoda.hyperopt.optuna import (
    StudyEarlyStoppingCallback,
    aggregate_metric_values,
    build_base_study_parser,
    build_training_command,
    extract_metric_value,
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


def test_init_study_and_metadata_heartbeat(tmp_path: Path) -> None:
    """Verify init_study_and_metadata configures RDBStorage heartbeat for distributed runs."""
    db_path = tmp_path / "distributed_test.db"
    storage_url = f"sqlite:///{db_path}"

    study, metadata = init_study_and_metadata(
        study_name="dist_study_test",
        storage=storage_url,
        heartbeat_interval=15,
        grace_period=45,
        metadata_dir=tmp_path,
    )

    assert study._storage.get_heartbeat_interval() == 15
    assert metadata["heartbeat_interval"] == 15
    assert metadata["grace_period"] == 45


def test_build_base_study_parser_heartbeat_flags() -> None:
    """Verify parser parses --heartbeat-interval and --grace-period."""
    parser = build_base_study_parser("Test")
    args = parser.parse_args(["--heartbeat-interval", "45", "--grace-period", "120"])
    assert args.heartbeat_interval == 45
    assert args.grace_period == 120


def test_build_base_study_parser_collapse_flags() -> None:
    """Verify parser parses --collapse-threshold and --collapse-patience-epochs."""
    parser = build_base_study_parser("Test")
    default_args = parser.parse_args([])
    assert default_args.collapse_threshold == 0.370
    assert default_args.collapse_patience_epochs == 2

    custom_args = parser.parse_args(
        [
            "--collapse-threshold",
            "0.40",
            "--collapse-patience-epochs",
            "3",
        ]
    )
    assert custom_args.collapse_threshold == 0.40
    assert custom_args.collapse_patience_epochs == 3


def test_run_training_trial_seed_randomization(tmp_path: Path) -> None:
    """Verify run_training_trial injects a randomized non-42 seed when not specified."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial_dir = tmp_path / "trial_seed_test"

    history_data = [{"epoch": 1, "eval_accuracy": 0.49}]
    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run) as patched_run,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open"),
        patch("json.load", return_value=history_data),
    ):
        run_training_trial(trial, trial_dir, {"lr": 1e-4})
        assert "seed" in trial.user_attrs
        assert trial.user_attrs["seed"] != 42
        assert isinstance(trial.user_attrs["seed"], int)

        # Check command passed to subprocess.run
        cmd = patched_run.call_args[0][0]
        assert "--seed" in cmd
        seed_val = int(cmd[cmd.index("--seed") + 1])
        assert seed_val == trial.user_attrs["seed"]
        assert seed_val != 42


def test_aggregate_metric_values() -> None:
    """Verify metric aggregation modes over per-epoch values."""
    assert aggregate_metric_values([0.4, 0.5, 0.45], agg="final") == 0.45
    assert aggregate_metric_values([0.4, 0.5, 0.45], agg="best") == 0.5
    assert aggregate_metric_values([0.4, 0.5, 0.45], agg="mean_last2") == pytest.approx(0.475)
    assert aggregate_metric_values([0.42], agg="mean_last2") == 0.42
    with pytest.raises(ValueError, match="Unknown metric aggregation"):
        aggregate_metric_values([0.4], agg="median")
    with pytest.raises(ValueError, match="empty"):
        aggregate_metric_values([], agg="final")


def test_extract_metric_value() -> None:
    """Verify metric extraction from history records, including the missing-metric case."""
    history = [{"epoch": 1, "eval_accuracy": 0.4}, {"epoch": 2, "eval_accuracy": 0.5}]
    assert extract_metric_value(history) == 0.5
    assert extract_metric_value(history, agg="best") == 0.5
    assert extract_metric_value(history, agg="mean_last2") == pytest.approx(0.45)
    assert extract_metric_value([{"epoch": 1}]) == 0.0
    assert extract_metric_value([], agg="best") == 0.0


def test_build_base_study_parser_explicit_paths() -> None:
    """Verify the parser exposes explicit eval/test dataset paths."""
    parser = build_base_study_parser("Test")
    args = parser.parse_args([])
    assert args.eval_path is None
    assert args.test_path is None
    args = parser.parse_args(["--eval-path", "e.parquet", "--test-path", "t.parquet"])
    assert args.eval_path == "e.parquet"
    assert args.test_path == "t.parquet"


def test_run_training_trial_metric_agg_and_user_attrs(tmp_path: Path) -> None:
    """Verify metric_agg aggregation and audit user attrs on the trial."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial_dir = tmp_path / "trial_agg"

    history_data = [
        {"epoch": 1, "macro_accuracy": 0.40},
        {"epoch": 2, "macro_accuracy": 0.50},
        {"epoch": 3, "macro_accuracy": 0.45},
    ]
    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.open"),
        patch("json.load", return_value=history_data),
    ):
        result = run_training_trial(
            trial, trial_dir, {}, metric_name="macro_accuracy", metric_agg="mean_last2"
        )

    assert result == pytest.approx(0.475)
    assert trial.user_attrs["final_metric"] == 0.45
    assert trial.user_attrs["best_metric"] == 0.5
    assert trial.user_attrs["best_epoch"] == 2
    assert trial.user_attrs["metric_name"] == "macro_accuracy"
    assert trial.user_attrs["metric_agg"] == "mean_last2"
