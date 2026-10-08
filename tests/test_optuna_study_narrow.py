"""Unit tests for the narrowed Optuna hyperparameter study module."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna

from experiments.optuna_study_narrow import (
    CHAMPION_TRIAL_21_PARAMS,
    FIXED_DROPOUT,
    FIXED_MARGIN_WEIGHT,
    FIXED_N_HEADS,
    FIXED_NUM_REASONING_BLOCKS,
    FIXED_RESIDUAL_WEIGHT,
    FIXED_T0_EPOCHS,
    FIXED_T_MULT,
    FIXED_TEXT_MODEL_NAME,
    objective,
)
from yoda.hyperopt.optuna import StudyEarlyStoppingCallback


def test_fixed_parameters_consistency() -> None:
    """Verify fixed parameters match the champion trial and requirements."""
    assert FIXED_TEXT_MODEL_NAME == "BAAI/bge-large-en-v1.5"
    assert FIXED_NUM_REASONING_BLOCKS == 1
    assert FIXED_N_HEADS == 2
    assert FIXED_T0_EPOCHS == 3
    assert FIXED_T_MULT == 2
    assert FIXED_RESIDUAL_WEIGHT == 0.6356
    assert FIXED_DROPOUT == 0.2857
    assert FIXED_MARGIN_WEIGHT == 0.6180


def test_champion_params_within_search_space() -> None:
    """Verify that enqueued champion Trial #21 parameters fall within the narrowed bounds."""
    assert 1.0e-4 <= CHAMPION_TRIAL_21_PARAMS["lr"] <= 3.5e-4
    assert 5.0e-6 <= CHAMPION_TRIAL_21_PARAMS["backbone_lr"] <= 4.5e-5
    assert 0.55 <= CHAMPION_TRIAL_21_PARAMS["lr_decay"] <= 0.75
    assert 12 <= CHAMPION_TRIAL_21_PARAMS["num_c_probes"] <= 20
    assert 4 <= CHAMPION_TRIAL_21_PARAMS["num_q_probes"] <= 7
    assert 1.4 <= CHAMPION_TRIAL_21_PARAMS["d_hidden_multiplier"] <= 3.9
    assert 0.75 <= CHAMPION_TRIAL_21_PARAMS["belnap_weight"] <= 1.45
    assert 0.35 <= CHAMPION_TRIAL_21_PARAMS["conflation_weight"] <= 0.50


def test_study_early_stopping_callback() -> None:
    """Verify that StudyEarlyStoppingCallback triggers stop() after patience without improvement."""
    callback = StudyEarlyStoppingCallback(patience=3)
    study = optuna.create_study(direction="maximize")

    # Values decrease after trial 0: trial 0 is best (10.0), trials 1-3 do not improve
    def dummy_objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 0:
            return 10.0
        return float(10.0 - trial.number + 0.01 * x)

    study.optimize(dummy_objective, n_trials=15, callbacks=[callback])

    # Should stop after trial 3 (trial 0 + 3 trials without improvement = 4 trials total)
    assert len(study.trials) == 4
    assert study.best_trial.number == 0
    assert callback.best_trial_number == 0
    assert callback.best_value == 10.0


def test_objective_command_construction(tmp_path: Path) -> None:
    """Verify objective function constructs expected subprocess command and parses history."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    args = argparse.Namespace(
        train_path=None,
        max_samples=16000,
        epochs=6,
        batch_size=8,
        early_stopping_patience=2,
    )

    history_content = [{"epoch": 1, "eval_accuracy": 0.48}]
    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run) as patched_run,
        patch("pathlib.Path.open"),
        patch("json.load", return_value=history_content),
        patch("pathlib.Path.exists", return_value=True),
    ):
        acc = objective(trial, args)

        assert acc == 0.48
        patched_run.assert_called_once()
        cmd = patched_run.call_args[0][0]

        # Verify command arguments
        assert "--text-model-name" in cmd
        idx_backbone = cmd.index("--text-model-name") + 1
        assert cmd[idx_backbone] == "BAAI/bge-large-en-v1.5"

        assert "--num-reasoning-blocks" in cmd
        idx_blocks = cmd.index("--num-reasoning-blocks") + 1
        assert cmd[idx_blocks] == "1"

        assert "--max-samples" in cmd
        idx_samples = cmd.index("--max-samples") + 1
        assert cmd[idx_samples] == "16000"

        assert "--epochs" in cmd
        idx_epochs = cmd.index("--epochs") + 1
        assert cmd[idx_epochs] == "6"

        assert "--early-stopping-patience" in cmd
        idx_patience = cmd.index("--early-stopping-patience") + 1
        assert cmd[idx_patience] == "2"
