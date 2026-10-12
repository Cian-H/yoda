"""Unit tests for the focused Optuna hyperparameter study module."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna

from experiments.optuna_study_focused import (
    CHAMPION_TRIAL_40_PARAMS,
    FIXED_BACKBONE_LR,
    FIXED_BELNAP_WEIGHT,
    FIXED_D_HIDDEN_MULTIPLIER,
    FIXED_DROPOUT,
    FIXED_LR,
    FIXED_LR_DECAY,
    FIXED_MARGIN_WEIGHT,
    FIXED_N_HEADS,
    FIXED_NUM_C_PROBES,
    FIXED_NUM_REASONING_BLOCKS,
    FIXED_RESIDUAL_WEIGHT,
    FIXED_T0_EPOCHS,
    FIXED_T_MULT,
    FIXED_TEXT_MODEL_NAME,
    objective,
)


def test_fixed_parameters_consistency() -> None:
    """Verify fixed parameters match the verified champion configurations."""
    assert FIXED_TEXT_MODEL_NAME == "BAAI/bge-large-en-v1.5"
    assert FIXED_NUM_REASONING_BLOCKS == 1
    assert FIXED_N_HEADS == 2
    assert FIXED_T0_EPOCHS == 3
    assert FIXED_T_MULT == 2
    assert FIXED_RESIDUAL_WEIGHT == 0.6356
    assert FIXED_DROPOUT == 0.2857
    assert FIXED_MARGIN_WEIGHT == 0.6180

    # Newly locked parameters
    assert FIXED_LR == 3.0e-4
    assert FIXED_BACKBONE_LR == 6.0e-6
    assert FIXED_LR_DECAY == 0.60
    assert FIXED_NUM_C_PROBES == 14
    assert FIXED_BELNAP_WEIGHT == 1.25
    assert FIXED_D_HIDDEN_MULTIPLIER == 1.4


def test_champion_params_within_search_space() -> None:
    """Verify that enqueued champion Trial #40 parameters fall within the focused bounds."""
    assert 7 <= CHAMPION_TRIAL_40_PARAMS["num_q_probes"] <= 12
    assert 0.48 <= CHAMPION_TRIAL_40_PARAMS["conflation_weight"] <= 0.65


def test_objective_command_construction(tmp_path: Path) -> None:
    """Verify objective function correctly builds parameter dictionary and executes trial."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_int.return_value = 8
    trial.suggest_float.return_value = 0.55

    args = argparse.Namespace(
        max_samples=16000,
        epochs=6,
        batch_size=8,
        early_stopping_patience=2,
        train_path=None,
    )

    with patch("experiments.optuna_study_focused.run_training_trial") as mock_runner:
        mock_runner.return_value = 0.525
        val = objective(trial, args)

        assert val == 0.525
        mock_runner.assert_called_once()
        _, trial_dir, params = mock_runner.call_args[0]
        assert trial_dir == Path("models/optuna_focused/trial_0")
        assert params["num_q_probes"] == 8
        assert params["conflation_weight"] == 0.55
        assert params["lr"] == 3.0e-4
        assert params["backbone_lr"] == 6.0e-6
        assert params["num_c_probes"] == 14
        assert params["d_hidden_multiplier"] == 1.4
