"""Tests for the base Optuna study objective search-space realignment (R5)."""

import argparse
from unittest.mock import MagicMock, patch

from experiments.optuna_study import objective


def _mock_trial() -> MagicMock:
    trial = MagicMock()
    trial.suggest_categorical.return_value = "dummy"
    trial.suggest_int.return_value = 2
    trial.suggest_float.return_value = 0.2
    return trial


def _mock_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_samples=10,
        epochs=1,
        batch_size=2,
        early_stopping_patience=1,
        train_path=None,
    )


@patch("experiments.optuna_study.run_training_trial")
def test_objective_bounds_match_hardened_architecture(mock_run: MagicMock) -> None:
    """Search bounds must conform to the hardened stable ranges."""
    trial = _mock_trial()
    objective(trial, _mock_args())

    float_calls = {c[0][0]: c[0][1:] for c in trial.suggest_float.call_args_list}
    assert float_calls["conflation_weight"][:2] == (0.01, 0.25)
    assert float_calls["residual_weight"][:2] == (0.10, 0.25)
    assert float_calls["lr"][:2] == (1e-5, 5e-4)
    assert float_calls["decisiveness_weight"][:2] == (0.001, 0.05)


@patch("experiments.optuna_study.run_training_trial")
def test_objective_includes_decisiveness_warmup(mock_run: MagicMock) -> None:
    """The contradiction penalty must be searched with linear warmup epochs."""
    trial = _mock_trial()
    objective(trial, _mock_args())

    int_names = [c[0][0] for c in trial.suggest_int.call_args_list]
    assert "decisiveness_warmup_epochs" in int_names


@patch("experiments.optuna_study.run_training_trial")
def test_objective_passes_decisiveness_params_to_trials(mock_run: MagicMock) -> None:
    """Decisiveness weight and warmup must reach the training trial arguments."""
    trial = _mock_trial()
    objective(trial, _mock_args())

    params = mock_run.call_args[0][2]
    assert "decisiveness_weight" in params
    assert "decisiveness_warmup_epochs" in params
