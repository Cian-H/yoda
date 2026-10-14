"""Unit tests for the post-hardening v2 Optuna study."""

import argparse
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from experiments import analyze_study
from experiments import optuna_study_v2 as v2

DSN_WITH_CREDENTIALS = re.compile(r"postgresql\+psycopg://\S+@\S+")


def make_mock_trial() -> MagicMock:
    trial = MagicMock()
    trial.number = 0
    trial.suggest_int.return_value = 2
    trial.suggest_float.return_value = 0.2
    trial.suggest_categorical.return_value = 2
    return trial


def captured(
    trial: MagicMock,
) -> tuple[dict[str, tuple], dict[str, tuple], dict[str, list]]:
    floats: dict[str, tuple] = {}
    ints: dict[str, tuple] = {}
    cats: dict[str, list] = {}
    for call in trial.suggest_float.call_args_list:
        floats[call[0][0]] = (call[0][1], call[0][2], call[1].get("log", False))
    for call in trial.suggest_int.call_args_list:
        ints[call[0][0]] = (call[0][1], call[0][2])
    for call in trial.suggest_categorical.call_args_list:
        cats[call[0][0]] = call[0][1]
    return floats, ints, cats


def make_args(**overrides: Any) -> argparse.Namespace:
    base = {
        "phase": "broad",
        "train_path": "data/processed/v2/train.parquet",
        "eval_path": "data/processed/v2/optuna_eval.parquet",
        "test_path": "data/processed/v2/test.parquet",
        "max_samples": 16000,
        "epochs": 8,
        "batch_size": 8,
        "early_stopping_patience": 2,
        "collapse_threshold": 0.33,
        "collapse_patience_epochs": 2,
        "best_params_json": None,
        "champion_json": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_broad_bounds_match_verdict() -> None:
    """Search bounds must conform to the M4-hardened ranges plus the new parameters."""
    trial = make_mock_trial()
    v2.suggest_broad_params(trial)
    floats, ints, cats = captured(trial)

    assert floats["conflation_weight"] == (0.01, 0.25, True)
    assert floats["residual_weight"] == (0.10, 0.25, False)
    assert floats["lr"] == (1e-5, 5e-4, True)
    assert floats["backbone_lr"] == (1e-6, 1e-4, True)
    assert floats["ltn_weight"] == (0.01, 1.0, True)
    assert floats["assertion_weight"] == (0.0, 0.5, False)
    assert floats["decisiveness_weight"] == (0.001, 0.05, True)
    assert floats["logit_scale"] == (2.0, 32.0, True)
    assert floats["d_hidden_multiplier"] == (0.8, 4.0, False)
    assert floats["dropout"] == (0.0, 0.3, False)
    assert floats["belnap_weight"] == (0.5, 2.0, False)
    assert floats["margin_weight"] == (0.1, 1.0, False)
    assert floats["lr_decay"] == (0.5, 1.0, False)

    assert ints["num_reasoning_blocks"] == (1, 3)
    assert ints["num_q_probes"] == (2, 16)
    assert ints["num_c_probes"] == (8, 32)
    assert ints["num_k_probes"] == (4, 16)
    assert ints["t0_epochs"] == (1, 5)
    assert ints["t_mult"] == (1, 3)
    assert ints["decisiveness_warmup_epochs"] == (1, 3)

    assert cats["n_heads"] == [2, 4, 8]

    # min_lr is wired to lr/100, never scanned.
    assert "min_lr" not in floats


def test_objective_passes_explicit_params_and_metric() -> None:
    """Every trial passes explicit runtime/paths; no CLI defaults; macro mean_last2 metric."""
    trial = make_mock_trial()
    args = make_args()
    with patch.object(v2, "run_training_trial", return_value=0.5) as mock_run:
        val = v2.objective(trial, args)

    assert val == 0.5
    _, trial_dir, params = mock_run.call_args[0]
    assert trial_dir == Path("models/optuna_v2/trial_0")
    assert params["min_lr"] == params["lr"] / 100.0
    assert params["text_model_name"] == "BAAI/bge-large-en-v1.5"
    assert params["decisiveness_weight"] == 0.2
    assert params["decisiveness_warmup_epochs"] == 2
    assert params["max_train_samples"] == 16000
    assert params["train_path"].endswith("train.parquet")
    assert params["eval_path"].endswith("optuna_eval.parquet")
    assert params["test_path"].endswith("test.parquet")
    assert params["collapse_threshold"] == 0.33
    assert params["collapse_patience_epochs"] == 2
    assert params["epochs"] == 8
    assert params["batch_size"] == 8
    assert params["early_stopping_patience"] == 2
    assert mock_run.call_args.kwargs["metric_name"] == "macro_accuracy"
    assert mock_run.call_args.kwargs["metric_agg"] == "mean_last2"


def test_validate_phase_uses_best_epoch_metric() -> None:
    trial = make_mock_trial()
    args = make_args(phase="validate")
    with patch.object(v2, "run_training_trial", return_value=0.5) as mock_run:
        v2.objective(trial, args)
    assert mock_run.call_args.kwargs["metric_agg"] == "best"


def test_warm_start_points_in_bounds() -> None:
    """Enqueued champions and the anchor must lie inside the suggested distributions."""
    trial = make_mock_trial()
    v2.suggest_broad_params(trial)
    floats, ints, cats = captured(trial)

    for point in [*v2.CLIPPED_CHAMPIONS, v2.ANCHOR_PARAMS]:
        assert set(point) == set(floats) | set(ints) | set(cats)
        for name, value in point.items():
            if name in floats:
                low, high, _ = floats[name]
                assert low <= value <= high, f"{name}={value} outside [{low}, {high}]"
            elif name in ints:
                low, high = ints[name]
                assert low <= value <= high, f"{name}={value} outside [{low}, {high}]"
                assert float(value).is_integer()
            else:
                assert value in cats[name], f"{name}={value} not in {cats[name]}"


def test_focused_probes_narrow_around_base() -> None:
    trial = make_mock_trial()
    base = dict(v2.ANCHOR_PARAMS)
    params = v2.suggest_focused_params(trial, base)
    floats, ints, cats = captured(trial)

    assert set(floats) == {
        "decisiveness_weight",
        "conflation_weight",
        "logit_scale",
        "margin_weight",
    }
    assert set(ints) == {"num_q_probes"}
    assert cats == {}
    assert floats["decisiveness_weight"][:2] == (0.0025, 0.04)
    assert floats["conflation_weight"][:2] == (0.025, 0.25)
    assert floats["logit_scale"][:2] == (2.0, 32.0)
    assert floats["margin_weight"][:2] == pytest.approx((0.1, 0.3))
    assert ints["num_q_probes"] == (2, 6)

    # Everything else locked at the base values.
    assert params["lr"] == base["lr"]
    assert params["backbone_lr"] == base["backbone_lr"]
    assert params["num_c_probes"] == base["num_c_probes"]
    assert params["residual_weight"] == base["residual_weight"]


def test_phase_presets() -> None:
    assert set(v2.PHASE_PRESETS) == {"sanity", "broad", "focused", "validate"}
    assert v2.PHASE_PRESETS["sanity"]["n_trials"] == 5
    assert v2.PHASE_PRESETS["validate"]["eval_split"] == "test"
    assert v2.PHASE_PRESETS["validate"]["max_samples"] is None
    for preset in v2.PHASE_PRESETS.values():
        assert preset["eval_split"] in ("optuna_eval", "test")


def test_no_hardcoded_remote_storage_or_secrets() -> None:
    """Studies default to local storage; no credentialed DSN anywhere in code.

    Repo-wide guard: the remote Postgres connect string is temporary and must
    never be committed — it lives only in the run command (via the
    YODA_OPTUNA_STORAGE env var). The scan matches any DSN of the form
    postgresql+psycopg://<user>:<password>@<host> without embedding fragments
    of an actual credential.
    """
    assert v2.DEFAULT_STORAGE == "sqlite:///data/optuna_v2.db"
    assert "YODA_OPTUNA_STORAGE" in Path(v2.__file__).read_text(encoding="utf-8")
    assert "YODA_OPTUNA_STORAGE" in Path(analyze_study.__file__).read_text(encoding="utf-8")
    scanned = [*Path("src").rglob("*.py"), *Path("experiments").rglob("*.py")]
    assert scanned, "expected to scan source files"
    for py_file in scanned:
        source = py_file.read_text(encoding="utf-8")
        assert not DSN_WITH_CREDENTIALS.search(source), f"hardcoded DSN in {py_file}"


def test_metric_constants() -> None:
    assert v2.METRIC_NAME == "macro_accuracy"
    assert v2.METRIC_AGG == "mean_last2"
    assert v2.FIXED_TEXT_MODEL_NAME == "BAAI/bge-large-en-v1.5"


def test_resolve_storage_prefers_flag_then_env_then_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage resolution: explicit flag > YODA_OPTUNA_STORAGE > local sqlite default."""
    monkeypatch.delenv("YODA_OPTUNA_STORAGE", raising=False)
    assert v2.resolve_storage(None) == "sqlite:///data/optuna_v2.db"
    monkeypatch.setenv("YODA_OPTUNA_STORAGE", "sqlite:///data/from_env.db")
    assert v2.resolve_storage(None) == "sqlite:///data/from_env.db"
    assert v2.resolve_storage("sqlite:///data/explicit.db") == "sqlite:///data/explicit.db"


def test_main_fails_fast_when_split_files_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing v2 split files must abort at startup, not prune every trial."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "optuna_study_v2.py",
            "--phase",
            "sanity",
            "--train-path",
            str(tmp_path / "missing_train.parquet"),
            "--eval-path",
            str(tmp_path / "missing_eval.parquet"),
        ],
    )
    with pytest.raises(SystemExit, match="train dataset not found"):
        v2.main()


def test_main_validate_phase_requires_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validate phase must fail fast when the locked test split is missing."""
    train = tmp_path / "train.parquet"
    eval_file = tmp_path / "eval.parquet"
    train.touch()
    eval_file.touch()
    monkeypatch.setattr(
        "sys.argv",
        [
            "optuna_study_v2.py",
            "--phase",
            "validate",
            "--train-path",
            str(train),
            "--eval-path",
            str(eval_file),
            "--test-path",
            str(tmp_path / "missing_test.parquet"),
        ],
    )
    with pytest.raises(SystemExit, match="test dataset not found"):
        v2.main()
