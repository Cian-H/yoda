"""Unit tests for experiments/analyze_study.py."""

import optuna
import pytest

from experiments import analyze_study


def test_missing_study_name_reports_available(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong --study-name exits with the available studies, not a raw KeyError."""
    storage = f"sqlite:///{tmp_path}/studies.db"
    optuna.create_study(study_name="real_study", storage=storage)

    monkeypatch.setattr(
        "sys.argv",
        ["analyze_study.py", "--study-name", "nope", "--storage", storage, "--no-write"],
    )
    with pytest.raises(SystemExit, match=r"Study 'nope' not found.*real_study"):
        analyze_study.main()


def test_no_studies_in_storage_exits_cleanly(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty storage exits with a clear message when no study name is given."""
    storage = f"sqlite:///{tmp_path}/empty.db"
    monkeypatch.setattr("sys.argv", ["analyze_study.py", "--storage", storage, "--no-write"])
    with pytest.raises(SystemExit, match="No studies found"):
        analyze_study.main()


def test_analyze_lists_top_trials(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The happy path prints the trial table without writing files."""
    storage = f"sqlite:///{tmp_path}/studies.db"
    study = optuna.create_study(study_name="s", storage=storage, direction="maximize")
    study.optimize(lambda t: 0.5, n_trials=2)

    monkeypatch.setattr(
        "sys.argv",
        ["analyze_study.py", "--study-name", "s", "--storage", storage, "--no-write"],
    )
    analyze_study.main()
    out = capsys.readouterr().out
    assert "2 complete trials" in out
