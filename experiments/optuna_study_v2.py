"""Post-hardening hyperparameter study (v2) for YodaDecisionEngine.

Implements the council-verdict conditions (2026-10-14):

- Gate 0 (data): explicit group-disjoint, source-stratified v2 split paths from
  ``experiments/prepare_hpo_split.py``; collapse threshold derived from the
  objective set's macro const-predictor rate.
- Gate 1 (metric): objective = macro-averaged per-source eval accuracy, mean of
  the last two epochs (``metric_name="macro_accuracy"``, ``metric_agg="mean_last2"``).
- Gate 2 (space): M4-hardened ranges plus previously unscanned parameters;
  ``min_lr`` wired to ``lr / 100`` (eta_min inversion fix); ``logit_scale``
  exposed and scanned; ``backbone_lr`` retained (the backbone is unfrozen by
  default, so the knob is live).
- Gate 3 (mechanics): fresh study per phase; in-bounds warm start; phased runs.

Phases (``--phase``):
  sanity   Anchor config x 5 seeds. Operator gates: mean macro-acc >
           const-predictor + 0.05, seed sigma < 0.02, collapse-prune rate < 40%,
           wall-clock per trial recorded.
  broad    Full space, warm-started; fANOVA checkpoints at trials 100/200 via
           ``experiments/analyze_study.py``.
  focused  Narrowed probes around a Phase-B best-params JSON (``--best-params-json``):
           decisiveness_weight x conflation_weight, logit_scale x margin_weight,
           num_q_probes.
  validate Top-K configs x fixed seeds on the locked test split
           (``--champion-json``), best-epoch metric, full-budget training.

Storage: local SQLite by default — no credentials in the repo. For distributed
runs on capable GPUs, export YODA_OPTUNA_STORAGE with a Postgres RDB URL
(postgresql+psycopg scheme) and launch 2-4 single-job workers against the same
study name; heartbeat monitoring and stale-trial recovery are wired in
``init_study_and_metadata``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

from yoda.hyperopt.optuna import (
    StudyEarlyStoppingCallback,
    build_base_study_parser,
    init_study_and_metadata,
    run_training_trial,
)

METRIC_NAME = "macro_accuracy"
METRIC_AGG = "mean_last2"
FIXED_TEXT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SPLIT_METADATA_PATH = Path("data/processed/v2/split_metadata.json")
DEFAULT_STORAGE = "sqlite:///data/optuna_v2.db"
VALIDATE_SEEDS = [11, 22, 33, 44, 55]

# Hardened-defaults anchor: mid-range, in-bounds reference point for the warm
# start and the Phase-0 sanity config.
ANCHOR_PARAMS: dict[str, Any] = {
    "num_reasoning_blocks": 1,
    "n_heads": 4,
    "d_hidden_multiplier": 2.0,
    "num_q_probes": 4,
    "num_c_probes": 16,
    "num_k_probes": 8,
    "dropout": 0.1,
    "conflation_weight": 0.1,
    "residual_weight": 0.15,
    "lr": 1e-4,
    "backbone_lr": 1e-5,
    "t0_epochs": 3,
    "t_mult": 2,
    "lr_decay": 0.75,
    "belnap_weight": 1.0,
    "margin_weight": 0.1,
    "ltn_weight": 0.1,
    "assertion_weight": 0.1,
    "decisiveness_weight": 0.01,
    "decisiveness_warmup_epochs": 2,
    "logit_scale": 8.0,
}

# Old champions clipped into the M4-hardened ranges. Reference points only:
# their conflation/residual optima were products of a contaminated objective.
CLIPPED_CHAMPIONS: list[dict[str, Any]] = [
    # broad yoda_sweep_1791233751 trial #21 (conflation 0.4256->0.25, residual 0.6356->0.25)
    {
        "num_reasoning_blocks": 1,
        "n_heads": 2,
        "d_hidden_multiplier": 1.9025,
        "num_q_probes": 6,
        "num_c_probes": 13,
        "num_k_probes": 5,
        "dropout": 0.2857,
        "conflation_weight": 0.25,
        "residual_weight": 0.25,
        "lr": 2.7843e-4,
        "backbone_lr": 8.919e-6,
        "t0_epochs": 3,
        "t_mult": 2,
        "lr_decay": 0.5825,
        "belnap_weight": 1.3556,
        "margin_weight": 0.618,
        "ltn_weight": 0.1,
        "assertion_weight": 0.1,
        "decisiveness_weight": 0.01,
        "decisiveness_warmup_epochs": 2,
        "logit_scale": 8.0,
    },
    # narrow yoda_narrow_sweep_1791462045 trial #40 (conflation 0.489->0.25, residual->0.25)
    {
        "num_reasoning_blocks": 1,
        "n_heads": 2,
        "d_hidden_multiplier": 1.4166,
        "num_q_probes": 7,
        "num_c_probes": 14,
        "num_k_probes": 5,
        "dropout": 0.2857,
        "conflation_weight": 0.25,
        "residual_weight": 0.25,
        "lr": 2.966e-4,
        "backbone_lr": 6.784e-6,
        "t0_epochs": 3,
        "t_mult": 2,
        "lr_decay": 0.6101,
        "belnap_weight": 1.2546,
        "margin_weight": 0.618,
        "ltn_weight": 0.1,
        "assertion_weight": 0.1,
        "decisiveness_weight": 0.01,
        "decisiveness_warmup_epochs": 2,
        "logit_scale": 8.0,
    },
    # focused yoda_focused trial #8 (conflation 0.597->0.25, residual 0.6356->0.25)
    {
        "num_reasoning_blocks": 1,
        "n_heads": 2,
        "d_hidden_multiplier": 1.4,
        "num_q_probes": 10,
        "num_c_probes": 14,
        "num_k_probes": 5,
        "dropout": 0.2857,
        "conflation_weight": 0.25,
        "residual_weight": 0.25,
        "lr": 3.0e-4,
        "backbone_lr": 6.0e-6,
        "t0_epochs": 3,
        "t_mult": 2,
        "lr_decay": 0.60,
        "belnap_weight": 1.25,
        "margin_weight": 0.618,
        "ltn_weight": 0.1,
        "assertion_weight": 0.1,
        "decisiveness_weight": 0.01,
        "decisiveness_warmup_epochs": 2,
        "logit_scale": 8.0,
    },
]

PHASE_PRESETS: dict[str, dict[str, Any]] = {
    "sanity": {"n_trials": 5, "epochs": 8, "max_samples": 16000, "eval_split": "optuna_eval"},
    "broad": {"n_trials": 300, "epochs": 8, "max_samples": 16000, "eval_split": "optuna_eval"},
    "focused": {"n_trials": 150, "epochs": 8, "max_samples": 16000, "eval_split": "optuna_eval"},
    "validate": {"n_trials": None, "epochs": 12, "max_samples": None, "eval_split": "test"},
}


def resolve_storage(storage_arg: str | None) -> str:
    """Resolves study storage: explicit flag > YODA_OPTUNA_STORAGE env > local sqlite.

    The shared parser's --storage default is None (its default_storage parameter is
    help-text only), so the resolution happens here. No credentials live in the repo.
    """
    return storage_arg or os.environ.get("YODA_OPTUNA_STORAGE") or DEFAULT_STORAGE


def load_split_metadata(path: Path = SPLIT_METADATA_PATH) -> dict[str, Any]:
    """Loads v2 split metadata; returns an empty dict when the split is missing."""
    if not path.exists():
        logger.warning(
            "Split metadata not found at {}; run experiments/prepare_hpo_split.py first", path
        )
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def suggest_broad_params(trial: optuna.Trial) -> dict[str, Any]:
    """Samples the full Phase-A search space (M4-hardened + previously unscanned)."""
    return {
        # Topology (re-opened: the old locks were artifacts of the collapse objective)
        "num_reasoning_blocks": trial.suggest_int("num_reasoning_blocks", 1, 3),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "d_hidden_multiplier": trial.suggest_float("d_hidden_multiplier", 0.8, 4.0),
        "num_q_probes": trial.suggest_int("num_q_probes", 2, 16),
        "num_c_probes": trial.suggest_int("num_c_probes", 8, 32),
        "num_k_probes": trial.suggest_int("num_k_probes", 4, 16),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        # Hardened architecture bounds (R5)
        "conflation_weight": trial.suggest_float("conflation_weight", 0.01, 0.25, log=True),
        "residual_weight": trial.suggest_float("residual_weight", 0.10, 0.25),
        # Optimization (M4-hardened; min_lr wired to lr/100 downstream)
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "backbone_lr": trial.suggest_float("backbone_lr", 1e-6, 1e-4, log=True),
        "t0_epochs": trial.suggest_int("t0_epochs", 1, 5),
        "t_mult": trial.suggest_int("t_mult", 1, 3),
        "lr_decay": trial.suggest_float("lr_decay", 0.5, 1.0),
        # Loss weights (ltn/assertion never scanned before; decisiveness new in M4)
        "belnap_weight": trial.suggest_float("belnap_weight", 0.5, 2.0),
        "margin_weight": trial.suggest_float("margin_weight", 0.1, 1.0),
        "ltn_weight": trial.suggest_float("ltn_weight", 0.01, 1.0, log=True),
        "assertion_weight": trial.suggest_float("assertion_weight", 0.0, 0.5),
        "decisiveness_weight": trial.suggest_float("decisiveness_weight", 0.001, 0.05, log=True),
        "decisiveness_warmup_epochs": trial.suggest_int("decisiveness_warmup_epochs", 1, 3),
        # New head temperature (exposed via --logit-scale)
        "logit_scale": trial.suggest_float("logit_scale", 2.0, 32.0, log=True),
    }


def suggest_focused_params(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Narrowed 2-D probes around a Phase-B best-params point; all else locked."""
    params: dict[str, Any] = dict(base)
    params["decisiveness_weight"] = trial.suggest_float(
        "decisiveness_weight",
        max(0.001, base["decisiveness_weight"] / 4.0),
        min(0.05, base["decisiveness_weight"] * 4.0),
        log=True,
    )
    params["conflation_weight"] = trial.suggest_float(
        "conflation_weight",
        max(0.01, base["conflation_weight"] / 4.0),
        min(0.25, base["conflation_weight"] * 4.0),
        log=True,
    )
    params["logit_scale"] = trial.suggest_float(
        "logit_scale",
        max(2.0, base["logit_scale"] / 4.0),
        min(32.0, base["logit_scale"] * 4.0),
        log=True,
    )
    params["margin_weight"] = trial.suggest_float(
        "margin_weight",
        max(0.1, base["margin_weight"] - 0.2),
        min(1.0, base["margin_weight"] + 0.2),
    )
    params["num_q_probes"] = trial.suggest_int(
        "num_q_probes",
        max(2, int(base["num_q_probes"]) - 2),
        min(16, int(base["num_q_probes"]) + 2),
    )
    return params


def runtime_params(args: argparse.Namespace) -> dict[str, Any]:
    """Explicit runtime/paths — never CLI defaults (several exceed the realigned caps)."""
    return {
        "train_path": args.train_path,
        "eval_path": args.eval_path,
        "test_path": args.test_path,
        "max_train_samples": args.max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "collapse_threshold": args.collapse_threshold,
        "collapse_patience_epochs": args.collapse_patience_epochs,
    }


def _load_best_params(args: argparse.Namespace) -> dict[str, Any]:
    if not args.best_params_json:
        raise SystemExit("--best-params-json is required for the focused phase")
    with open(args.best_params_json, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("params", payload)


def _load_champions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int]]:
    if not args.champion_json:
        raise SystemExit("--champion-json is required for the validate phase")
    with open(args.champion_json, encoding="utf-8") as f:
        payload = json.load(f)
    configs = payload.get("configs") or [payload]
    seeds = payload.get("seeds", VALIDATE_SEEDS)
    return configs, seeds


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective: macro-averaged per-source eval accuracy, mean of last 2 epochs."""
    if args.phase == "focused":
        params = suggest_focused_params(trial, _load_best_params(args))
    else:
        params = suggest_broad_params(trial)
    params.update(runtime_params(args))
    # eta_min inversion fix: the cosine floor must track the peak LR.
    params["min_lr"] = params["lr"] / 100.0
    params["text_model_name"] = FIXED_TEXT_MODEL_NAME

    trial_dir = Path("models/optuna_v2") / f"trial_{trial.number}"
    metric_agg = "best" if args.phase == "validate" else METRIC_AGG
    return run_training_trial(
        trial, trial_dir, params, metric_name=METRIC_NAME, metric_agg=metric_agg
    )


def main() -> None:
    """Creates the phase study and runs the optimization sweep."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--phase", default="broad")
    known, _ = pre.parse_known_args()
    phase = known.phase
    if phase not in PHASE_PRESETS:
        raise SystemExit(f"Unknown phase {phase!r}; expected one of {sorted(PHASE_PRESETS)}")
    preset = PHASE_PRESETS[phase]

    metadata = load_split_metadata()
    split_dir = SPLIT_METADATA_PATH.parent
    parser = build_base_study_parser(
        description=f"Post-hardening Yoda study (phase: {phase}).",
        default_n_trials=preset["n_trials"] or 150,
        default_max_samples=preset["max_samples"] if preset["max_samples"] is not None else 16000,
        default_epochs=preset["epochs"],
        default_batch_size=8,
        default_early_stopping_patience=2,
        default_study_patience=40 if phase == "broad" else 0,
        default_storage=resolve_storage(None),
    )
    parser.add_argument("--phase", default=phase, choices=sorted(PHASE_PRESETS))
    parser.add_argument(
        "--best-params-json",
        type=str,
        default=None,
        help="Phase-B best-params JSON (focused phase)",
    )
    parser.add_argument(
        "--champion-json",
        type=str,
        default=None,
        help="Top-configs JSON with fixed seeds (validate phase)",
    )
    parser.set_defaults(
        train_path=str(split_dir / "train.parquet"),
        eval_path=str(split_dir / f"{preset['eval_split']}.parquet"),
        test_path=str(split_dir / "test.parquet"),
        collapse_threshold=metadata.get("collapse_threshold", 0.370),
    )
    args = parser.parse_args()
    args.phase = phase

    # Fail fast on missing data: every trial would prune otherwise. The v2 splits
    # are gitignored (derived data), so a fresh training host must either receive
    # them out-of-band or regenerate them with prepare_hpo_split.py.
    for label, flag, path in (
        ("train", "--train-path", args.train_path),
        ("eval", "--eval-path", args.eval_path),
    ):
        if path is None or not Path(path).exists():
            raise SystemExit(
                f"{label} dataset not found: {path!r}. Run "
                "`uv run python experiments/prepare_hpo_split.py` first, or pass "
                f"an explicit {flag}."
            )
    if phase == "validate" and (args.test_path is None or not Path(args.test_path).exists()):
        raise SystemExit(
            f"test dataset not found: {args.test_path!r}. The validate phase evaluates "
            "on the locked test split; run "
            "`uv run python experiments/prepare_hpo_split.py` first."
        )

    study_name = args.study_name or f"yoda_hardened_v2_{phase}"
    study, _ = init_study_and_metadata(
        study_name=study_name,
        storage=resolve_storage(args.storage),
        seed=args.seed,
        extra_metadata={
            "phase": phase,
            "n_trials": args.n_trials,
            "max_train_samples": args.max_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "metric": METRIC_NAME,
            "metric_agg": "best" if phase == "validate" else METRIC_AGG,
            "eval_path": args.eval_path,
            "test_path": args.test_path,
            "collapse_threshold": args.collapse_threshold,
            "split_metadata": str(SPLIT_METADATA_PATH),
        },
    )

    if phase == "sanity":
        for _ in range(preset["n_trials"]):
            study.enqueue_trial(dict(ANCHOR_PARAMS))
    elif phase == "broad":
        for champion in CLIPPED_CHAMPIONS:
            study.enqueue_trial(dict(champion))
        study.enqueue_trial(dict(ANCHOR_PARAMS))
    elif phase == "focused":
        study.enqueue_trial(dict(_load_best_params(args)))
    elif phase == "validate":
        configs, seeds = _load_champions(args)
        for config in configs:
            for seed in seeds:
                point = dict(config)
                point["seed"] = seed
                study.enqueue_trial(point)
        args.n_trials = len(configs) * len(seeds)

    callbacks = []
    if args.study_patience and args.study_patience > 0:
        callbacks.append(StudyEarlyStoppingCallback(patience=args.study_patience))

    study.optimize(
        lambda t: objective(t, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=callbacks,
    )

    try:
        logger.info("Optuna study completed.")
        logger.info("Best trial: #{}", study.best_trial.number)
        logger.info("Best {}: {:.4f}", METRIC_NAME, study.best_value)
        logger.info("Best parameters: {}", study.best_params)
    except ValueError:
        logger.warning("No trials completed successfully.")


if __name__ == "__main__":
    main()
