"""Analyzes a v2 Optuna study: top trials, fANOVA importances, Phase-B/C/D exports.

Usage:
    uv run python experiments/analyze_study.py --study-name yoda_hardened_v2_broad
    uv run python experiments/analyze_study.py --storage sqlite:///data/optuna_v2.db

Prints the top-K trials and fANOVA parameter importances (requires scikit-learn),
then writes (unless --no-write):
    <out-dir>/best_params.json   best trial params + importances + sub-threshold locks
                                (input to optuna_study_v2.py --phase focused)
    <out-dir>/top_configs.json   top-3 distinct configs x fixed seeds
                                (input to optuna_study_v2.py --phase validate)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import optuna
from loguru import logger

DEFAULT_STORAGE = "sqlite:///data/optuna_v2.db"
DEFAULT_OUT_DIR = Path("data/processed/v2")
VALIDATE_SEEDS = [11, 22, 33, 44, 55]


def main() -> None:
    """Loads a study, prints the analysis, and writes Phase-B/C/D export files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--storage",
        default=os.environ.get("YODA_OPTUNA_STORAGE") or DEFAULT_STORAGE,
        help="Optuna storage URL (default: YODA_OPTUNA_STORAGE or local sqlite)",
    )
    parser.add_argument("--study-name", default=None, help="Study name (default: most recent)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--importance-threshold", type=float, default=0.05)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-write", action="store_true", help="Print only; write nothing")
    args = parser.parse_args()

    names = optuna.study.get_all_study_names(storage=args.storage)
    if args.study_name:
        if args.study_name not in names:
            available = ", ".join(names) if names else "(none)"
            raise SystemExit(
                f"Study {args.study_name!r} not found in storage {args.storage!r}. "
                f"Available studies: {available}. Check --storage / YODA_OPTUNA_STORAGE."
            )
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    else:
        if not names:
            raise SystemExit(f"No studies found in storage {args.storage}")
        study = optuna.load_study(study_name=names[-1], storage=args.storage)
        logger.info("Analyzing most recent study: {}", names[-1])

    trials = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not trials:
        raise SystemExit(f"Study {study.study_name!r} has no complete trials")
    trials.sort(key=lambda t: t.value, reverse=True)

    print(f"Study {study.study_name!r}: {len(trials)} complete trials")
    print(f"{'rank':>4} {'trial':>6} {'value':>8}  params")
    for rank, t in enumerate(trials[: args.top_k], start=1):
        params_preview = json.dumps(t.params, sort_keys=True, default=str)
        print(f"{rank:>4} {t.number:>6} {t.value:>8.4f}  {params_preview}")

    best = trials[0]
    importances: dict[str, float] = {}
    try:
        from optuna.importance import FanovaImportanceEvaluator

        importances = FanovaImportanceEvaluator(seed=42)(study)
    except ImportError as exc:
        logger.warning("scikit-learn not installed; skipping fANOVA importances ({})", exc)
    except Exception as exc:
        logger.warning("fANOVA importance failed ({}); continuing without it", exc)
    if importances:
        print("\nfANOVA parameter importances:")
        for param, value in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"  {param:28s} {value:.4f}")

    locks = {
        param: best.params[param]
        for param, value in importances.items()
        if value < args.importance_threshold and param in best.params
    }
    if locks:
        print(f"\nSub-threshold locks (<{args.importance_threshold}) at best-trial values:")
        print(json.dumps(locks, indent=2, default=str))

    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in trials:
        key = json.dumps(t.params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        configs.append(dict(t.params))
        if len(configs) >= 3:
            break

    if not args.no_write:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_payload = {
            "study": study.study_name,
            "trial": best.number,
            "value": best.value,
            "params": best.params,
            "importances": importances,
            "locked": locks,
        }
        best_path = out_dir / "best_params.json"
        best_path.write_text(
            json.dumps(best_payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        top_payload = {"study": study.study_name, "configs": configs, "seeds": VALIDATE_SEEDS}
        top_path = out_dir / "top_configs.json"
        top_path.write_text(json.dumps(top_payload, indent=2, default=str) + "\n", encoding="utf-8")
        logger.info("Wrote {} and {}", best_path, top_path)


if __name__ == "__main__":
    main()
