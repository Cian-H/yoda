# ADR 0008: Modular Submodule Realignment (NeSy, Data, and XAI)

## Status
Accepted

## Date
2026-10-02

## Context
As the Yoda architecture expanded through rapid iterations (implementing Belnap Bilattice transformers, epistemic multihead pooling, DLA diagnostic probes, and Logic Tensor Network constraints), certain modules were provisionally located in interim subpackages:
1. `ltn.py` (continuous differentiable logic constraints) was located in `yoda.probabilistic`. Per `agents.md`, `yoda.nesy` is the designated domain for neuro-symbolic integration and rule constraints, whereas `yoda.probabilistic` handles uncertainty quantification and bilattice algebras.
2. `dataset.py` (PyTorch dataset and collation utilities) resided under `yoda.training`, whereas data ingestion, normalization, and loaders belong under `yoda.data`.
3. Direct Logit Attribution (DLA) diagnostic routines were written as ad-hoc helpers inside `experiments/train_yoda.py`, making them unavailable for programmatic inspection, auditing, or unit testing under `yoda.xai`.
4. Legacy `from __future__ import annotations` headers were present across 22 files despite Python 3.14 targeting native postponed evaluation of annotations.

## Decision
1. **Move `ltn.py` to `yoda.nesy`**: Relocated `src/yoda/probabilistic/ltn.py` to `src/yoda/nesy/ltn.py`. Exported in `yoda.nesy` with backwards-compatible re-export in `yoda.probabilistic`.
2. **Move `dataset.py` to `yoda.data`**: Relocated `src/yoda/training/dataset.py` to `src/yoda/data/dataset.py`. Exported in `yoda.data` with backwards-compatible re-export in `yoda.training`.
3. **Extract DLA to `yoda.xai.dla`**: Created `DirectLogitAttribution` class and `run_dla_evaluation` entrypoint in `src/yoda/xai/dla.py`, exported in `yoda.xai`, and covered with unit tests in `tests/test_xai_dla.py`.
4. **Remove `from __future__ import annotations`**: Stripped future annotations import across all source, test, and experiment files in favor of Python 3.14 idioms.

## Consequences
- **Positive**: Strict alignment with `agents.md` package layout (`architecture`, `nesy`, `probabilistic`, `xai`, `data`).
- **Positive**: DLA diagnostics can now be utilized across any benchmark or evaluation script without duplicating code.
- **Positive**: Backward compatibility preserved across package boundaries via re-exports.
- **Negative**: Required updating import statements across internal modules and tests.
