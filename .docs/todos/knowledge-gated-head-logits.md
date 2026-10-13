# Synchronize Decision Head Logits with Epistemic Knowledge Coordinates

- **Area:** Decision Head Architecture & Loss Calibration (ADR-0004, ADR-0012)
- **Refs:** `../plan.md`, `../adrs/0004-belnap-bilattice-native-transformer.md`, `../adrs/0012-focal-margin-loss-and-onecycle-lr-scheduling.md`, `../../src/yoda/architecture/engine.py`, `../../src/yoda/architecture/belnap_transformer.py`, `../../src/yoda/training/losses.py`

## Context

Currently, two diverging definitions of classification logits exist across the repository. In `src/yoda/architecture/belnap_transformer.py`, `BelnapDecisionTransformer.forward` gates logits by epistemic knowledge coordinates: $\text{logits} = (choice\_pos - choice\_neg) \times knowledge$. In contrast, `src/yoda/architecture/engine.py` defines `BelnapDecisionHead.forward` using raw pre-sigmoid differences: $\text{logits} = raw\_pos - raw\_neg$. Because `FocalMarginLoss` (`src/yoda/training/losses.py`) operates on these raw differences, a candidate with $raw\_pos = -8.0$ and $raw\_neg = -12.0$ has virtually zero knowledge ($k \approx 0.0001$, pure $Neither$), yet its logit is $(-8.0) - (-12.0) = +4.0$. This produces artificially high Softmax confidence in `FocalLoss` despite complete lack of epistemic support. Gating head logits by knowledge, or introducing an epistemic penalty on high logits when $k \to 0$, tightly couples model confidence to actual evidence grounding.

## Deferred because

Harmonizing logit definitions impacts both classification head tests in `tests/test_engine.py` and loss evaluation tests in `tests/test_losses.py`. This work is staged into Phase 2 of the roadmap.

## Revisit when

Next commit modifying `src/yoda/architecture/engine.py` or `src/yoda/training/losses.py`, or when evaluating calibrated uncertainty metrics.
