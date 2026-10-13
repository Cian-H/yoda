# Replace Sigmoid Jacobian Minimization with Contradiction Penalty and Warmup

- **Area:** Neuro-Symbolic Logic & Loss Constraints (ADR-0001, ADR-0004, ADR-0012)
- **Refs:** `../plan.md`, `../adrs/0001-architecture-foundations.md`, `../adrs/0012-focal-margin-loss-and-onecycle-lr-scheduling.md`, `../../src/yoda/nesy/ltn.py`, `../../tests/test_ltn.py`

## Context

In `src/yoda/nesy/ltn.py`, `LTNConstraintLoss.forward` evaluates a decisiveness penalty defined as `decisiveness_terms = choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)`. Because `choice_pos = torch.sigmoid(raw_pos)`, the mathematical derivative (Jacobian) of the sigmoid activation is identically $\sigma(z)(1 - \sigma(z)) = \text{choice\_pos} \cdot (1 - \text{choice\_pos})$. Minimizing this term is literally penalizing the Jacobian of the decision head's activation function. At initialization ($\text{raw\_pos} \approx 0 \implies \text{choice\_pos} \approx 0.5$), the loss resides at its global maximum ($0.25$), incentivizing the optimizer on Step 1 to rapidly blow linear biases to $\pm 5.0$ to force outputs to $0$ or $1$. The moment $p(1 - p) \to 0$, upstream gradients through `raw_pos` and `raw_neg` collapse to zero, permanently cutting off all NeSy constraints (XOR, OR, hierarchy) from early training and contributing to the 36.5% accuracy plateau. Replacing this penalty with a true paraconsistent contradiction penalty ($Both$: $\text{choice\_pos} \times \text{choice\_neg}$) and introducing a linear warmup schedule resolves this failure mode.

## Deferred because

The immediate work is establishing the comprehensive plan and documenting all architectural feedback. Modifying `LTNConstraintLoss` requires updating constraint tests in `tests/test_ltn.py` and adjusting sweep hyperparameters in `src/yoda/hyperopt/optuna.py`.

## Revisit when

Next commit modifying `src/yoda/nesy/ltn.py` or preceding any training sweep on multi-choice logic benchmarks.
