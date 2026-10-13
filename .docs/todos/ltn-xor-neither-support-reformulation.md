# Reformulate LTN XOR and OR Constraints on Positive Evidence Support

- **Area:** Neuro-Symbolic Logic & Bilattice Constraints (ADR-0001, ADR-0004)
- **Refs:** `../plan.md`, `../adrs/0001-architecture-foundations.md`, `../adrs/0004-belnap-bilattice-native-transformer.md`, `../../src/yoda/nesy/ltn.py`, `../../tests/test_ltn.py`

## Context

In `src/yoda/nesy/ltn.py`, `LTNConstraintLoss._compute_xor` enforces single-choice exclusivity by maximizing $\sum_i t_i \prod_{j \neq i} (1 - t_j)$, where truth is computed from Bilattice coordinates as $t = \frac{e_{\text{pos}} - e_{\text{neg}} + 1}{2}$. For an unchosen candidate $j \neq y$, minimizing the loss requires driving $t_j \to 0$. In Belnap coordinates, $t_j = 0$ requires explicit refutation ($e_j^+ = 0, e_j^- = 1$), corresponding to non-zero knowledge ($k_j = 0.5$). For candidates that are completely unmentioned in the state ($Neither$: $e_j^+ = 0, e_j^- = 0$), truth is naturally $t_j = \frac{0 - 0 + 1}{2} = 0.5$. Consequently, evaluating XOR on truth penalizes every unmentioned $Neither$ candidate by $-\log(1 - 0.5) = 0.693$, actively punishing the network for expressing ignorance and forcing it to fabricate false negative evidence ($e_j^- \to 1$). Reformulating `_compute_xor` and `_compute_or` on positive support ($\text{choice\_pos}$, $e^+$) or logit probabilities allows unchosen options to rest peacefully at either $False$ $(0, 1)$ or $Neither$ $(0, 0)$ without penalty.

## Deferred because

Refining the truth coordinate formulation in `LTNConstraintLoss` requires validating whether downstream evaluations and loss logging rely on the existing truth interface, as well as updating property-based tests in `tests/test_ltn.py`. Scheduled for Phase 2 implementation.

## Revisit when

Next commit modifying `src/yoda/nesy/ltn.py` or when benchmarking uncertainty calibration on unmentioned options.
