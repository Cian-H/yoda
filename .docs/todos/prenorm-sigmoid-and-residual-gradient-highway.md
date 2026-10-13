# Center Pre-Sigmoid Activations and Recalibrate Residual Interpolation Weight

- **Area:** Transformer Core & Gradient Flow (ADR-0004, ADR-0006, ADR-0007)
- **Refs:** `../plan.md`, `../adrs/0004-belnap-bilattice-native-transformer.md`, `../adrs/0007-convex-combination-residual-joins.md`, `../../src/yoda/architecture/belnap_transformer.py`, `../../tests/test_belnap_transformer.py`

## Context

Inside `src/yoda/architecture/belnap_transformer.py`, `BelnapAttention` and `BelnapFFN` apply `torch.sigmoid` directly to unnormalized linear projections (`w_out_pos`, `w_o1`), where activation variances drift across reasoning stages. Because the derivative of an unnormalized sigmoid is $\le 0.25$, chaining multiple operations without pre-normalization leads to compound attenuation $(\le 0.25)^5 \le 0.00097$ on early layers. Simultaneously, `BelnapTransformerBlock._residual_join` executes convex combination updates $x_{l+1} = (1 - \alpha) x_l + \alpha f(x_l)$. Across 2 reasoning blocks containing 6 total joins, a default residual weight of $\alpha = 0.5$ attenuates the skip connection highway by $(1 - 0.5)^6 = (0.5)^6 = \frac{1}{64} \approx 0.0156$, leaving only 1.5% of the identity gradient reaching the base representation. Inserting `nn.LayerNorm(d_model)` immediately prior to the bounding sigmoid centers pre-activations at zero with unit variance (guaranteeing $\sigma'(0) = 0.25$), and lowering $\alpha$ to $0.15 - 0.20$ retains $(1 - 0.15)^6 \approx 38\%$ of the residual gradient signal.

## Deferred because

Tuning $\alpha$ and adding pre-normalization LayerNorm modifies the core parameter set and state dictionary structure of `BelnapTransformerBlock`, requiring careful updating of serialization tests in `tests/test_belnap_transformer.py`. This work is staged into Phase 1 of the architecture plan.

## Revisit when

Next commit modifying `src/yoda/architecture/belnap_transformer.py` or before re-running trial sweeps in `experiments/run_optuna_study.py`.
