# Handoff Report: Explorer M1_3 (Pre-Activation Normalization & Residual Gradient Highway)

## 1. Observation

Direct observations from the Yoda codebase and architecture documentation:

### 1.1 Unnormalized Pre-Sigmoid Activations
- **`src/yoda/architecture/belnap_transformer.py:110-117`**:
  `BelnapAttention` instantiates output projections `self.w_out_pos = nn.Linear(d_model, d_model, bias=bias)` and `self.w_out_neg = nn.Linear(d_model, d_model, bias=bias)`.
- **`src/yoda/architecture/belnap_transformer.py:201-204`**:
  ```python
  # Output bounds in [0, 1]
  out_pos = torch.sigmoid(self.w_out_pos(agg_pos))
  out_neg = torch.sigmoid(self.w_out_neg(agg_neg))
  ```
  `agg_pos` and `agg_neg` are passed directly through linear projections into `torch.sigmoid` without intermediate normalization.
- **`src/yoda/architecture/belnap_transformer.py:234-235` & `247-249`**:
  ```python
  self.w_o1 = nn.Linear(d_hidden, d_model)
  self.w_o2 = nn.Linear(d_hidden, d_model)
  # ...
  e_ffn_pos = torch.sigmoid(self.w_o1(h_mid))
  e_ffn_neg = torch.sigmoid(self.w_o2(h_mid))
  ```
  `h_mid` is generated via unbounded Mish activation (`nn.functional.mish(...)`), then mapped to `d_model` and directly squashed with `torch.sigmoid`.

### 1.2 Residual Interpolation Weight Default ($\alpha = 0.5$)
- **`src/yoda/architecture/belnap_transformer.py:262`**:
  ```python
  def __init__(..., residual_weight: float = 0.5, ...):
  ```
- **`src/yoda/architecture/belnap_transformer.py:289-293`**:
  ```python
  @staticmethod
  def _residual_join(
      base: BelnapState,
      delta: BelnapState,
      alpha: float = 0.5,
  ) -> BelnapState:
  ```
- **`src/yoda/architecture/belnap_transformer.py:334`**:
  ```python
  class BelnapDecisionTransformer(nn.Module):
      def __init__(..., residual_weight: float = 0.5, ...):
  ```
- **Mathematical Attenuation**:
  For each join, $\frac{\partial\,\text{res}}{\partial\,\text{base}} = 1 - \alpha = 0.5$. Across 6 cascaded joins (2 reasoning blocks $\times$ 3 joins per block), identity skip gradient retention is $(0.5)^6 = 0.015625 \approx 1.56\%$.

### 1.3 Fixed Residual Weight in Narrow Optuna Study
- **`experiments/optuna_study_narrow.py:33`**:
  ```python
  FIXED_RESIDUAL_WEIGHT = 0.6356
  ```
- **`experiments/optuna_study_narrow.py:67, 130`**:
  `params["residual_weight"] = FIXED_RESIDUAL_WEIGHT` is injected into trial training.
- **`tests/test_optuna_study_narrow.py:31`**:
  ```python
  assert FIXED_RESIDUAL_WEIGHT == 0.6356
  ```
  Tests directly assert that `FIXED_RESIDUAL_WEIGHT` matches `0.6356`.

### 1.4 Downstream Engine & CLI Defaults
- **`src/yoda/architecture/engine.py:165`**:
  `YodaDecisionEngine.__init__` has `residual_weight: float = 0.5`.
- **`experiments/train_yoda.py:230-234`**:
  CLI parser flag `--residual-weight` defaults to `0.5`.
- **`tests/test_training.py:1178`**:
  `assert default_args.residual_weight == 0.5`.

---

## 2. Logic Chain

1. **Mechanism of Gradient Collapse (Observation 1.1 & 1.2)**:
   The sigmoid derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ has a global maximum at $\sigma'(0) = 0.25$. When activations drift ($|z| \ge 4.0$), $\sigma'(z) \le 0.018$. Compounding unnormalized sigmoid operations across 2 blocks yields $(\le 0.25)^5 \le 0.00097$. Concurrently, the residual skip highway with $\alpha = 0.5$ attenuates gradients by $(1 - 0.5)^6 = 1.56\%$. Multiplying these factors extinguishes upstream backpropagation, causing the 36.5% Epoch 1 gradient death documented in `.docs/plan.md`.

2. **Pre-Activation LayerNorm Stabilization**:
   Inserting `nn.LayerNorm(self.d_model)` immediately before `torch.sigmoid(self.w_out_pos(...))` and `torch.sigmoid(self.w_out_neg(...))` in `BelnapAttention`, and before `torch.sigmoid(self.w_o1(...))` and `torch.sigmoid(self.w_o2(...))` in `BelnapFFN`, standardizes the pre-activation channel dimensions to $\mu = 0, \sigma^2 = 1$. This centers the inputs to sigmoid in the linear slope regime ($\sigma'(0) = 0.25$), preventing premature saturation into flat asymptotic tails ($0$ or $1$) on Step 1.

3. **Preserving Initial Bias Suppression in Attention**:
   `LayerNorm` subtracts the mean across the feature dimension, which cancels out uniform channel biases like `w_out_pos.bias = -3.0` (used for Neither suppression in `test_zero_knowledge_suppression`). By initializing `self.norm_pos.bias` and `self.norm_neg.bias` to `init_bias` when `init_bias != 0.0`, the normalized output retains the desired negative shift $\beta = \text{init\_bias}$, ensuring compatibility with `test_zero_knowledge_suppression` while defaulting to 0-centering when `init_bias = 0.0`.

4. **Residual Highway Recalibration to $\alpha = 0.15$**:
   Lowering default `residual_weight` and `alpha` from $0.5$ to $0.15$ transforms the skip gradient retention across 6 cascaded joins to:
   $$\text{Retention} = (1 - 0.15)^6 = (0.85)^6 = 0.37715 \approx 37.7\%$$
   This exceeds the $>35\%$ threshold required by ADR-0007 and R2, providing a $>24\times$ improvement in gradient survival over $\alpha = 0.5$.

5. **Hyperopt Narrow Sweep Alignment**:
   `experiments/optuna_study_narrow.py` previously locked `FIXED_RESIDUAL_WEIGHT = 0.6356` from early empirical runs before the architectural gradient highway diagnosis. At $\alpha = 0.6356$, skip gradient retention across 6 joins is $(1 - 0.6356)^6 = (0.3644)^6 \approx 0.0022$ ($0.22\%$), causing severe gradient extinction. Recalibrating `FIXED_RESIDUAL_WEIGHT = 0.15` in `experiments/optuna_study_narrow.py` and updating the assertion in `tests/test_optuna_study_narrow.py:31` brings the narrow sweep into full alignment with the hardened architecture.

6. **Test Verification Architecture**:
   `tests/test_prenorm_highway.py` supplies dedicated unit and property tests verifying:
   - LayerNorm instantiation on positive and negative evidence channels
   - Pre-sigmoid standardized normal distribution ($\mu \approx 0, \sigma \approx 1$)
   - Anti-saturation and finite gradient norms under extreme magnitude inputs
   - Single-join gradient identity $\frac{\partial\,\text{res}}{\partial\,\text{base}} = 0.85$
   - 6-join gradient retention $> 35\%$ ($37.7\%$)
   - Consistency of `FIXED_RESIDUAL_WEIGHT == 0.15`

---

## 3. Caveats

1. **`YodaDecisionEngine` Default Residual Weight**:
   `src/yoda/architecture/engine.py:165` defines default `residual_weight: float = 0.5`. If an engine instance is initialized without explicit `residual_weight`, it forwards `0.5` to its inner `BelnapTransformerBlock` instances. We recommend updating `YodaDecisionEngine.__init__` default `residual_weight` to `0.15` as part of Phase 1 to ensure system-level consistency.
2. **`train_yoda.py` CLI Argument Default**:
   `experiments/train_yoda.py:230` defaults `--residual-weight` to `0.5`. `tests/test_training.py:1178` asserts `assert default_args.residual_weight == 0.5`. If `train_yoda.py` is updated, `test_training.py` must be updated concurrently.
3. **`optuna_study_focused.py` Scope**:
   `experiments/optuna_study_focused.py:37` also has `FIXED_RESIDUAL_WEIGHT = 0.6356`, asserted in `tests/test_optuna_study_focused.py:36`. While the dispatch specifically targeted `optuna_study_narrow.py`, `optuna_study_focused.py` should be realigned during Phase 4.

---

## 4. Conclusion

The implementation strategy for Requirement R2 (Pre-Activation Normalization & Residual Gradient Highway) is complete and fully specified.

### 4.1 Changes to `src/yoda/architecture/belnap_transformer.py`
1. **`BelnapAttention`**:
   - Add `self.norm_pos = nn.LayerNorm(d_model)` and `self.norm_neg = nn.LayerNorm(d_model)` in `__init__`.
   - Initialize `self.norm_pos.bias` and `self.norm_neg.bias` to `init_bias`.
   - Update forward pass:
     ```python
     out_pos = torch.sigmoid(self.norm_pos(self.w_out_pos(agg_pos)))
     out_neg = torch.sigmoid(self.norm_neg(self.w_out_neg(agg_neg)))
     ```
2. **`BelnapFFN`**:
   - Store `self.d_model = d_model` in `__init__`.
   - Add `self.norm_pos = nn.LayerNorm(d_model)` and `self.norm_neg = nn.LayerNorm(d_model)`.
   - Update forward pass:
     ```python
     e_ffn_pos = torch.sigmoid(self.norm_pos(self.w_o1(h_mid)))
     e_ffn_neg = torch.sigmoid(self.norm_neg(self.w_o2(h_mid)))
     ```
3. **`BelnapTransformerBlock` & `_residual_join`**:
   - Change default `residual_weight: float = 0.15` in `BelnapTransformerBlock.__init__`.
   - Change default `alpha: float = 0.15` in `_residual_join`.
4. **`BelnapDecisionTransformer`**:
   - Change default `residual_weight: float = 0.15` in `BelnapDecisionTransformer.__init__`.

### 4.2 Changes to `experiments/optuna_study_narrow.py`
- Change `FIXED_RESIDUAL_WEIGHT = 0.6356` to `FIXED_RESIDUAL_WEIGHT = 0.15`.

### 4.3 Changes to `tests/test_optuna_study_narrow.py`
- Line 31: Update assertion to `assert FIXED_RESIDUAL_WEIGHT == 0.15`.

### 4.4 New Test Suite: `tests/test_prenorm_highway.py`
- Complete implementation provided in `.agents/teamwork/explorer_m1_3/proposed_test_prenorm_highway.py`.

### 4.5 Artifact Index in Working Directory
- `.agents/teamwork/explorer_m1_3/proposed_belnap_transformer.patch`: Ready-to-apply patch for `belnap_transformer.py`.
- `.agents/teamwork/explorer_m1_3/proposed_optuna_study_narrow.patch`: Ready-to-apply patch for `optuna_study_narrow.py` and `test_optuna_study_narrow.py`.
- `.agents/teamwork/explorer_m1_3/proposed_test_prenorm_highway.py`: Complete test implementation.

---

## 5. Verification Method

To independently verify the implementation once applied:

1. **Targeted Test Execution**:
   ```bash
   uv run pytest tests/test_prenorm_highway.py tests/test_belnap_transformer.py tests/test_optuna_study_narrow.py -v
   ```
   Must pass 100% of tests.
2. **Full Regression Suite**:
   ```bash
   uv run pytest
   ```
   All 181 existing tests + new unit tests must pass without regressions.
3. **Lint & Style Validation**:
   ```bash
   uv run ruff check .
   ```
   Must exit with code 0 ("All checks passed!").
4. **Invalidation Conditions**:
   - Skip gradient retention across 6 joins $\le 0.35$.
   - Pre-sigmoid pre-activations std $\ne 1.0 \pm 0.01$.
   - Any failure in `test_zero_knowledge_suppression` or `test_fixed_parameters_consistency`.
