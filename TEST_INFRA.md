# Test Infrastructure & E2E Testing Architecture

## 1. Overview & Test Runner

The Yoda project test infrastructure is built on Python 3.14 and `pytest` (>= 8.0), managed via `uv`.
Configuration is centralized in `pyproject.toml`:
- **Pythonpath:** `src`, `.`
- **Test Discovery:** `tests/` (unit, integration, and property tests) and `tests/e2e/` (opaque-box end-to-end suites).
- **Linter & Style:** `ruff` (>= 0.4.0) with strict McCabe complexity (`<= 15`) and 100-character line limit.

### Runner Invocations
```bash
# Execute entire test suite (unit + regression + e2e)
uv run pytest

# Execute opaque-box E2E suites only
uv run pytest tests/e2e/

# Execute individual E2E tiers
uv run pytest tests/e2e/test_tier1_feature_coverage.py
uv run pytest tests/e2e/test_tier2_boundary_corner.py
uv run pytest tests/e2e/test_tier3_cross_feature.py

# Execute individual Tier 4 Acceptance Gates
uv run pytest tests/e2e/test_gradient_health_gate.py
uv run pytest tests/e2e/test_accuracy_ceiling_gate.py
uv run pytest tests/e2e/test_paraconsistent_gate.py
uv run pytest tests/e2e/test_throughput_gate.py

# Code style and lint validation
uv run ruff check src/ tests/
```

---

## 2. Test Architecture: Dual-Track Methodology

The test architecture implements the **Dual-Track Testing Methodology**, decoupling feature implementation milestones from opaque-box acceptance verification. Test specifications are derived from the authoritative requirements (`ORIGINAL_REQUEST.md`) and architectural blueprints (`PROJECT.md`, `.docs/plan.md`, `.docs/todos/`).

```
tests/
├── (181 Unit & Integration Tests)
└── e2e/
    ├── conftest.py                       # Shared test fixtures & generators
    ├── test_tier1_feature_coverage.py    # Tier 1: Feature Coverage (R1-R5, >=5 tests/feat)
    ├── test_tier2_boundary_corner.py     # Tier 2: Boundary & Corner Cases (R1-R5, >=5 tests/feat)
    ├── test_tier3_cross_feature.py       # Tier 3: Pairwise Cross-Feature Interactions
    ├── test_gradient_health_gate.py      # Tier 4 Gate: 10 seeds, Epoch 1 grad norms > 1e-4
    ├── test_accuracy_ceiling_gate.py     # Tier 4 Gate: Validation accuracy > 60.0%
    ├── test_paraconsistent_gate.py       # Tier 4 Gate: Neither k < 0.20 & logit suppression
    └── test_throughput_gate.py           # Tier 4 Gate: Independent eval speedup >= 2.0x
```

---

## 3. Test Suites & Tier Inventory

### Tier 1: Feature Coverage (`tests/e2e/test_tier1_feature_coverage.py`)
Provides opaque-box behavioral tests for all core hardening requirements (25 tests total, 5 per requirement):
- **R1. Attention Mask Propagation & Scale Normalization:**
  - `test_r1_text_encoder_forward_with_mask`: `TextEncoder` returns `(embeddings, attention_mask)`.
  - `test_r1_belnap_attention_masks_padding_keys`: `BelnapAttention` masks padding keys ($s \to -1\text{e}9$).
  - `test_r1_active_mass_scale_normalization`: Aggregation normalized by active mass denominator `mask.sum().clamp(min=1.0)`.
  - `test_r1_variable_sequence_length_boundedness`: Intermediate activations bounded in $[0, 1]$ independent of sequence length.
  - `test_r1_belnap_mpa_mask_propagation`: `BelnapMPA` propagates token masks into epistemic pooling.
- **R2. Paraconsistent Contradiction Loss & Pre-Activation Normalization:**
  - `test_r2_contradiction_loss_replaces_quadratic_penalty`: Contradiction penalty evaluates $e^+ \cdot e^-$ rather than $p(1-p)$.
  - `test_r2_linear_warmup_schedule`: Linear warmup schedule ramps `decisiveness_weight` from 0 to target.
  - `test_r2_prenorm_layernorm_in_attention`: Pre-sigmoid `LayerNorm` in `BelnapAttention` centers activations.
  - `test_r2_prenorm_layernorm_in_ffn`: Pre-sigmoid `LayerNorm` in `BelnapFFN` centers activations.
  - `test_r2_residual_highway_alpha_recalibration`: Residual weight $\alpha \le 0.20$ retains $>35\%$ gradient signal.
- **R3. NeSy Constraint Reformulation & Epistemic Logit Synchronization:**
  - `test_r3_positive_support_xor_zero_penalty_for_neither`: Positive support XOR guarantees zero loss for $Neither$ $(0, 0)$.
  - `test_r3_positive_support_or_zero_penalty_for_neither`: Positive support OR guarantees zero loss for $Neither$.
  - `test_r3_coupled_cross_polarity_value_routing`: $2 \times 2$ bilattice aggregation preserves refutation keys.
  - `test_r3_refutation_signal_preserved_under_coherent_query`: Key negative refutation is preserved under coherent query.
  - `test_r3_epistemic_logit_synchronization`: Decision head logits are gated by epistemic knowledge coordinate $k$.
- **R4. Bilinear Candidate-Query Alignment & Negation Equivariance:**
  - `test_r4_bilinear_belnap_inner_product_scoring`: Scalar decision head evaluates bilinear inner product.
  - `test_r4_elimination_of_additive_context_cancellation`: Candidate ranking difference depends on query context.
  - `test_r4_dual_channel_ffn_projections`: Dual-channel positive/negative projections in `BelnapFFN`.
  - `test_r4_dual_channel_cross_gating`: Cross-channel gating in `BelnapFFN` maintains representation capacity.
  - `test_r4_negation_equivariance_under_state_inversion`: $f(x.\text{negate}()) = f(x).\text{negate}()$.
- **R5. Context Encoding Deduplication & Hyperparameter Sweep Realignment:**
  - `test_r5_context_encoding_deduplication_cache`: Unique query/state pairs cached per group across candidates.
  - `test_r5_grouped_candidate_scoring_consistency`: Cached outputs match uncached numerical precision.
  - `test_r5_cache_isolation_across_distinct_groups`: Distinct question groups maintain isolated cache entries.
  - `test_r5_optuna_search_bounds_realignment`: Search spaces bounds conform to stable numerical regimes.
  - `test_r5_learning_rate_safe_regime`: Peak learning rate bound $\le 5\text{e-}4$.

### Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundary_corner.py`)
Stresses boundary and adversarial corner conditions (25 tests total, 5 per requirement):
- **R1 Boundaries:** All-padding sequences, single active token with 255 padding tokens, empty batch ($B=0$), extreme length scaling ($L=512$), zero active mass denominator clamping.
- **R2 Boundaries:** Pure $Both$ contradiction ($1, 1$), pure $Neither$ uncommitted ($0, 0$), warmup epoch 0 nullification, extreme pre-sigmoid activations ($\pm 100$), deep residual stack (12 joins).
- **R3 Boundaries:** Pure $Neither$ zero XOR penalty, pure refutation candidate ($0, 1$), pure truth candidate ($1, 0$), zero-knowledge logit suppression ($k \to 0$), maximal knowledge dynamic range ($k = 1$).
- **R4 Boundaries:** Orthogonal query-candidate vectors, anti-aligned query-candidate vectors, positive vs negative candidates, extreme coordinate negation equivariance, single-candidate group ($K=1$).
- **R5 Boundaries:** Single item batch cache idempotence, large group scaling ($K=16$), ragged candidate counts per group, identical query with different states, Optuna edge bounds.

### Tier 3: Pairwise Cross-Feature Interactions (`tests/e2e/test_tier3_cross_feature.py`)
Verifies end-to-end coupling across requirement pairs (10 tests total):
- R1 x R2: Masked variable-length sequences under contradiction loss and pre-sigmoid LayerNorm.
- R1 x R3: Variable-length masked padding interacting with 2x2 coupled bilattice routing.
- R1 x R4: Attention-masked sequence representations scored with bilinear candidate alignment.
- R1 x R5: Mask propagation with cached/deduplicated context encodings.
- R2 x R3: Contradiction loss with linear warmup trained jointly with positive-support XOR.
- R2 x R4: Pre-sigmoid normalized activations processed through dual-channel FFN.
- R2 x R5: Residual highway $\alpha=0.15$ gradient propagation through cached unrolled candidates.
- R3 x R4: Epistemic knowledge gating combined with bilinear candidate scoring under negation inversion.
- R3 x R5: Grouped unrolled candidate evaluation with positive support XOR and context caching.
- R4 x R5: Bilinear candidate scoring on cached context representations with dual-channel FFN.

### Tier 4: Real-World Acceptance Gates
Authoritative acceptance gates required for project completion (14 tests total):
1. **Gradient Health Gate (`tests/e2e/test_gradient_health_gate.py`):**
   - 10 random seeds (0 through 9).
   - Asserts average gradient norm across `self.reasoning_layers[0]` parameters $> 1\text{e-}4$ throughout Epoch 1 backpropagation.
2. **Accuracy Ceiling Breakthrough Gate (`tests/e2e/test_accuracy_ceiling_gate.py`):**
   - Multi-choice validation accuracy on independent choice evaluation benchmarks $> 60.0\%$, resolving the 50.0% context cancellation ceiling.
3. **Paraconsistent Calibration Gate (`tests/e2e/test_paraconsistent_gate.py`):**
   - Unmentioned candidates retain epistemic knowledge $k < 0.20$ ($Neither$) without incurring constraint loss penalties ($< 0.05$).
   - Low-knowledge states ($k \to 0$) suppress logits ($|\text{logits}| < 0.25$).
4. **Throughput Benchmark Gate (`tests/e2e/test_throughput_gate.py`):**
   - Candidate evaluation throughput speedup $\ge 2.0\times$ between cached and uncached unrolled evaluation.

---

## 4. Coverage Thresholds & Progression Matrix

| Suite / Gate | Test Count | Current Status (M0) | Target Milestone |
|---|---|---|---|
| Existing Unit Suite (`tests/`) | 181 | 181 Passed (100%) | Baseline (Must not regress) |
| Tier 1: Feature Coverage | 25 | 16 Passed, 9 Failed | M1 (R1, R2), M2 (R3), M3 (R4), M4 (R5) |
| Tier 2: Boundary & Corner Cases | 25 | 19 Passed, 6 Failed | M1 (R1, R2), M2 (R3), M3 (R4), M4 (R5) |
| Tier 3: Cross-Feature Interactions | 10 | 9 Passed, 1 Failed | M1, M2, M3, M4 |
| Tier 4 Gate 1: Gradient Health | 10 | 0 Passed, 10 Failed | M1 (Contradiction loss, prenorm, $\alpha=0.15$) |
| Tier 4 Gate 2: Accuracy Ceiling | 1 | 0 Passed, 1 Failed | M3 (Bilinear candidate-query alignment) |
| Tier 4 Gate 3: Paraconsistent Calibration | 2 | 0 Passed, 2 Failed | M2 (Positive support XOR, epistemic logit gating) |
| Tier 4 Gate 4: Throughput Speedup | 1 | 0 Passed, 1 Failed | M4 (Context encoding deduplication cache) |
| **Total E2E Suite** | **74** | **45 Passed, 29 Failed** | **All 74 Pass upon M5 Acceptance** |
