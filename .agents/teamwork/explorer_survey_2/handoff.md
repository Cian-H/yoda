# Handoff Report: Test Suite & Baseline Evaluation Survey

**Explorer**: Test and Baseline Explorer (Survey Phase)  
**Target Path**: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_survey_2/handoff.md`  
**Date**: 2026-10-13  

---

## 1. Observation

### 1.1 Tool Command Execution & Baseline Health

1. **Test Runner (`uv run pytest`)**:
   Executed via bash command `uv run pytest`.
   - Result: `181 passed, 4 warnings in 5.26s` (Exit code 0).
   - Warnings observed: 4 warnings in `tests/test_lightning.py::test_lightning_adapter_fast_dev_run` (GPU available but not used, `TreeSpec` deprecation, `DataLoader` worker suggestion).
   - Test count: Exactly 181 collected items across 20 test files, 100% passing.

2. **Linter & Formatter (`uv run ruff check .`)**:
   Executed via bash command `uv run ruff check .`.
   - Result: `All checks passed!` (Exit code 0).
   - Configuration in `pyproject.toml` (lines 36–58): target `py314`, line length 100, selects `["E", "F", "I", "W", "C90", "B", "C4", "SIM", "RET", "UP", "RUF"]` with `max-complexity = 15`.

3. **Latency Benchmark Script (`experiments/benchmark_latency.py`)**:
   Executed via `uv run python experiments/benchmark_latency.py --iterations 100 --warmup 10`.
   - Result: 100 iterations completed in `0.0001 s`, throughput reported as `1,930,688.5 evaluations/sec`, Mean Latency `0.0004 ms`.
   - Critical Inspection: In `experiments/benchmark_latency.py:55-72`, `simulate_system1_evaluation` computes:
     ```python
     constraints_satisfied = len(payload.constraints) > 0
     embeddings = payload.context.semantic_embedding
     score = sum(embeddings[:8]) if embeddings else 0.0
     ```
     The existing latency benchmark is a synthetic stub over `DecisionPayload` that does not invoke PyTorch, `YodaDecisionEngine`, `BelnapAttention`, or candidate scoring.

### 1.2 Test Inventory & Architecture Structure

The repository test directory (`tests/`) contains 20 test files spanning 181 tests:

| Test File | Test Count | Primary Coverage | Observed Deficiencies / Contracts |
|---|---|---|---|
| `test_activations.py` | 7 | `SoftExp`, `Mish`, `ParametricSoftExp` | Pure mathematical activation tests. |
| `test_attention.py` | 8 | `MultiheadPooledAttention` | Only tests `mask=None`; does not test padding token masks. |
| `test_belnap.py` | 4 | Belnap canonical states, uncertainty decomposition, `FuzzyBelnapLoss`, Łukasiewicz property test | Validates continuous $\mathcal{B}_4$ bilattice logic operators. |
| `test_belnap_mpa.py` | 11 | `BelnapMultiheadPooledAttention` | Tests 4-probe pooling, suppression with `init_bias=-3.0`, boolean/additive masks on probes, gradient flow. |
| `test_belnap_transformer.py` | 14 | `BelnapAttention`, `BelnapFFN`, `BelnapTransformerBlock`, `BelnapDecisionTransformer` | Tests `residual_join` at $\alpha=0.5$ (line 241); tests `BelnapFFN` summing $w_1(e_{\text{pos}}) + w_2(e_{\text{neg}})$; tests attention without LayerNorm. |
| `test_data_ingest.py` | 6 | ETL schemas, tokens, batch ingestion | Ingestion pipeline validation. |
| `test_encoders.py` | 10 | `TextEncoder`, `SymbolicStateEncoder`, `ConstraintEncoder` | Tests `TextEncoder.forward` returning a single `torch.Tensor` (lines 23, 38, 83) without attention masks. |
| `test_engine.py` | 14 | `YodaDecisionEngine`, `BelnapDecisionHead` | Lines 91–108: `BelnapDecisionHead` with `num_choices == 1` completely ignores `candidate_states`; line 96: `logits = raw_pos - raw_neg` without knowledge gating. |
| `test_hyperopt.py` | 12 | Optuna trial runner, early stopping, CLI command builder | Mock subprocess tests for trial execution. |
| `test_imports.py` | 9 | Subpackage import integrity | Top-level exports. |
| `test_independent_eval.py` | 7 | Independent collation, grouped losses (`FocalLoss`, `MarginLoss`), grouped LTN, single candidate forward pass | Only 7 unit tests. Does not test query cancellation or throughput. |
| `test_lightning.py` | 3 | `YodaLightningAdapter` step, fast dev run | Decoupled adapter validation. |
| `test_losses.py` | 7 | `FocalLoss`, `MarginLoss`, `FocalMarginLoss` | Margin and focal behavior on logits. |
| `test_ltn.py` | 9 | `LTNConstraintLoss` | Lines 248–286 explicitly verify the quadratic decisiveness penalty $p(1-p)$ that causes Jacobian suicide. Lines 64–82: `_compute_xor` evaluates on truth, penalizing $Neither$ by 0.693. |
| `test_optuna_study_focused.py` | 4 | Focused Optuna trial #40 parameter space | Validates focused bounds. |
| `test_optuna_study_narrow.py` | 4 | Narrow Optuna trial #21 parameter space | Line 31 asserts `FIXED_RESIDUAL_WEIGHT == 0.6356` (known gradient-attenuating residual weight). |
| `test_parquet_dataset.py` | 8 | `YodaParquetDataset`, dynamic splits, target matching | Parquet data loader and slicing. |
| `test_schema.py` | 2 | `DecisionPayload` schema | Schema validation. |
| `test_training.py` | 39 | `YodaTrainer`, cyclical schedulers, loss telemetry, collapse pruning, CLI args | Line 896 explicitly mocks collapsed accuracy `0.366875` for collapse pruning test; line 566 tests infinitesimal gradient warning via mock. |
| `test_xai_dla.py` | 3 | Direct Logit Attribution evaluator | Stage attribution metrics. |

### 1.3 Fixture Inventory & Duplications

1. **Shared Fixtures**: No global `tests/conftest.py` exists in the repository.
2. **Local Fixture Duplication**:
   - `dummy_engine`: Defined independently in `tests/test_engine.py:68` (`embed_dim=64`, 4 q-probes, 8 c-probes, 4 k-probes, 5 choices) and in `tests/test_training.py:110` (`embed_dim=32`, 2 q-probes, 4 c-probes, 2 k-probes, 5 choices).
   - `mock_jsonl_file` / `mock_jsonl`: Defined in `tests/test_training.py:20` and `tests/test_parquet_dataset.py:21`.
   - `synthetic_loader`: Defined in `tests/test_training.py:298` (10 items, batch size 4).
   - `synthetic_batch`: Defined in `tests/test_lightning.py:47` (2 items).

---

## 2. Logic Chain

### 2.1 Why the Current 181 Tests Pass Despite Mathematical Bottlenecks
1. In `tests/test_ltn.py:248-286`, `test_decisiveness_penalty` asserts that `outputs_fence` ($pos=0.5, neg=0.5$) incurs loss $0.1$ and tests that the gradient pushes away from 0.5. The test directly verifies the quadratic formula $pos(1-pos) + neg(1-neg)$. The test passes because it tests the *presence* of the flawed loss function, not its effect on upstream training dynamics.
2. In `tests/test_engine.py:134-166` and `tests/test_engine.py:376-408`, gradient flow is tested on a single synthetic step where initial activations are random and parameters have not yet saturated. The test asserts `grad is not None` on step 1, which succeeds because gradient vanishing occurs after optimizer step updates drive pre-sigmoid logits to $\pm 5.0$.
3. In `tests/test_independent_eval.py:142-162`, `test_single_candidate_forward_pass` checks that the output shapes match `(3,)` and coordinates are in $[0, 1]$. It does not test whether candidate rankings within a group vary with query changes or whether the query term cancels out.
4. In `tests/test_optuna_study_narrow.py:31`, the test explicitly enforces `assert FIXED_RESIDUAL_WEIGHT == 0.6356`. This locked hyperparameter causes residual attenuation $(1 - 0.6356)^6 = (0.3644)^6 \approx 0.0023$ (99.77% gradient attenuation across 6 joins).

### 2.2 Coverage Gap Analysis Against Acceptance Criteria

1. **AC: Epoch 1 Gradient Norms ($> 1\text{e-}4$ across 10 initialization seeds)**:
   - *Current Coverage*: 0%.
   - *Gap*: No test executes multi-seed backpropagation over an entire epoch while logging `torch.norm(engine.reasoning_layers[0].parameters())`. Existing tests either check step 1 gradient non-nullity (`test_backward_gradient_flow`) or mock zero gradients (`test_trainer_infinitesimal_gradient_warning`).
2. **AC: Multi-Choice Validation Accuracy (> 60.0%)**:
   - *Current Coverage*: 0%.
   - *Gap*: The test suite only tests that training stops when accuracy $\le 0.370$ (`test_trainer_collapse_pruning`). There is no benchmark test verifying that independent candidate evaluation breaks the 50% ceiling and reaches $> 60.0\%$.
3. **AC: Paraconsistent Calibration ($k < 0.20$ Neither without constraint loss)**:
   - *Current Coverage*: 0%.
   - *Gap*: In `src/yoda/nesy/ltn.py:64-82`, `_compute_xor` computes XOR over $t_j$. For an unmentioned candidate at $Neither$ ($e^+=0, e^-=0$), $t = 0.5$, which incurs $-\log(1 - 0.5) = 0.693$ loss per candidate. No test currently checks that an unmentioned candidate with $k < 0.20$ receives zero constraint penalty.
4. **AC: Low-Knowledge Logit Synchronization ($k \to 0 \implies \text{logits} \approx 0$)**:
   - *Current Coverage*: 0%.
   - *Gap*: `BelnapDecisionHead.forward` in `src/yoda/architecture/engine.py:96` computes $\text{logits} = raw\_pos - raw\_neg$. For a candidate with $raw\_pos = -8.0, raw\_neg = -12.0$, knowledge is $k \approx 0.0001$, yet logit is $+4.0$, creating high softmax confidence. No test currently verifies that low-knowledge candidates have suppressed logits.
5. **AC: Performance & Throughput ($\ge 2.0\times$ throughput speedup)**:
   - *Current Coverage*: 0%.
   - *Gap*: `experiments/benchmark_latency.py` benchmarks a mock rule on `DecisionPayload` (taking $0.0001$ seconds total) and does not touch `YodaDecisionEngine`. There is no candidate evaluation throughput benchmark before and after context caching.

---

## 3. Caveats

1. **Read-Only Scope**: In accordance with the Explorer archetype, no source files or existing test files were modified during this survey.
2. **GPU Environment**: The current evaluation environment ran on CPU. While CPU execution is sufficient for unit tests, gradient flow checks, and throughput ratio comparisons, absolute wall-clock latency figures differ on GPU accelerators.
3. **Existing Test Modification Risk**: When implementing Phase 1 (Item 1.2: replacing quadratic decisiveness penalty with contradiction penalty) and Phase 1 (Item 1.3: tuning residual weight from 0.6356 to 0.15), existing tests `tests/test_ltn.py::test_decisiveness_penalty` and `tests/test_optuna_study_narrow.py::test_fixed_parameters_consistency` will fail unless their assertion targets are updated to match the new architectural specification.

---

## 4. Conclusion

The Yoda test suite is currently green (181/181 passing, 0 ruff errors), but it **validates the exact contracts that cause the 36.5% plateau and 50% ceiling**. To meet the project Acceptance Criteria, the engineering phases must implement the fixes in `.docs/plan.md` and introduce three new test categories:

1. **Global Test Fixture Refactoring (`tests/conftest.py`)**:
   Consolidate duplicated fixtures across `test_engine.py`, `test_training.py`, and `test_parquet_dataset.py` into a unified `conftest.py` with reusable multi-choice batches, variable padding masks, and unmentioned candidate states.

2. **Required New Unit & Property Tests**:
   - `tests/test_attention_masking.py`: Verify tokenizer padding mask propagation through `TextEncoder`, zero-attention weighting on padding tokens, and active token mass scale normalization in $[0, 1]$.
   - `tests/test_contradiction_loss.py`: Verify contradiction penalty ($Both$: $choice\_pos \times choice\_neg$) gives 0 for $Neither$ $(0,0)$, $True$ $(1,0)$, and $False$ $(0,1)$, and only penalizes $Both$ $(1,1)$; verify linear warmup ramps weight from 0.0.
   - `tests/test_prenorm_highway.py`: Verify pre-sigmoid LayerNorm centers activations (mean 0, std 1) in `BelnapAttention` and `BelnapFFN`; verify residual gradient retention with $\alpha = 0.15$ exceeds $35\%$ across 6 joins.
   - `tests/test_nesy_reformulation.py`: Verify XOR/OR evaluated on positive support $e^+$ does not penalize unmentioned options at $Neither$ ($k < 0.20$); verify property-based invariants (commutativity, monotonicity) using Hypothesis per `nesy-constraint-check` skill.
   - `tests/test_coupled_routing.py`: Verify $2 \times 2$ bilattice aggregation preserves refutation signals when queries align with negative key evidence.
   - `tests/test_knowledge_gating.py`: Verify decision head logits scale with knowledge $k$, preventing high confidence on zero-knowledge inputs.
   - `tests/test_bilinear_head.py`: Verify bilinear Belnap inner product conditioning on candidate states breaks query cancellation in independent evaluation mode.
   - `tests/test_dual_ffn.py`: Verify negation equivariance $ffn(x.negate()) == ffn(x).negate()$ through dual-channel FFN.
   - `tests/test_context_cache.py`: Verify identical queries and states across unrolled candidate groups are encoded once per group.

3. **Required Integration & Regression Harnesses (Acceptance Gates)**:
   - `tests/test_gradient_health.py` (Gate AC 2): Run Epoch 1 backprop across 10 random seeds ($0..9$) and assert average gradient norm of `engine.reasoning_layers[0]` $> 1\text{e-}4$ at all steps.
   - `tests/test_eval_accuracy_gate.py` (Gate AC 2): Benchmark multi-choice validation accuracy on independent choice evaluation dataset to confirm $> 60.0\%$ (breaking the 50.0% ceiling).
   - `experiments/benchmark_independent_throughput.py` (Gate AC 4): True neural engine candidate evaluation throughput benchmark comparing un-cached vs cached encoder passes to confirm $\ge 2.0\times$ speedup.

---

## 5. Verification Method

To independently verify all findings and test suite states reported above:

1. **Verify Existing Test Suite Pass (181 Tests)**:
   ```bash
   uv run pytest
   ```
   *Expected output*: `181 passed, 4 warnings in ~5s`.
   *Invalidation condition*: Any test failures or collection errors.

2. **Verify Code Style and Lint Status**:
   ```bash
   uv run ruff check .
   ```
   *Expected output*: `All checks passed!`.
   *Invalidation condition*: Any lint or formatting errors.

3. **Inspect the Flawed Contracts in Existing Tests**:
   - Inspect quadratic decisiveness penalty test:
     ```bash
     grep -n -A 20 "def test_decisiveness_penalty" tests/test_ltn.py
     ```
   - Inspect XOR truth evaluation penalizing Neither:
     ```bash
     grep -n -A 20 "def _compute_xor" src/yoda/nesy/ltn.py
     ```
   - Inspect query-context cancellation in scalar head:
     ```bash
     grep -n -A 25 "if self.num_choices == 1:" src/yoda/architecture/engine.py
     ```
   - Inspect locked residual weight 0.6356:
     ```bash
     grep -n "FIXED_RESIDUAL_WEIGHT" tests/test_optuna_study_narrow.py
     ```

4. **Verify Mock-Only Benchmark Status**:
   ```bash
   uv run python experiments/benchmark_latency.py --iterations 50
   ```
   *Inspect `experiments/benchmark_latency.py:55-72` to verify it does not load PyTorch model weights or evaluate candidate tokens.*
