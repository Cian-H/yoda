# Handoff Report: Review & Adversarial Scrutiny of Milestone 2 (Phase 2 NeSy Reformulation)

**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

### 1.1 Test & Lint Execution Commands and Results

1. **Dedicated Milestone 2 Verification Suite**:
   ```bash
   uv run pytest tests/test_ltn.py tests/test_nesy_reformulation.py tests/test_belnap_transformer.py tests/test_coupled_routing.py tests/test_engine.py tests/test_knowledge_gating.py -v
   ```
   **Output:**
   ```
   ============================== 91 passed in 3.29s ==============================
   ```
   *Status*: Confirmed. All 91 dedicated unit and property tests pass.

2. **Codebase Lint and Style Check**:
   ```bash
   uv run ruff check .
   ```
   **Output:**
   ```
   Found 29 errors.
   [*] 5 fixable with the `--fix` option.
   ```
   Specifically targeting the modified/created files in M2:
   ```bash
   uv run ruff check src/yoda/nesy/ltn.py src/yoda/architecture/belnap_transformer.py src/yoda/architecture/engine.py tests/test_ltn.py tests/test_belnap_transformer.py tests/test_engine.py tests/test_nesy_reformulation.py tests/test_coupled_routing.py tests/test_knowledge_gating.py
   ```
   **Output:**
   ```
   E501 Line too long (113 > 100)
      --> src/yoda/architecture/engine.py:136:101
   E501 Line too long (107 > 100)
      --> src/yoda/nesy/ltn.py:101:101
   E501 Line too long (105 > 100)
      --> src/yoda/nesy/ltn.py:230:101
   E501 Line too long (106 > 100)
      --> src/yoda/nesy/ltn.py:281:101
   E501 Line too long (102 > 100)
      --> tests/test_engine.py:65:101
   E501 Line too long (101 > 100)
      --> tests/test_engine.py:81:101
   E501 Line too long (102 > 100)
      --> tests/test_nesy_reformulation.py:7:101
   I001 [*] Import block is un-sorted or un-formatted
      --> tests/test_nesy_reformulation.py:12:1
   E501 Line too long (105 > 100)
      --> tests/test_nesy_reformulation.py:59:101
   E501 Line too long (103 > 100)
      --> tests/test_nesy_reformulation.py:111:101
   E501 Line too long (104 > 100)
      --> tests/test_nesy_reformulation.py:240:101
   Found 11 errors.
   [*] 1 fixable with the `--fix` option.
   ```
   *Observation*: In `worker_m2_1/handoff.md` Section 5, the worker stated:
   `"Expected Outcome: All checks pass without violations."`
   This claim is refuted by independent execution: 11 lint errors directly exist in the worker's files, causing `uv run ruff check .` to exit with status 1.

3. **Tier 4 Paraconsistent Calibration Gate**:
   ```bash
   uv run pytest tests/e2e/test_paraconsistent_gate.py -v
   ```
   **Output:**
   ```
   FAILED tests/e2e/test_paraconsistent_gate.py::test_neither_candidates_retain_low_knowledge_without_penalty
   E AssertionError: Unmentioned candidate 1 knowledge must be < 0.20, got 0.8343
   E assert 0.8342664837837219 < 0.2
   tests/e2e/test_paraconsistent_gate.py:41: AssertionError
   ```

### 1.2 Code Inspection Observations

1. **`src/yoda/nesy/ltn.py`**:
   - Lines 121-123:
     ```python
     if K == 1:
         # Single option: probability of being chosen is e_0^+
         return -torch.log(flat[0].clamp(min=self.eps, max=1.0))
     ```
     When $K=1$ and $flat[0] = 0.0$, returns $-\log(\epsilon) \approx 16.11$.
     When $K \ge 2$ and all elements are $0.0$, lines 137-140 evaluate $p_{\text{none}} = \prod (1 - 0) = 1.0$, returning $-\log(1.0) = 0.0$.
   - Lines 126-132: $O(K^2)$ unvectorized nested Python loop for $p_{\text{xor}}$ accumulation.

2. **`src/yoda/architecture/belnap_transformer.py`**:
   - Lines 222-229:
     ```python
     agg_pos = torch.matmul(a_pos, v_pos) + torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)
     agg_neg = torch.matmul(a_pos, v_neg) + torch.matmul(a_neg, v_pos)  # (B, H, L_q, d_k)

     if bool_mask is not None:
         denom = bool_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
         agg_pos = agg_pos / denom
         agg_neg = agg_neg / denom
     ```
     When `mask is None`, `bool_mask` is `None`, so `denom` is completely bypassed. Passing an all-ones boolean mask (`torch.ones(B, L_kv, dtype=torch.bool)`) divides aggregations by $L_{kv}$, but omitting the mask (`mask=None`) leaves aggregations unscaled by $L_{kv}$.

3. **`src/yoda/architecture/engine.py`**:
   - Lines 110-140: `BelnapDecisionHead.forward` implements candidate evidence gating using `cand_k` when `candidate_states` is passed:
     ```python
     cand_k = (candidate_states.e_pos + candidate_states.e_neg).mean(dim=-1) / 2.0
     cand_gate = (2.0 * cand_k).clamp(max=1.0)
     choice_pos = torch.sigmoid(pos_affinity) * cand_gate
     choice_neg = torch.sigmoid(neg_affinity) * cand_gate
     ```
   - Lines 455-457: In `YodaDecisionEngine.forward`:
     ```python
     if is_independent:
         head = self.scalar_head if self.num_choices != 1 else self.decision_head
         final_out = head(final_state)
     ```
     `candidate_states=k_state` is NOT passed to `head`. As a result, `state_k` is computed from `final_state` (which has high residual query evidence), resulting in knowledge `0.8343` rather than candidate knowledge `< 0.20`.

---

## 2. Logic Chain

1. **Acceptance Criteria & Quality Gate Failure (Finding 1)**:
   - *Observation*: `ORIGINAL_REQUEST.md` Acceptance Criteria specifies: `"Codebase passes lint and style validation via uv run ruff check ."`
   - *Observation*: `worker_m2_1/handoff.md` Section 5 stated `Expected Outcome: All checks pass without violations`.
   - *Direct Measurement*: `uv run ruff check .` fails with 29 errors, 11 of which are in M2 files (`engine.py`, `ltn.py`, `test_engine.py`, `test_nesy_reformulation.py`).
   - *Inference*: The work product violates explicit lint acceptance criteria, and the handoff contained an unverified passing claim.

2. **Scale Invariance & Mask Equivalence Inconsistency (Finding 2)**:
   - *Observation*: In `BelnapAttention`, `agg_pos` and `agg_neg` are only divided by `denom` when `bool_mask is not None`.
   - *Reasoning*: A tensor sequence with no padding (`mask=None`) has identical semantic meaning and active mass as a sequence provided with an all-ones mask (`mask=torch.ones(...)`). Under the current implementation, `attn(q, kv, mask=None)` outputs activations whose pre-LayerNorm scale is $L_{kv}$ times larger than `attn(q, kv, mask=torch.ones(...))`.
   - *Inference*: Scale normalization violates consistency across masked vs unmasked call sites.

3. **Candidate Knowledge Disconnect in Engine Forward (Finding 3)**:
   - *Observation*: `Tier 4 Paraconsistent Gate` (`test_neither_candidates_retain_low_knowledge_without_penalty`) fails with `assert 0.8343 < 0.20`.
   - *Reasoning*: The worker implemented candidate knowledge gating in `BelnapDecisionHead.forward` for `candidate_states`, but in `YodaDecisionEngine.forward` line 457, `head(final_state)` is invoked without `candidate_states=k_state`. The residual highway in `BelnapTransformerBlock` ($\alpha=0.15$) ensures `final_state` retains $\approx 85\%$ of the query's activations, so `head` measures query knowledge rather than candidate knowledge.
   - *Inference*: Either `candidate_states=k_state` must be wired into `head` in independent choice evaluation mode, or the milestone boundaries must explicitly reconcile this dependency with M3.

4. **Mathematical Boundary Discontinuity (Finding 4)**:
   - *Observation*: For a slice of size $K=1$, `_compute_xor([0.0])` returns $16.11$ (severe violation penalty). For a slice of size $K=2$, `_compute_xor([0.0, 0.0])` returns $0.0$ (neutral satisfaction).
   - *Reasoning*: In open-world Belnap semantics, an unmentioned option ($Neither$) should not incur a contradiction or exclusivity penalty. A single unmentioned candidate being evaluated in an unrolled slice is penalized solely because $K=1$ assumes classical positive requirement rather than non-assertion.

---

## 3. Caveats

1. **Test Pass Rate**: The mathematical core of R3 ($2 \times 2$ coupled bilattice routing, double-negation symmetry, positive-support XOR/OR for multi-candidate sets, and epistemic logit gating on `BelnapDecisionHead`) is well-designed and passed all 91 unit/property tests.
2. **Phase 3 and 4 Downstream Failures**: The failures in `tests/e2e/test_gradient_health_gate.py`, `tests/e2e/test_accuracy_ceiling_gate.py`, and `tests/e2e/test_throughput_gate.py` are recognized as downstream requirements belonging to Milestone 3 (R4) and Milestone 4 (R5).

---

## 4. Conclusion & Findings

### Findings Summary

#### [Critical] Finding 1: Codebase Lint & Style Validation Failure (`uv run ruff check .`)
- **What**: 11 lint errors in M2 files (29 repo-wide) across `engine.py`, `ltn.py`, `test_engine.py`, and `test_nesy_reformulation.py`.
- **Where**:
  - `src/yoda/architecture/engine.py:136` (line length 113)
  - `src/yoda/nesy/ltn.py:101, 230, 281` (line lengths 105-107)
  - `tests/test_engine.py:65, 81` (line lengths 101-102)
  - `tests/test_nesy_reformulation.py:7, 12, 59, 111, 240` (import sort I001, line lengths 102-105)
- **Why**: Violates `ORIGINAL_REQUEST.md` acceptance criteria and contradicts the handoff report claim.
- **Suggestion**: Run `uv run ruff check --fix` and wrap lines at $\le 100$ characters.

#### [Major] Finding 2: Inconsistent Active Mass Normalization in `BelnapAttention`
- **What**: Attention value aggregation is only divided by `denom` when `bool_mask is not None`.
- **Where**: `src/yoda/architecture/belnap_transformer.py:226-229`
- **Why**: `attn(q, kv, mask=None)` scales with sequence length $L_{kv}$ while `attn(q, kv, mask=all_ones)` divides by $L_{kv}$, breaking scale invariance.
- **Suggestion**: Set `denom = float(l_kv)` when `bool_mask is None` (or clamp to 1.0) so that unmasked attention matches unpadded masked attention.

#### [Major] Finding 3: `YodaDecisionEngine.forward` Does Not Pass `candidate_states` to `scalar_head`
- **What**: Line 457 calls `final_out = head(final_state)` without `candidate_states=k_state`.
- **Where**: `src/yoda/architecture/engine.py:456-457`
- **Why**: Causes `tests/e2e/test_paraconsistent_gate.py:41` to fail because `head` computes knowledge from the query rather than the candidate options.
- **Suggestion**: Pass `candidate_states=k_state` into `head` in independent evaluation mode.

#### [Minor] Finding 4: Discontinuity in `_compute_xor` for $K=1$ on Zero Evidence
- **What**: $K=1$ with $e^+=0.0$ returns loss $16.11$, whereas $K \ge 2$ with all zeros returns loss $0.0$.
- **Where**: `src/yoda/nesy/ltn.py:121-123`
- **Why**: Unmentioned candidates in single-option slices are penalized while multi-candidate slices are neutral.
- **Suggestion**: Handle $flat[0] == 0.0$ gracefully under non-assertion semantics when $K=1$.

**Verdict**: **REQUEST_CHANGES**

---

## 5. Verification Method

To verify the requested changes:
1. **Lint Check**:
   ```bash
   uv run ruff check .
   ```
   *Expected outcome*: 0 errors.

2. **Milestone 2 Unit & Integration Suites**:
   ```bash
   uv run pytest tests/test_ltn.py tests/test_nesy_reformulation.py tests/test_belnap_transformer.py tests/test_coupled_routing.py tests/test_engine.py tests/test_knowledge_gating.py -v
   ```
   *Expected outcome*: 91 passed.

3. **Mask Consistency Regression Test**:
   Verify that `BelnapAttention` produces identical output for `mask=None` and `mask=torch.ones(B, L_kv, dtype=torch.bool)`:
   ```python
   out_none = attn(q, kv, mask=None)
   out_ones = attn(q, kv, mask=torch.ones(B, L_kv, dtype=torch.bool))
   assert torch.allclose(out_none.e_pos, out_ones.e_pos, atol=1e-5)
   ```

4. **Tier 4 Paraconsistent Gate**:
   ```bash
   uv run pytest tests/e2e/test_paraconsistent_gate.py -v
   ```
   *Expected outcome*: Both tests pass.
