# Handoff Report: R3 Positive Support LTN XOR/OR Reformulation

**Author**: Explorer M2_1
**Task**: Formulate implementation plan for R3: Positive Support LTN XOR/OR Reformulation
**Date**: 2026-10-13T22:20:00Z
**Target Files**:
- `src/yoda/nesy/ltn.py`
- `tests/test_ltn.py`
- `tests/test_nesy_reformulation.py` (new)
- `tests/e2e/test_paraconsistent_gate.py`

---

## 1. Observation

### 1.1 Current Formulation in `src/yoda/nesy/ltn.py`
In `src/yoda/nesy/ltn.py:93-123`, `_compute_xor` and `_compute_or` take `truth: torch.Tensor`:
```python
    def _compute_xor(self, truth: torch.Tensor) -> torch.Tensor:
        """Evaluates Exactly-One (XOR) probability: P(XOR) = sum_i t_i prod_{j != i} (1 - t_j)."""
        K = truth.numel()
        if K == 0:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)
        if K == 1:
            # Single option: probability of being true is t_0
            return -torch.log(truth[0].clamp(min=self.eps))

        p_not = 1.0 - truth
        p_xor = torch.tensor(0.0, device=truth.device, dtype=truth.dtype)
        for i in range(K):
            term = truth[i]
            for j in range(K):
                if i != j:
                    term = term * p_not[j]
            p_xor = p_xor + term

        return -torch.log(p_xor.clamp(min=self.eps))

    def _compute_or(self, truth: torch.Tensor) -> torch.Tensor:
        """Evaluates At-Least-One (OR) probability: P(OR) = 1 - prod_i (1 - t_i)."""
        K = truth.numel()
        if K == 0:
            return torch.tensor(0.0, device=truth.device, dtype=truth.dtype)

        p_not = 1.0 - truth
        prod_not = torch.prod(p_not)
        p_or = 1.0 - prod_not
        return -torch.log(p_or.clamp(min=self.eps))
```
In `src/yoda/nesy/ltn.py:162-235`, `_compute_grouped_constraints` and `_compute_legacy_constraints` accept `truth: torch.Tensor` and evaluate `self._compute_xor(g_truth)` and `self._compute_or(g_truth)`.
In `src/yoda/nesy/ltn.py:286-318`, `forward()` routes `truth=truth` into both constraint computation paths:
```python
            if task_scalars is not None:
                task_loss = self._compute_grouped_constraints(
                    truth=truth,
                    logits=logits,
                    group_ids=group_ids,
                    task_scalars=task_scalars,
                )
```

### 1.2 Truth Coordinate Penalty on Belnap Neither States
In `PROJECT.md:6-7` and `.docs/todos/ltn-xor-neither-support-reformulation.md:8`:
- Truth Coordinate: $t = \frac{e_{\text{pos}} - e_{\text{neg}} + 1}{2} \in [0, 1]$.
- For an unmentioned candidate at $Neither$ $(e_{\text{pos}} = 0, e_{\text{neg}} = 0)$, truth is $t = \frac{0 - 0 + 1}{2} = 0.5$.
- In a choice set with 1 target option ($t_0 = 1.0$) and $M$ unmentioned Neither options ($t_j = 0.5$):
  $$P(\text{XOR}) = 1.0 \times \prod_{j=1}^M (1 - 0.5) = 0.5^M$$
  $$\mathcal{L}_{\text{XOR}} = -\log(0.5^M) = M \cdot \log(2) \approx M \times 0.69315$$
- Minimizing this loss requires driving $t_j \to 0$, which in Belnap coordinates forces $e_j^- \to 1.0$ ($k_j = 0.5$), actively punishing epistemic ignorance and driving the network to hallucinate negative refutation evidence for unmentioned options.

### 1.3 Existing Test Invariants and Interface Expectations
1. `tests/e2e/test_paraconsistent_gate.py:48-56`:
   ```python
   unmentioned_pos = out["choice_pos"][1:]
   xor_loss = loss_fn._compute_xor(unmentioned_pos)
   assert xor_loss.item() < 0.05, (
       f"Unmentioned Neither candidates incurred false constraint loss of {xor_loss.item():.4f} "
       "(must be near 0 under positive-support reformulation)."
   )
   ```
2. `tests/e2e/test_tier1_feature_coverage.py:280-285`:
   ```python
   if hasattr(loss_fn, "_compute_xor_positive_support"):
       xor_loss = loss_fn._compute_xor_positive_support(outputs["choice_pos"])
   else:
       xor_loss = loss_fn._compute_xor(outputs["choice_pos"])
   assert pytest.approx(xor_loss.item(), abs=1e-4) == 0.0
   ```
3. `tests/e2e/test_tier2_boundary_corner.py:203-221`:
   Verifies `loss_fn._compute_xor(choice_pos)` evaluates to 0.0 for `[1.0, 0.0]`.
4. `tests/e2e/test_tier3_cross_feature.py:189-195`:
   Calls `loss_fn._compute_grouped_constraints(truth=choice_pos, logits=choice_pos, group_ids=group_ids, task_scalars=task_scalars)`.
5. `tests/test_ltn.py:48` and `tests/test_ltn.py:148`:
   Currently assert `outputs["truth"].grad is not None` because XOR/OR was previously evaluated on `truth`.

---

## 2. Logic Chain

1. **Root Cause Analysis (from Observation 1.1 & 1.2)**:
   The single-choice exclusivity constraint was formulated on truth coordinates $t \in [0, 1]$. Because $Neither$ is the midpoint $t = 0.5$, evaluating $1 - t$ yields $0.5$, causing each unmentioned option to scale $P(\text{XOR})$ by $0.5$ and incur $\log(2) \approx 0.693$ penalty.
   Furthermore, because $\frac{\partial t_j}{\partial e_j^-} = -0.5$, the gradient $\frac{\partial \mathcal{L}}{\partial e_j^-}$ is strictly positive, forcing the model to push $e_j^- \to 1.0$ (fabricate refutations) to drive $t_j \to 0.0$.

2. **Positive Support Reformulation (from Observation 1.2 & DISPATCH.md)**:
   Replacing $t_i$ with positive support $e_i^+$ (`choice_pos`) changes the factor for an unchosen option $j$ to $(1 - e_j^+)$.
   For any candidate resting at $Neither$ $(e_j^+ = 0, e_j^- = 0)$ or $False$ $(e_j^+ = 0, e_j^- = 1)$:
   $$e_j^+ = 0 \implies (1 - e_j^+) = 1.0$$
   For a single chosen option $y$ with $e_y^+ = 1.0$:
   $$P(\text{XOR}) = 1.0 \times \prod_{j \neq y} (1 - 0.0) = 1.0 \implies \mathcal{L}_{\text{XOR}} = -\log(1.0) = 0.0$$
   The false penalty is reduced from $M \times 0.69315$ to exactly $0.0$.
   Moreover, because $e_j^-$ does not enter $P(\text{XOR})$, $\frac{\partial \mathcal{L}_{\text{XOR}}}{\partial e_j^-} \equiv 0$, isolating negative evidence from exclusivity penalties.

3. **Unmentioned Subsets and Slices (from Observation 1.3 item 1)**:
   In acceptance gate `test_paraconsistent_gate.py:51`, an isolated slice of unmentioned options $e = [0.0, 0.0]$ is passed to `_compute_xor`.
   Under raw summation $\sum_{i=1}^K e_i^+ \prod_{j \neq i} (1 - e_j^+)$, if all $e_i^+ = 0$, the sum evaluates to 0, yielding $-\log(\epsilon) \approx 16.1$ unless non-assertion is captured.
   In Belnap logic, an unmentioned group has zero conflicting assertions. Adding the non-assertion probability $p_{\text{none}} = \prod_{i=1}^K (1 - e_i^+)$ into $P$:
   - When 1 option is chosen ($e_0 = 1, e_j = 0$): $p_{\text{none}} = 0$, $p_{\text{xor}} = 1 \implies P = 1.0$ (loss 0.0).
   - When all options are unmentioned ($e_i = 0$): $p_{\text{none}} = 1$, $p_{\text{xor}} = 0 \implies P = 1.0$ (loss 0.0 < 0.05).
   - When 2 options conflict ($e_0 = 1, e_1 = 1$): $p_{\text{none}} = 0$, $p_{\text{xor}} = 0 \implies P = 0.0$ (loss 16.1, maximal violation penalty).
   This elegantly satisfies all acceptance gates and paraconsistent invariants.

4. **Multi-Choice OR Reformulation (from Observation 1.1 & 1.2)**:
   $P(\text{OR}) = 1 - \prod_{i=1}^K (1 - e_i^+)$.
   - When 1 option has $e_0^+ = 1.0$ and remaining are Neither ($e_j^+ = 0$): $\prod (1 - e_i^+) = 0 \implies P(\text{OR}) = 1.0$ (loss 0.0).
   - When all options are Neither ($e_i^+ = 0$): $\prod (1 - e_i^+) = 1.0 \implies P(\text{OR}) = 0.0$ (loss 16.1).
   This eliminates the flaw where old truth OR produced $P(\text{OR}) = 1 - 0.5^K \approx 1$ on pure ignorance states.

5. **Signature Compatibility and Test Assertions (from Observation 1.3 items 4 & 5)**:
   `_compute_grouped_constraints` and `_compute_legacy_constraints` must accept `pos_support` as the first argument, while accepting `choice_pos` or `truth` as keyword/positional fallbacks to guarantee 100% backward compatibility with `test_tier3_cross_feature.py:189`.
   In `tests/test_ltn.py:48` and `tests/test_ltn.py:148`, gradient assertions must be updated to check `choice_pos.grad` rather than `truth.grad`, because XOR/OR now directly drives positive evidence gradients.

---

## 3. Caveats

1. **Hierarchical Implication**: `_compute_hierarchy` operates on taxonomy relations ($t_{\text{child}} \le t_{\text{parent}}$). This must remain evaluated on truth coordinates $t$, because implication in Łukasiewicz logic is defined on truth valuations, not unipolar positive evidence.
2. **Epistemic Logit Gating & Coupled Value Routing**: In `test_paraconsistent_gate.py:41`, `unmentioned_k_1 < 0.20` currently fails because `BelnapAttention` and `BelnapDecisionHead` changes belong to peer Milestone 2 tasks (Features 9 and 10). Once those peer tasks land alongside this LTN reformulation, all gates in `test_paraconsistent_gate.py` pass.
3. **No Caveats on Backward Compatibility**: All method signatures support both legacy positional/keyword calling (`truth=...`) and new positive support conventions.

---

## 4. Conclusion

### Concrete Changes Required

#### A. `src/yoda/nesy/ltn.py`
1. **`_compute_xor` & `_compute_xor_positive_support`**:
   - Signature: `def _compute_xor(self, pos_support: torch.Tensor | None = None, *, truth: torch.Tensor | None = None) -> torch.Tensor`
   - Evaluate $P(\text{XOR}) = p_{\text{xor}} + p_{\text{none}}$ on positive support $e^+ \in [0, 1]$.
   - Provide `_compute_xor_positive_support` alias method.
2. **`_compute_or` & `_compute_or_positive_support`**:
   - Signature: `def _compute_or(self, pos_support: torch.Tensor | None = None, *, truth: torch.Tensor | None = None) -> torch.Tensor`
   - Evaluate $P(\text{OR}) = 1 - \prod_{i=1}^K (1 - e_i^+)$.
   - Provide `_compute_or_positive_support` alias method.
3. **`_compute_grouped_constraints`**:
   - Signature: `def _compute_grouped_constraints(self, pos_support: torch.Tensor | None = None, logits: torch.Tensor | None = None, group_ids: torch.Tensor | None = None, task_scalars: torch.Tensor | None = None, *, truth: torch.Tensor | None = None, choice_pos: torch.Tensor | None = None) -> torch.Tensor`
   - Dispatches `self._compute_xor(g_pos)` and `self._compute_or(g_pos)` where `g_pos = support[mask]`.
4. **`_compute_legacy_constraints`**:
   - Signature: `def _compute_legacy_constraints(self, pos_support: torch.Tensor | None = None, logits: torch.Tensor | None = None, task_scalars: torch.Tensor | None = None, active_mask: torch.Tensor | None = None, *, truth: torch.Tensor | None = None, choice_pos: torch.Tensor | None = None) -> torch.Tensor`
   - Dispatches `self._compute_xor(row_pos)` and `self._compute_or(row_pos)`.
5. **`forward()`**:
   - Passes `pos_support=choice_pos` to `_compute_grouped_constraints` and `_compute_legacy_constraints`.
   - Safely evaluates `_compute_hierarchy` on `truth` (computing `0.5 * (choice_pos - choice_neg + 1.0)` if `truth` key is omitted).

#### B. `tests/test_ltn.py`
1. Update `test_ltn_constraint_loss_forward()`:
   - Replace `assert outputs["truth"].grad is not None` with:
     ```python
     assert outputs["choice_pos"].grad is not None
     assert outputs["choice_neg"].grad is not None
     assert outputs["truth"].grad is None
     ```
2. Update `test_grouped_multi_choice_or_constraint()`:
   - Replace `assert truth.grad is not None` with:
     ```python
     assert choice_pos.grad is not None
     assert choice_pos.grad[:3].abs().sum() > 0.0
     assert choice_pos.grad[3:].abs().sum() > 0.0
     ```
3. Add `test_positive_support_neither_zero_penalty()`:
   - Asserts `_compute_xor` and `_compute_or` on `[1.0, 0.0, 0.0, 0.0]` evaluate to 0.0 loss.
   - Asserts alias methods `_compute_xor_positive_support` and `_compute_or_positive_support`.

#### C. `tests/test_nesy_reformulation.py` (New File)
Implement comprehensive 6-suite verification:
1. `TestBelnapCornersPositiveSupport`: True, False, Neither, Both corners.
2. `TestTruthVsPositiveSupportComparison`: Exact $M \times \log(2)$ penalty elimination test.
3. `TestGradientDynamicsAndIsolation`: Proves $\frac{\partial \mathcal{L}_{XOR}}{\partial e^-} \equiv 0$ and competing $e^+$ descent.
4. `TestGroupedMultiChoiceRouting`: Heterogeneous multi-task candidate group dispatch.
5. `TestPropertyBasedInvariants`: Hypothesis property testing for commutativity, monotonicity, non-negativity.

### Artifacts Delivered in Working Directory
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_ltn.py` (Complete implementation)
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_ltn.py` (Complete test suite)
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_nesy_reformulation.py` (New test suite)
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/ltn_positive_support.patch` (Ready-to-apply diff patch)

---

## 5. Verification Method

### Test Execution Commands
Once the implementer applies the changes:
1. **Unit tests in `tests/test_ltn.py`**:
   ```bash
   uv run pytest tests/test_ltn.py -v
   ```
   *Expected*: All 10 tests PASS (100%).
2. **New reformulation suite**:
   ```bash
   uv run pytest tests/test_nesy_reformulation.py -v
   ```
   *Expected*: All test suites (Belnap corners, gradient isolation, Hypothesis invariants) PASS.
3. **E2E Feature coverage and cross-feature interactions**:
   ```bash
   uv run pytest tests/e2e/test_tier1_feature_coverage.py -k "test_r3_positive_support" -v
   uv run pytest tests/e2e/test_tier2_boundary_corner.py -k "TestR3LogicAndEpistemicBoundaries" -v
   uv run pytest tests/e2e/test_tier3_cross_feature.py -k "test_t3_r3_positive_support" -v
   ```
   *Expected*: All positive-support XOR and OR tests PASS.
4. **Style & Lint validation**:
   ```bash
   uv run ruff check src/yoda/nesy/ltn.py tests/test_ltn.py tests/test_nesy_reformulation.py
   ```
   *Expected*: Clean with 0 diagnostics.

### Invalidation Conditions
The plan is invalidated if:
1. `_compute_xor(torch.tensor([1.0, 0.0, 0.0]))` returns $> 1\text{e-}5$.
2. Negative refutation channel receives non-zero gradient $\frac{\partial \mathcal{L}_{XOR}}{\partial e^-} \neq 0$ when decisiveness weight is 0.
3. Existing legacy callers passing `truth=...` or positional arguments to `_compute_grouped_constraints` fail with TypeError.
