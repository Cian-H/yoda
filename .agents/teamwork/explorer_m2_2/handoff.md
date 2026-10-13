# Handoff Report: R3 Coupled Cross-Polarity Bilattice Evidence Routing (2x2)

**Author**: Explorer M2_2  
**Milestone**: Milestone 2 (Phase 2 NeSy Reformulation)  
**Date**: 2026-10-13  
**Working Directory**: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_2`  
**Artifacts Generated**:
- Patch: `proposed_belnap_transformer.patch`
- Unit Test Patch: `proposed_test_belnap_transformer.patch`
- New Test Suite: `proposed_test_coupled_routing.py`

---

## 1. Observation

1. **Current Decoupled Aggregation in `src/yoda/architecture/belnap_transformer.py` (lines 222–230)**:
   ```python
   # Step 4: Evidence Aggregation
   agg_pos = torch.matmul(a_pos, v_pos)  # (B, H, L_q, d_k)
   agg_neg = torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)

   if bool_mask is not None:
       denom = bool_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
       agg_pos = agg_pos / denom
       agg_neg = agg_neg / denom
   ```

2. **Refutation Dropping Defect**:
   When a key token carries strong negative refutation evidence ($V_+ = 0, V_- = 1$) and an epistemic query aligns coherently with that key ($A_+ \gg 0, A_- \approx 0$):
   $$\text{agg}_+ = A_+ V_+ = A_+ \cdot 0 = 0$$
   $$\text{agg}_- = A_- V_- = 0 \cdot 1 = 0$$
   Both evidence channels collapse to $(0, 0)$ ($Neither$). The key's negative refutation evidence is dropped completely from the aggregated context.
   Direct empirical check:
   ```python
   # Coherent query (A_pos = 1.0, A_neg = 0.0) attending to refutation key (V_pos = 0.0, V_neg = 1.0)
   decoupled_pos = a_pos * v_pos  # 0.0
   decoupled_neg = a_neg * v_neg  # 0.0 -> Refutation dropped!
   ```

3. **Existing Roadmap Specifications**:
   - `ORIGINAL_REQUEST.md:24`: "Implement coupled cross-polarity evidence routing in bilattice attention so refutation signals are preserved under coherent query alignment."
   - `PROJECT.md:27`: "2x2 bilattice aggregation preserving refutation signals"
   - `.docs/todos/belnap-coupled-cross-polarity-value-routing.md:8`:
     $$\text{agg}_{\text{pos}} = A_+ V_+ + A_- V_-, \qquad \text{agg}_{\text{neg}} = A_+ V_- + A_- V_+$$
   - Acceptance criteria in `tests/e2e/test_tier1_feature_coverage.py:301–330`:
     `test_r3_coupled_cross_polarity_value_routing` and `test_r3_refutation_signal_preserved_under_coherent_query` require that coherent queries aligned with refutation keys route negative evidence into `out.e_neg`.

4. **Flaky Fixture in `tests/test_belnap_mpa.py:128–143`**:
   ```python
   def test_belnap_mpa_masking() -> None:
       ...
       # First 2 tokens have knowledge, last 2 tokens have knowledge
       e_pos = torch.ones(batch_size, seq_len, embed_dim) * 0.8
       e_neg = torch.zeros(batch_size, seq_len, embed_dim)
       ...
       assert not torch.allclose(latent_unmasked, latent_masked, atol=1e-4)
   ```
   All 4 tokens in this test are initialized identically (`0.8`), yet the test asserts `not allclose` when masking out the last 2 positions. Because scale normalization and `LayerNorm` enforce sequence-length invariance on identical tokens, unmasked (4 tokens) and masked (2 tokens) representations are nearly identical. In unpatched code, diff was $5.4\text{e-}4 > 1\text{e-}4$ (passing marginally). When coupled routing is introduced, depending on preceding test RNG states, diff can drop to $8.0\text{e-}5 < 1\text{e-}4$, failing this assertion. If the intended state `e_pos[:, 2:, :] = 0.0` is used, diff is $0.024$ ($240\times$ tolerance) and passes unconditionally.

---

## 2. Logic Chain

1. **Premise 1: Belnap-Dunn $\mathcal{B}_4$ Involutive Sign Multiplication**:
   In the four-valued bilattice $\mathcal{B}_4$, logical negation is an involution: $\neg(e^+, e^-) = (e^-, e^+)$. Opposing polarity alignment ($A_-$) represents an inverted relation ($\neg$), while coherent polarity alignment ($A_+$) represents an identity relation ($\text{id}$).
   Evidence routing must form the $2 \times 2$ circulant matrix operator:
   $$\begin{pmatrix} \text{agg}_+ \\ \text{agg}_- \end{pmatrix} = \begin{pmatrix} A_+ & A_- \\ A_- & A_+ \end{pmatrix} \begin{pmatrix} V_+ \\ V_- \end{pmatrix} = \begin{pmatrix} A_+ V_+ + A_- V_- \\ A_+ V_- + A_- V_+ \end{pmatrix}$$

2. **Premise 2: Refutation Preservation**:
   - For coherent alignment ($A_+ \gg 0, A_- \approx 0$) with refutation key ($V_+ = 0, V_- > 0$):
     $$\text{agg}_+ = A_+ \cdot 0 + 0 \cdot V_- = 0$$
     $$\text{agg}_- = A_+ V_- + 0 \cdot 0 = A_+ V_- \gg 0$$
   - The negative refutation signal is strictly preserved in $\text{agg}_-$.

3. **Premise 3: Double Negation Symmetry**:
   - Negating the query swaps affinities: $A'_+ = A_-, A'_- = A_+$.
     $$\text{agg}'_+ = A_- V_+ + A_+ V_- = \text{agg}_-$$
     $$\text{agg}'_- = A_- V_- + A_+ V_+ = \text{agg}_+$$
     Hence, $\text{Agg}(\neg Q, K, V) = \neg \text{Agg}(Q, K, V)$.
   - Negating the source token keys and values: $K' = \neg K, V' = \neg V$.
     Affinities swap ($A'_+ = A_-, A'_- = A_+$) and values swap ($V'_+ = V_-, V'_- = V_+$).
     $$\text{agg}''_p = A_- V_- + A_+ V_+ = \text{agg}_+$$
     $$\text{agg}''_n = A_- V_+ + A_+ V_- = \text{agg}_-$$
     Hence, $\text{Agg}(Q, \neg K, \neg V) = \text{Agg}(Q, K, V)$ (Law of Double Negation: $\neg(\neg P) \equiv P$).

4. **Premise 4: Active Mass Normalization Integration**:
   Dividing by $\text{denom} = \text{clamp}(\sum \mathbf{1}_{\text{mask}}, \min=1.0)$ when a mask is provided preserves the $[0, 1]$ bounding invariant and prevents variable-length sequence dilation.

5. **Deduction**:
   Replacing `agg_pos = torch.matmul(a_pos, v_pos)` and `agg_neg = torch.matmul(a_neg, v_neg)` with:
   ```python
   agg_pos = torch.matmul(a_pos, v_pos) + torch.matmul(a_neg, v_neg)
   agg_neg = torch.matmul(a_pos, v_neg) + torch.matmul(a_neg, v_pos)
   ```
   resolves the refutation dropping flaw, guarantees double negation symmetry, satisfies all 16 cases of the $\mathcal{B}_4 \times \mathcal{B}_4$ truth table, and maintains non-vanishing gradient flow across cross channels.

---

## 3. Caveats

1. **LayerNorm Scale Invariance & Linear Bias Perturbations**:
   The output projections in `BelnapAttention` apply pre-sigmoid LayerNorm (`norm_pos` and `norm_neg`). Because `LayerNorm(c * x) == LayerNorm(x)` for scalar $c > 0$, scale variations from identical repeated tokens are primarily moderated by LayerNorm, with residual offsets governed by linear projection biases (`w_out_pos.bias`).
2. **Weight Asymmetry in Untied Initializations**:
   In randomly initialized attention modules, `w_q_pos != w_q_neg` and `w_out_pos != w_out_neg`. Exact algebraic double negation symmetry ($\text{diff} = 0.0$) holds strictly on the core bilattice routing tensor operator and on modules with tied projection weights. For independently initialized layers, double negation symmetry holds within statistical feature variance until trained.
3. **Flaky Assertion in `tests/test_belnap_mpa.py`**:
   As noted in Observation 4, `test_belnap_mpa_masking` has a fixture with identical token evidence across all 4 positions (`0.8`). If run in sequence after certain tests that mutate PyTorch's default RNG state, the unmasked and masked outputs may differ by $8.0\text{e-}5$, marginally below `atol=1e-4`. We recommend updating `test_belnap_mpa.py:129` to `e_pos[:, 2:, :] = 0.0` to reflect the intended test comment ("First 2 tokens have knowledge, last 2 tokens have no knowledge").

---

## 4. Conclusion

The implementation strategy for **R3: Coupled Cross-Polarity Bilattice Evidence Routing (2x2)** is fully formulated, verified, and ready for application:

1. **Target File**: `src/yoda/architecture/belnap_transformer.py`
   - Modify Step 4 in `BelnapAttention.forward` (lines 222–225):
     ```python
     # Step 4: Evidence Aggregation (2x2 Coupled Cross-Polarity Bilattice Routing)
     agg_pos = torch.matmul(a_pos, v_pos) + torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)
     agg_neg = torch.matmul(a_pos, v_neg) + torch.matmul(a_neg, v_pos)  # (B, H, L_q, d_k)
     ```
   - Retain existing active mass denominator normalization:
     ```python
     if bool_mask is not None:
         denom = bool_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
         agg_pos = agg_pos / denom
         agg_neg = agg_neg / denom
     ```
2. **Unit Tests in `tests/test_belnap_transformer.py`**:
   - Add `test_coupled_routing_refutation_preservation` and `test_coupled_routing_double_negation_symmetry`.
3. **New Test Suite `tests/test_coupled_routing.py`**:
   - Place `proposed_test_coupled_routing.py` (25 tests covering 16-state truth table, double negation symmetry, refutation preservation, masking interaction, gradient flow, and transformer block integration).
   - 100% pass rate achieved and zero ruff lint errors.

---

## 5. Verification Method

### 1. Apply Proposed Patches
```bash
# In working tree:
git apply .agents/teamwork/explorer_m2_2/proposed_belnap_transformer.patch
git apply .agents/teamwork/explorer_m2_2/proposed_test_belnap_transformer.patch
cp .agents/teamwork/explorer_m2_2/proposed_test_coupled_routing.py tests/test_coupled_routing.py
```

### 2. Run Dedicated Verification Suites
```bash
# Verify new 2x2 coupled routing test suite (25 tests)
uv run pytest tests/test_coupled_routing.py -vv

# Verify existing transformer unit tests (16 tests)
uv run pytest tests/test_belnap_transformer.py -vv

# Verify E2E acceptance tests for R3 coupled routing
uv run pytest tests/e2e/test_tier1_feature_coverage.py -k "coupled or refutation" -vv
uv run pytest tests/e2e/test_tier3_cross_feature.py -k "coupled_routing" -vv
```

### 3. Verify Code Style & Linting
```bash
uv run ruff check src/yoda/architecture/belnap_transformer.py tests/test_belnap_transformer.py tests/test_coupled_routing.py
```

### 4. Invalidation Conditions
- If `test_coupled_routing_refutation_preservation` fails, negative refutation evidence is not routing to `agg_neg` under coherent positive query affinity ($A_+$).
- If `test_double_negation_symmetry_query_inversion` produces diff $> 1\text{e-}5$ on tied weights, circulant matrix cross-terms ($A_+ V_- + A_- V_+$) are asymmetric.
- If `test_cross_channel_gradient_flow` fails, cross-channel gradients are disconnected or vanishing.
