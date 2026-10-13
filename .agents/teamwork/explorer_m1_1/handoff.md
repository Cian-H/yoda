# Handoff Report: R1 Attention Mask Propagation & Scale Normalization

**Agent:** Explorer M1_1  
**Working Directory:** `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1`  
**Milestone:** M1 (Phase 1 Architectural Hardening - R1)  
**Parent Agent:** `ead94b72-0198-425b-a453-dfb30ce62240`  
**Date:** 2026-10-13  

---

## 1. Observation

### 1.1 Baseline Test Suite Execution
- **Command:** `uv run pytest`
- **Result:** 181 passed in 5.23s across 20 test modules.
- **Status:** Baseline suite is 100% green with zero failures.

### 1.2 Discarded Attention Mask in `TextEncoder`
In `src/yoda/architecture/encoders.py` (lines 50–82):
```python
50:     def forward(self, texts: list[str]) -> torch.Tensor:
...
70:         inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
71:         if self.model is not None:
72:             target_device = next(self.model.parameters()).device
73:             inputs = inputs.to(target_device)
...
78:         outputs = self.model(**inputs)
79:         # Sequence of hidden states (batch_size, seq_len, hidden_size)
80:         hidden_states = outputs.last_hidden_state
81:         return self.adapter(hidden_states)
```
- `inputs["attention_mask"]` is computed by the HuggingFace tokenizer with `padding=True`, but discarded on line 81.
- For `model_name == "dummy"`, line 67 returns `self.dummy_embed(fake_ids)` without generating or returning any attention mask.
- `SymbolicStateEncoder.forward` (line 100) and `ConstraintEncoder.forward` (line 155) return `self.text_encoder(...)` directly without mask propagation.

### 1.3 `BelnapMultiheadPooledAttention` Mask Calling Convention
In `src/yoda/architecture/attention.py` (lines 244–290):
```python
244:     def forward(
245:         self,
246:         x: BelnapState | torch.Tensor,
247:         mask: torch.Tensor | None = None,
248:         return_latent: bool | None = None,
249:     ) -> BelnapState | tuple[torch.Tensor, BelnapState]:
...
289:         pooled_state = self.belnap_attn(q=q_state, kv=kv_state, mask=mask)
```
- Line 255 docstring states: `mask: Optional attention mask of shape (batch_size, 1, num_queries, seq_len)`.
- In `src/yoda/architecture/engine.py` (lines 374–376), all three epistemic pooling calls (`q_mpa`, `c_mpa`, `k_mpa`) pass `mask=None`:
```python
374:         q_state = cast(BelnapState, self.q_mpa(q_emb))
375:         c_state = cast(BelnapState, self.c_mpa(c_emb))
376:         k_state = cast(BelnapState, self.k_mpa(k_emb))
```

### 1.4 `BelnapAttention` Mask Injection and Aggregation Deficiencies
In `src/yoda/architecture/belnap_transformer.py` (lines 179–203):
```python
179:         if mask is not None:
180:             if mask.dtype == torch.bool:
181:                 s_pos = s_pos.masked_fill(~mask, -1e9)
182:                 s_neg = s_neg.masked_fill(~mask, -1e9)
183:             else:
184:                 s_pos = s_pos + mask
185:                 s_neg = s_neg + mask
186: 
187:         # Step 3: Knowledge Gating by source token intrinsic knowledge
188:         # Intrinsic knowledge mass per source token: k_src in [0, 1]
189:         k_src = kv.knowledge.mean(dim=-1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_kv)
190:         k_gate = 0.5 + 0.5 * k_src
191:         a_pos = torch.sigmoid(s_pos) * k_gate
192:         a_neg = torch.sigmoid(s_neg) * k_gate
193: 
194:         # Step 4: Evidence Aggregation
195:         agg_pos = torch.matmul(a_pos, v_pos)  # (B, H, L_q, d_k)
196:         agg_neg = torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)
```
1. **Dimension mismatch on 2D masks:** `s_pos` has shape `(B, H, L_q, L_kv)`. If `mask` is 2D `(B, L_kv)` from standard tokenizers, `~mask` has 2 dimensions. PyTorch broadcasting fails with a shape mismatch error because trailing dimensions cannot align without unsqueezing to `(B, 1, 1, L_kv)`.
2. **Mask dtype confusion:** If `mask` is an integer tensor (`torch.long` from Hugging Face with 1 for active, 0 for pad), line 180 `mask.dtype == torch.bool` is False, falling into line 184 `s_pos = s_pos + mask`. For padding tokens (0), `s_pos + 0 = s_pos`, so padding tokens are NOT masked at all.
3. **Linear sequence-length scaling saturation:** `agg_pos = torch.matmul(a_pos, v_pos)` sums across all $L_{\text{kv}}$ key-value tokens without dividing by active sequence length or sequence mass. As detailed in `.docs/plan.md:48-51` (Diagnosis 3), unnormalized aggregation scales linearly with sequence length $L_{\text{kv}}$, driving `torch.sigmoid(self.w_out_pos(agg_pos))` into hard saturation ($0.0$ or $1.0$) on Step 1.

---

## 2. Logic Chain

1. **Step 1 (Tokenizer Mask Retention):**
   - *From Observation 1.2:* `TextEncoder` produces `inputs["attention_mask"]` during tokenization but drops it on return.
   - *Inference:* Exposing `return_mask: bool = False` and `forward_with_mask` allows callers needing masks to receive `(embeddings, attention_mask)` while guaranteeing that existing callers expecting `torch.Tensor` (e.g. `tests/test_encoders.py:21`) experience zero regressions.
   - *Dummy Mode:* For `model_name == "dummy"`, `texts[i].split()` defines active words up to `seq_len`. Generating `mask[i, :words] = True` mirrors tokenizer behavior in offline testing.

2. **Step 2 (Cascaded Propagation into Encoders & Engine):**
   - *From Observation 1.2 & 1.3:* `SymbolicStateEncoder` and `ConstraintEncoder` wrap `TextEncoder`.
   - *Inference:* Adding `return_mask: bool = False` and `forward_with_mask` to both encoders ensures that formatted state strings (`"k: v | ..."`) and constraint strings (`"c1 || c2"`) preserve token masks into `YodaDecisionEngine`.
   - In `YodaDecisionEngine.forward`, passing `q_mask`, `c_mask`, and `k_mask` into `self.q_mpa`, `self.c_mpa`, and `self.k_mpa` propagates masks directly into cross-attention pooling.

3. **Step 3 (BelnapAttention Mask Ingestion & Numerical Robustness):**
   - *From Observation 1.4:* `s_pos` and `s_neg` are 4D `(B, H, L_q, L_kv)`.
   - *Inference:* Any input `mask` (2D `(B, L_kv)` or 3D `(B, L_q, L_kv)`) must be unsqueezed to 4D `(B, 1, 1, L_kv)` or `(B, 1, L_q, L_kv)`.
   - *Dtype Normalization:* Derive `bool_mask: torch.Tensor`:
     - If boolean: use directly.
     - If float with negative values ($< -1\text{e}4$): additive mask where active is $\ge -1\text{e}4$.
     - If float $\in [0, 1]$: binary float mask where active is $> 0.5$.
     - If integer: active is $\ne 0$.
   - *Affinity Masking:* Apply `s_pos.masked_fill(~bool_mask, -1e9)` and `s_neg.masked_fill(~bool_mask, -1e9)`.
   - *Zero-Weight Precision:* Apply `a_pos = a_pos.masked_fill(~bool_mask, 0.0)` and `a_neg = a_neg.masked_fill(~bool_mask, 0.0)` to eliminate subnormal float leakage.

4. **Step 4 (Active Sequence Mass Normalization):**
   - *From Observation 1.4 & Plan Diagnosis 3:* Summing across $L_{\text{kv}}$ without normalization causes activation explosion on long sequences and padding drowning.
   - *Inference:* Compute active token mass:
     $$\text{denom} = \begin{cases} \text{bool\_mask}.\text{float}().\text{sum}(\text{dim}=-1, \text{keepdim}=\text{True}).\text{clamp}(\min=1.0) & \text{if mask is not None} \\ \max(L_{\text{kv}}, 1.0) & \text{if mask is None} \end{cases}$$
   - Dividing `agg_pos = agg_pos / denom` and `agg_neg = agg_neg / denom` guarantees that evidence coordinates remain bounded in $O(1)$ independent of sequence length, perfectly preserving the linear gradient regime of downstream layers.

---

## 3. Caveats

1. **LayerNorm Pre-Sigmoid Interaction:**
   Explorer M1_3 is formulating Pre-Sigmoid LayerNorm (`nn.LayerNorm(d_model)`) in `BelnapAttention` and `BelnapFFN`. Scale normalization ($\text{agg} / \text{denom}$) and LayerNorm are complementary: scale normalization ensures input scale invariance to sequence length, while LayerNorm ensures zero-mean unit-variance centering. The implementer should ensure both are present in the final `BelnapAttention` pipeline.
2. **Batch-Level Completely Masked Edge Case:**
   If a sequence consists of 100% padding tokens, `bool_mask.float().sum()` is 0. Clamping `min=1.0` prevents division by zero, resulting in `agg_pos = 0.0 / 1.0 = 0.0` (pure Neither/uninformative state). This is safe and tested in `test_all_masked_tokens_stability`.
3. **Candidate Pooling in `ConstraintEncoder.encode_candidates`:**
   In `ConstraintEncoder.encode_candidates`, `cand_seq = self.text_encoder(flat_candidates)` pools across candidate tokens via `cand_seq.mean(dim=1)`. Using `forward_with_mask` with active-mask mean pooling will further prevent padding tokens within candidate text strings from diluting candidate representations.

---

## 4. Conclusion & Actionable Implementation Plan

### 4.1 Changes to `src/yoda/architecture/encoders.py`
1. Add `@overload` type annotations to `TextEncoder.forward`:
   - `forward(texts, return_mask: Literal[False] = False) -> torch.Tensor`
   - `forward(texts, return_mask: Literal[True]) -> tuple[torch.Tensor, torch.Tensor]`
   - `forward(texts, return_mask: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]`
2. Implement dummy mode mask generation:
   ```python
   mask = torch.zeros((batch_size, seq_len), device=dev, dtype=torch.bool)
   for i, t in enumerate(texts):
       words = len(t.split())
       if words > 0:
           mask[i, : min(words, seq_len)] = True
   ```
3. Implement HF mode mask extraction:
   ```python
   raw_mask = inputs["attention_mask"]
   mask = raw_mask.bool() if raw_mask.dtype != torch.bool else raw_mask
   ```
4. Expose `forward_with_mask(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]`.
5. Propagate `return_mask` and `forward_with_mask` through `SymbolicStateEncoder` and `ConstraintEncoder`.

### 4.2 Changes to `src/yoda/architecture/belnap_transformer.py`
In `BelnapAttention.forward(q, kv=None, mask=None)`:
```python
bool_mask: torch.Tensor | None = None
if mask is not None:
    if mask.dim() == 2:
        mask_4d = mask.unsqueeze(1).unsqueeze(2)
    elif mask.dim() == 3:
        mask_4d = mask.unsqueeze(1)
    else:
        mask_4d = mask

    if mask_4d.dtype == torch.bool:
        bool_mask = mask_4d
    elif mask_4d.dtype.is_floating_point:
        if (mask_4d < -1e4).any():
            bool_mask = mask_4d > -1e4
        else:
            bool_mask = mask_4d > 0.5
    else:
        bool_mask = mask_4d != 0

    s_pos = s_pos.masked_fill(~bool_mask, -1e9)
    s_neg = s_neg.masked_fill(~bool_mask, -1e9)

k_src = kv.knowledge.mean(dim=-1).unsqueeze(1).unsqueeze(2)
k_gate = 0.5 + 0.5 * k_src
a_pos = torch.sigmoid(s_pos) * k_gate
a_neg = torch.sigmoid(s_neg) * k_gate

if bool_mask is not None:
    a_pos = a_pos.masked_fill(~bool_mask, 0.0)
    a_neg = a_neg.masked_fill(~bool_mask, 0.0)

agg_pos = torch.matmul(a_pos, v_pos)
agg_neg = torch.matmul(a_neg, v_neg)

if bool_mask is not None:
    denom = bool_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
else:
    denom = max(float(l_kv), 1.0)

agg_pos = agg_pos / denom
agg_neg = agg_neg / denom

agg_pos = agg_pos.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)
agg_neg = agg_neg.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)
```

### 4.3 Changes to `src/yoda/architecture/engine.py`
In `YodaDecisionEngine.forward`:
- In Step 1 (Encoding): Call `forward_with_mask` on `text_encoder`, `state_encoder`, and `constraint_encoder` to obtain `(q_emb, q_mask)`, `(c_emb, c_mask)`, and `(k_emb, k_mask)`.
- In Step 2 (Pooling): Pass `mask=q_mask` into `self.q_mpa`, `mask=c_mask` into `self.c_mpa`, and `mask=k_mask` into `self.k_mpa`.

### 4.4 Reference Artifacts Provided
- Unified Git Patch: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/proposed_changes.patch`
- New Attention Masking Test Suite: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/proposed_test_attention_masking.py`

---

## 5. Verification Method

### 5.1 Independent Test Commands
```bash
# 1. Verify existing 181 tests continue to pass without regression
uv run pytest

# 2. Run new attention masking and scale normalization tests
uv run pytest tests/test_attention_masking.py -v

# 3. Run updated encoder tests
uv run pytest tests/test_encoders.py -v

# 4. Run Belnap transformer and decision engine tests
uv run pytest tests/test_belnap_transformer.py tests/test_belnap_mpa.py tests/test_engine.py -v

# 5. Run static lint and type checking
uv run ruff check .
```

### 5.2 Specific Conditions for Invalidation
The implementation is invalid if:
1. Any caller to `TextEncoder(texts)` expecting a single `torch.Tensor` receives a tuple instead (regression in `tests/test_encoders.py`).
2. A 2D mask `(B, L_kv)` passed to `BelnapAttention` causes PyTorch dimension broadcasting errors.
3. Adding padding tokens to a sequence alters the output of `BelnapAttention` on valid tokens (`test_padding_isolation` fails).
4. Replicating identical tokens across 2 vs 20 sequence lengths scales output evidence proportionally rather than remaining scale-invariant (`test_scale_normalization_sequence_length_invariance` fails).
5. All-masked sequences produce `NaN` or `Inf` due to zero-division in normalization (`test_all_masked_tokens_stability` fails).
