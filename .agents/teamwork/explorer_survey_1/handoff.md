# Architecture Mapping and Diagnostic Handoff Report

- **Agent:** Codebase Architecture Explorer (`explorer_survey_1`)
- **Working Directory:** `/home/cianh/Projects/yoda/.agents/teamwork/explorer_survey_1`
- **Milestone:** Survey & Architecture Mapping
- **Status:** Complete (Hard Handoff)
- **Target Codebase:** `/home/cianh/Projects/yoda/src/yoda/`

---

## 1. Observation

Direct code inspections across `/home/cianh/Projects/yoda/src/yoda/` and baseline test execution reveal the following verbatim lines and mechanisms:

### Baseline Test Suite Status
- **Test execution command:** `uv run pytest`
  - Result: 181 passed in 5.32s across 20 test files (`tests/`).
- **Linter execution command:** `uv run ruff check .`
  - Result: All checks passed.

---

### R1. Attention Mask Propagation & Scale Normalization
- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/encoders.py`
  - **Class:** `TextEncoder`
  - **Lines 70–81:**
    ```python
    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    if self.model is not None:
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
    elif self.device is not None:
        inputs = inputs.to(self.device)

    assert self.model is not None
    outputs = self.model(**inputs)
    # Sequence of hidden states (batch_size, seq_len, hidden_size)
    hidden_states = outputs.last_hidden_state
    return self.adapter(hidden_states)
    ```
    *Observation:* `inputs["attention_mask"]` is discarded. `TextEncoder.forward` returns only `self.adapter(hidden_states)`.
  - **Lines 52–67 (Dummy mode):**
    ```python
    fake_ids = torch.randint(0, 1000, (batch_size, seq_len), device=dev)
    return self.dummy_embed(fake_ids)
    ```
    *Observation:* Dummy mode also returns only embedding tensors without mask metadata.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/engine.py`
  - **Class:** `YodaDecisionEngine`
  - **Lines 363, 370–376:**
    ```python
    q_emb = self.text_encoder(queries)
    ...
    c_emb = self.state_encoder(states)
    k_emb = self.constraint_encoder(target_cand)

    # 2. Pooling (Epistemic Sieve)
    q_state = cast(BelnapState, self.q_mpa(q_emb))
    c_state = cast(BelnapState, self.c_mpa(c_emb))
    k_state = cast(BelnapState, self.k_mpa(k_emb))
    ```
    *Observation:* `self.q_mpa`, `self.c_mpa`, and `self.k_mpa` are invoked with default `mask=None`.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/belnap_transformer.py`
  - **Class:** `BelnapAttention`
  - **Lines 187–204:**
    ```python
    # Step 3: Knowledge Gating by source token intrinsic knowledge
    # Intrinsic knowledge mass per source token: k_src in [0, 1]
    k_src = kv.knowledge.mean(dim=-1).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_kv)
    k_gate = 0.5 + 0.5 * k_src
    a_pos = torch.sigmoid(s_pos) * k_gate
    a_neg = torch.sigmoid(s_neg) * k_gate

    # Step 4: Evidence Aggregation
    agg_pos = torch.matmul(a_pos, v_pos)  # (B, H, L_q, d_k)
    agg_neg = torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)

    agg_pos = agg_pos.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)
    agg_neg = agg_neg.transpose(1, 2).contiguous().view(batch_size, l_q, self.d_model)

    # Output bounds in [0, 1]
    out_pos = torch.sigmoid(self.w_out_pos(agg_pos))
    out_neg = torch.sigmoid(self.w_out_neg(agg_neg))
    ```
    *Observation:* Step 4 computes an unnormalized sum across all $L_{\text{kv}}$ key tokens ($\text{agg} = A \cdot V$) without dividing by active sequence length mass.

---

### R2. Paraconsistent Contradiction Loss & Pre-Activation Normalization
- **File:** `/home/cianh/Projects/yoda/src/yoda/nesy/ltn.py`
  - **Class:** `LTNConstraintLoss`
  - **Lines 251–254, 272–279:**
    ```python
    # Grouped candidate evaluation path
    if group_ids is not None:
        decisiveness_penalty = (
            choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
        ).mean() * self.decisiveness_weight
    ...
    # Legacy fixed-dimension evaluation path
    if mask_f is not None:
        denom = mask_f.sum().clamp(min=1.0)
        decisiveness_terms = choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
        decisiveness_penalty = (
            (decisiveness_terms * mask_f).sum() / denom
        ) * self.decisiveness_weight
    else:
        decisiveness_penalty = (
            choice_pos * (1.0 - choice_pos) + choice_neg * (1.0 - choice_neg)
        ).mean() * self.decisiveness_weight
    ```
    *Observation:* `decisiveness_terms` evaluates $p(1 - p)$ for both positive and negative evidence. No warmup schedule exists for `decisiveness_weight` in `LTNConstraintLoss`.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/belnap_transformer.py`
  - **Class:** `BelnapAttention` (Lines 202–203), `BelnapFFN` (Lines 247–248):
    ```python
    # BelnapAttention:
    out_pos = torch.sigmoid(self.w_out_pos(agg_pos))
    out_neg = torch.sigmoid(self.w_out_neg(agg_neg))

    # BelnapFFN:
    e_ffn_pos = torch.sigmoid(self.w_o1(h_mid))
    e_ffn_neg = torch.sigmoid(self.w_o2(h_mid))
    ```
    *Observation:* Linear projections `w_out_pos`, `w_out_neg`, `w_o1`, and `w_o2` feed directly into `torch.sigmoid` without intermediate normalization layers (`LayerNorm`).
  - **Class:** `BelnapTransformerBlock` (Lines 262, 289–301):
    ```python
    def __init__(..., residual_weight: float = 0.5, ...):
    ...
    @staticmethod
    def _residual_join(base: BelnapState, delta: BelnapState, alpha: float = 0.5) -> BelnapState:
        res_pos = (1.0 - alpha) * base.e_pos + alpha * delta.e_pos
        res_neg = (1.0 - alpha) * base.e_neg + alpha * delta.e_neg
        return BelnapState(e_pos=res_pos, e_neg=res_neg)
    ```
    *Observation:* Default $\alpha = 0.5$. Each block applies 3 residual joins (self-attn, cross-attn, FFN). Stacked across 2 layers, 6 convex joins attenuate skip gradients by $(1 - 0.5)^6 = 0.0156$ (1.56%).
- **File:** `experiments/optuna_study_narrow.py` (Line 33):
  `FIXED_RESIDUAL_WEIGHT = 0.6356`
  *Observation:* Under this fixed setting, $(1 - 0.6356)^6 = 0.0022$ (0.22% skip gradient survival).

---

### R3. NeSy Constraint Reformulation & Epistemic Logit Synchronization
- **File:** `/home/cianh/Projects/yoda/src/yoda/nesy/ltn.py`
  - **Class:** `LTNConstraintLoss`
  - **Lines 64–83 (`_compute_xor`) & Lines 84–94 (`_compute_or`):**
    ```python
    def _compute_xor(self, truth: torch.Tensor) -> torch.Tensor:
        ...
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
        ...
        p_not = 1.0 - truth
        prod_not = torch.prod(p_not)
        p_or = 1.0 - prod_not
        return -torch.log(p_or.clamp(min=self.eps))
    ```
    *Observation:* Both constraints evaluate $p\_not = 1.0 - truth$. In Belnap coordinates, $truth = 0.5(e_{\text{pos}} - e_{\text{neg}} + 1.0)$. For unmentioned candidate states ($Neither$: $e_{\text{pos}} = 0, e_{\text{neg}} = 0$), $truth = 0.5 \implies p\_not = 0.5$.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/belnap_transformer.py`
  - **Class:** `BelnapAttention`
  - **Lines 195–196:**
    ```python
    agg_pos = torch.matmul(a_pos, v_pos)  # (B, H, L_q, d_k)
    agg_neg = torch.matmul(a_neg, v_neg)  # (B, H, L_q, d_k)
    ```
    *Observation:* Value aggregation routes positive affinity $a_{\text{pos}}$ exclusively to $v_{\text{pos}}$, and negative affinity $a_{\text{neg}}$ exclusively to $v_{\text{neg}}$. Cross-polarity terms ($a_{\text{pos}} V_-$ and $a_{\text{neg}} V_+$) are omitted.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/engine.py` vs `belnap_transformer.py`
  - **Class:** `BelnapDecisionHead` in `engine.py` (Lines 96, 123, 129):
    ```python
    # num_choices == 1:
    logits = raw_pos - raw_neg

    # candidate_states is not None:
    logits = pos_affinity - neg_affinity

    # candidate_states is None:
    logits = raw_pos - raw_neg
    ```
  - **Class:** `BelnapDecisionTransformer` in `belnap_transformer.py` (Line 433):
    ```python
    logits = (choice_pos - choice_neg) * knowledge
    ```
    *Observation:* `BelnapDecisionHead` uses raw differences without knowledge gating. A candidate with $raw_{\text{pos}} = -8.0, raw_{\text{neg}} = -12.0$ has knowledge $k \approx 0.0001$, yet its logit is $(-8.0) - (-12.0) = +4.0$.

---

### R4. Bilinear Candidate-Query Alignment & Negation Equivariance
- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/engine.py`
  - **Class:** `YodaDecisionEngine`, `BelnapDecisionHead`
  - **Lines 398–400:**
    ```python
    if is_independent:
        head = self.scalar_head if self.num_choices != 1 else self.decision_head
        final_out = head(final_state)
    ```
    *Observation:* `head(final_state)` is called without passing `candidate_states=k_state`.
  - **Lines 91–109:**
    ```python
    if self.num_choices == 1:
        raw_pos = self.w_pos(pooled_pos).squeeze(-1)
        raw_neg = self.w_neg(pooled_neg).squeeze(-1)
        choice_pos = torch.sigmoid(raw_pos)
        choice_neg = torch.sigmoid(raw_neg)
        logits = raw_pos - raw_neg
        choice_state = BelnapState(e_pos=choice_pos, e_neg=choice_neg)
        truth = choice_state.truth
        knowledge = choice_state.knowledge
        choice = (logits > 0).long()
        return {
            "logits": logits,
            "truth": truth,
            "knowledge": knowledge,
            "choice": choice,
            "choice_pos": choice_pos,
            "choice_neg": choice_neg,
        }
    ```
    *Observation:* When `num_choices == 1`, `BelnapDecisionHead` unconditionally returns linear scoring on `pooled_pos` and `pooled_neg`, ignoring candidate probe states.

- **File:** `/home/cianh/Projects/yoda/src/yoda/architecture/belnap_transformer.py`
  - **Class:** `BelnapFFN`
  - **Lines 242–248:**
    ```python
    h_mid = nn.functional.mish(self.w_1(x_res.e_pos) + self.w_2(x_res.e_neg))
    h_mid = self.dropout(h_mid)
    e_ffn_pos = torch.sigmoid(self.w_o1(h_mid))
    e_ffn_neg = torch.sigmoid(self.w_o2(h_mid))
    ```
    *Observation:* Linear projections of $e_{\text{pos}}$ and $e_{\text{neg}}$ are summed into a single bottleneck vector $h_{\text{mid}}$.

---

### R5. Context Encoding Deduplication & Hyperparameter Sweep Realignment
- **File:** `/home/cianh/Projects/yoda/src/yoda/data/dataset.py` & `src/yoda/architecture/engine.py`
  - **Lines 279–285 (`dataset.py`):**
    ```python
    for c_idx, c_str in enumerate(active_c):
        cand_queries.append(q)
        cand_states.append(s)
        candidates.append(str(c_str))
        cand_labels.append(1.0 if c_idx == t_idx else 0.0)
        cand_group_ids.append(b_idx)
        cand_task_scalars.append(task_sc)
    ```
  - **Lines 363, 370 (`engine.py`):**
    ```python
    q_emb = self.text_encoder(queries)
    ...
    c_emb = self.state_encoder(states)
    ```
    *Observation:* If batch has $B$ samples with $K$ choices, `queries` and `states` contain $B \times K$ items ($K$ identical copies per sample). `self.text_encoder` and `self.state_encoder` execute the transformer backbone $K$ times on identical inputs.
- **File:** `experiments/optuna_study.py` (Lines 46–56) & `experiments/optuna_study_narrow.py` (Lines 33, 79):
  *Observation:* Search spaces allow `residual_weight` in $[0.1, 0.9]$ (or locked at $0.6356$), and `conflation_weight` up to $0.50$, directly exploring known gradient-collapse parameter regimes.

---

## 2. Logic Chain

1. **Diagnosis of the 36.5% Epoch 1 Gradient Collapse (Root Causes R1 & R2):**
   - *Premise 1 (Sigmoid-Jacobian minimization):* In `src/yoda/nesy/ltn.py:251-253`, the decisiveness loss term is $L_{\text{dec}} = p(1 - p) + q(1 - q)$, where $p = \sigma(z_{\text{pos}})$ and $q = \sigma(z_{\text{neg}})$.
   - *Premise 2 (Derivative identity):* By calculus, $\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z))$. Hence, $L_{\text{dec}} \equiv \frac{\partial p}{\partial z_{\text{pos}}} + \frac{\partial q}{\partial z_{\text{neg}}}$.
   - *Inference 2.1:* Minimizing $L_{\text{dec}}$ directly drives the gradient of the decision head activation to zero. At initialization ($z \approx 0 \implies p \approx 0.5$), $L_{\text{dec}}$ is maximized ($0.25 + 0.25 = 0.5$). The optimizer rapidly pushes weights and biases to large values ($|z| \ge 5.0$) so $p \to \{0, 1\}$.
   - *Inference 2.2:* The moment $p(1 - p) \to 0$, backpropagation gradients $\frac{\partial \text{loss}}{\partial z} \to 0$, permanently terminating gradient flow to upstream reasoning layers and NeSy constraints.
   - *Premise 3 (Unmasked tokens & scale explosion):* `src/yoda/architecture/encoders.py:70-81` discards `inputs["attention_mask"]`, passing `mask=None` into `BelnapMultiheadPooledAttention`. In variable-length batches, padding tokens ($a_{\text{pos}} \approx 0.375$) outnumber text tokens (up to 19:1). In `src/yoda/architecture/belnap_transformer.py:195-203`, $\text{agg} = A \cdot V$ sums over $L_{\text{kv}}$ without normalization, scaling linearly with length and forcing `out_pos = torch.sigmoid(self.w_out_pos(agg_pos))` into hard saturation ($0$ or $1$) on Step 1.
   - *Premise 4 (Skip highway attenuation):* In `src/yoda/architecture/belnap_transformer.py:289-301`, each convex join attenuates skip gradients by $(1 - \alpha)$. With 2 layers (6 joins) and default $\alpha = 0.5$, residual gradient retention is $(0.5)^6 = 0.0156$ (1.56%). Chained unnormalized sigmoids concurrently scale gradients by $(\le 0.25)^5 \le 0.00097$.
   - *Conclusion 1:* Upstream reasoning layers (`self.reasoning_layers[0]`) receive vanishing gradients ($< 1\text{e-}5$) within Epoch 1, freezing weights at initialization and collapsing accuracy to uniform random guess ($36.5\%$).

2. **Diagnosis of the 50.0% Validation Accuracy Ceiling (Root Causes R3 & R4):**
   - *Premise 5 (Additive context cancellation):* In `src/yoda/architecture/engine.py:398-400`, `is_independent = True` evaluates candidates unrolled via `self.scalar_head(final_state)`. `final_state` is formed by convex residual combination:
     $$\text{final\_state}_i = (1 - \alpha) q_{\text{ctx}} + \alpha x_{\text{cand}, i}$$
   - *Premise 6 (Linear projection):* Because `self.scalar_head` has `num_choices == 1`, lines 91–109 compute $z_i = (1 - \alpha) w^\top q_{\text{ctx}} + \alpha w^\top x_{\text{cand}, i}$.
   - *Inference 6.1:* In `FocalLoss` and `MarginLoss` (`src/yoda/training/losses.py:69, 153`), candidate comparisons within the same question group evaluate:
     $$z_y - z_j = \alpha w^\top (x_{\text{cand}, y} - x_{\text{cand}, j})$$
     **The context-query term $(1 - \alpha) w^\top q_{\text{ctx}}$ is identical across candidates in the group and cancels out identically.**
   - *Conclusion 2:* The decision head cannot condition its candidate ranking on query-context matching. The model can only assess whether a candidate is generically plausible in isolation, capping validation accuracy at 50% (binary coin toss on candidate plausibility).

3. **Diagnosis of the Unmentioned Candidate Penalty (Root Cause R3):**
   - *Premise 7 (Truth coordinate formulation):* In `src/yoda/nesy/ltn.py:65, 85`, `_compute_xor` and `_compute_or` maximize terms involving $p\_not = 1.0 - truth$.
   - *Premise 8 (Belnap coordinate algebra):* $truth = \frac{e_{\text{pos}} - e_{\text{neg}} + 1.0}{2}$. An unmentioned candidate is at $Neither$ ($e_{\text{pos}} = 0, e_{\text{neg}} = 0$), so $truth = 0.5$.
   - *Inference 8.1:* $p\_not = 1.0 - 0.5 = 0.5$. For every unchosen candidate at $Neither$, XOR evaluates $\prod_{j \neq i} (1 - t_j)$, imposing a penalty of $-\log(0.5) \approx 0.693$ per unchosen candidate.
   - *Conclusion 3:* To minimize constraint loss, the network is forced to fabricate false negative refutation ($e_{\text{neg}} \to 1.0$) for unmentioned options, corrupting paraconsistent epistemic calibration.

4. **Diagnosis of Redundant Context Encoding (Root Cause R5):**
   - *Premise 9 (Dataset unrolling):* `src/yoda/data/dataset.py:279-285` unrolls $B$ questions with $K$ choices into $B \times K$ flattened samples.
   - *Premise 10 (Encoder execution):* `src/yoda/architecture/engine.py:363, 370` executes `TextEncoder` on `queries` ($B \times K$) and `SymbolicStateEncoder` on `states` ($B \times K$).
   - *Conclusion 4:* For $K \in [3, 5]$, between 66% and 80% of all backbone encoder passes are duplicate computations on identical strings, creating a severe throughput bottleneck.

---

## 3. Caveats

1. **Read-Only Scope:** This investigation is strictly read-only; no code modifications have been applied to `src/yoda/`, `experiments/`, or `tests/`.
2. **PyTorch Lightning vs YodaTrainer Symmetry:** While `YodaTrainer` (`src/yoda/training/trainer.py`) is the primary training loop, `YodaLightningAdapter` (`src/yoda/training/lightning.py`) also exists and duplicates loss routing logic. Changes to `engine.py` and `ltn.py` must maintain compatibility with both harnesses.
3. **Optuna Historical Databases:** Existing SQLite databases in `data/optuna.db` and `data/optuna_narrow.db` contain trials from pre-hardened configurations. Realigned sweeps should write to new study names or databases.
4. **No caveats** regarding the mathematical identification of the root causes: the exact lines and mechanisms match the empirical failure modes identified in `.docs/plan.md`.

---

## 4. Conclusion

The 36.5% Epoch 1 gradient collapse and the 50.0% validation accuracy ceiling are caused by four distinct architectural defects across five modules:

| Requirement | Target Modules | Primary Defect Mechanism | Target Resolution |
|---|---|---|---|
| **R1** | `encoders.py`, `attention.py`, `belnap_transformer.py`, `engine.py` | Discarded tokenizer padding masks; unnormalized $\text{agg} = A \cdot V$ blowing up sigmoid inputs linearly with sequence length | Return mask from `TextEncoder`, pass to `BelnapMPA` & `BelnapAttention`, mask padding keys, normalize by active sequence mass $\text{denom} = \text{mask}.\text{sum}()$ |
| **R2** | `ltn.py`, `belnap_transformer.py`, `train_yoda.py`, `optuna_study_narrow.py` | Quadratic decisiveness penalty $p(1-p)$ minimizing sigmoid Jacobian; unnormalized chained sigmoids; $(0.5)^6$ residual attenuation | Replace $p(1-p)$ with paraconsistent contradiction penalty $e_{\text{pos}} \times e_{\text{neg}}$ with linear warmup; add pre-sigmoid `LayerNorm`; lower default $\alpha$ to $0.15$ |
| **R3** | `ltn.py`, `belnap_transformer.py`, `engine.py`, `losses.py` | XOR/OR evaluating truth coordinate $t=0.5$ on $Neither$, penalizing ignorance by $0.693$; isolated parallel value routing dropping refutation; ungated head logits producing high confidence at $k \to 0$ | Reformulate XOR/OR on positive support ($1 - \text{choice\_pos}$); implement $2 \times 2$ coupled value routing; gate decision head logits by candidate knowledge ($k$) |
| **R4** | `engine.py`, `belnap_transformer.py` | Scalar head linear projection causing additive context cancellation $(1-\alpha)w^\top q_{\text{ctx}}$ in ranking losses; single vector bottleneck in `BelnapFFN` breaking negation equivariance | Supply `candidate_states=k_state` to `scalar_head` and evaluate bilinear Belnap inner product when `num_choices == 1`; refactor `BelnapFFN` to dual cross-gated channels |
| **R5** | `engine.py`, `dataset.py`, `optuna.py`, `optuna_study.py`, `optuna_study_narrow.py` | $B \times K$ unrolled encoder execution repeating query/state forward passes $K$ times; search bounds stuck in collapsed regimes | Cache unique query/state representations per group before cross-attention; realign search bounds to validated stable ranges |

---

## 5. Verification Method

To independently verify these findings:

1. **Verify Test Suite & Environment:**
   ```bash
   uv run pytest
   uv run ruff check .
   ```
   Both commands must exit 0 on the existing codebase (181 passed).

2. **Inspect Identified Lines Directly:**
   - Run `grep -n "decisiveness_terms = choice_pos" src/yoda/nesy/ltn.py` -> Confirms lines 252, 272.
   - Run `grep -n "inputs = self.tokenizer" src/yoda/architecture/encoders.py` -> Confirms line 70, showing absence of mask propagation.
   - Run `grep -n "if self.num_choices == 1:" src/yoda/architecture/engine.py` -> Confirms line 91, showing early return before candidate bilinear alignment.
   - Run `grep -n "p_not = 1.0 - truth" src/yoda/nesy/ltn.py` -> Confirms lines 73, 90, showing XOR/OR calculated on truth coordinates.
   - Run `grep -n "h_mid = nn.functional.mish" src/yoda/architecture/belnap_transformer.py` -> Confirms line 243, showing positive and negative evidence summation.

3. **Invalidation Conditions:**
   - If `TextEncoder.forward` is found to already return `inputs["attention_mask"]` to callers.
   - If `LTNConstraintLoss.forward` already uses $e_{\text{pos}} \times e_{\text{neg}}$ rather than $p(1 - p)$.
   - If `BelnapDecisionHead.forward` already computes bilinear inner products when `num_choices == 1`.
   - If unmentioned candidates at $Neither$ $(0, 0)$ do not produce $truth = 0.5$ in `BelnapState`.
