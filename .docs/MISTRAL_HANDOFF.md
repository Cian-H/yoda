# Mistral Handoff: Yoda System 1 Architectural Hardening (Milestones 3 & 4)

> **Context:** Milestone 1 (R1 & R2) and Milestone 2 (R3) have been completed and verified by the Google Antigravity Teamwork system.  
> **Status:** Handed off to Mistral CLI (`vibe` or equivalent) for **Milestone 3 (Requirement R4)** and **Milestone 4 (Requirement R5)** to optimize token and computational credits.

---

## 1. Current State & Completed Invariants

The codebase currently has **325 passing unit tests** (`uv run pytest tests/ -k "not e2e"`).

### Milestone 1 Completed (R1 & R2)
- **Token Mask Propagation & Normalization (R1):** Preserved `attention_mask` through `TextEncoder`, `BelnapMPA`, and `BelnapAttention`. Normalized sigmoid attention aggregation by active sequence mass $\sum \text{mask}$ to prevent sequence-length explosion.
- **Contradiction Loss & Pre-Activation Normalization (R2):** Replaced quadratic decisiveness Jacobian penalty with paraconsistent contradiction loss ($e^+ \cdot e^-$) with linear warmup in `src/yoda/nesy/ltn.py`. Added pre-sigmoid `LayerNorm` in attention and FFN; lowered residual weight to $\alpha = 0.15$ in `src/yoda/architecture/belnap_transformer.py`.

### Milestone 2 Completed (R3)
- **Positive Support NeSy Reformulation:** Reformulated single-choice exclusivity (`_compute_xor`) and existence (`_compute_or`) in `src/yoda/nesy/ltn.py` on positive evidence support $e^+$, eliminating the false $-\log(1 - 0.5) = 0.693$ penalty on unmentioned $Neither$ $(0, 0)$ states.
- **Coupled $2 \times 2$ Bilattice Routing:** Implemented coupled evidence aggregation in `BelnapAttention`:
  $$\text{agg}_{\text{pos}} = a_{\text{pos}} V_+ + a_{\text{neg}} V_-, \qquad \text{agg}_{\text{neg}} = a_{\text{pos}} V_- + a_{\text{neg}} V_+$$
- **Epistemic Logit Gating:** Synchronized decision head logits with knowledge coordinate $k$: $\text{logits} = \text{raw\_diff} \times \text{knowledge}$. Zero-knowledge inputs ($k \to 0$) are suppressed to uniform probability with zero confidence.

---

## 2. Tasks Remaining for Mistral

### Task 1: Milestone 3 — Requirement R4 (Break the 50% Accuracy Ceiling)
- **TODO Slug:** `.docs/todos/bilinear-candidate-query-affinity-in-scalar-head.md`
- **Impacted Files:** `src/yoda/architecture/engine.py`, `tests/test_independent_eval.py`, `tests/e2e/test_accuracy_ceiling_gate.py`
- **Core Diagnosis:**
  In independent choice evaluation mode (`num_choices == 1`), `scalar_head` evaluated $z_i = w^\top \text{pooled}(final\_state_i)$. Because $final\_state_i = (1 - \alpha) q_{\text{ctx}} + \alpha x_{\text{cand}, i}$, comparing candidates in `FocalLoss` ($z_y - z_j$) caused the context term $(1 - \alpha) w^\top q_{\text{ctx}}$ to **cancel out identically**. The model only ranked candidates on isolated plausibility, capping accuracy at 50%.
- **Action Required:**
  1. In `BelnapDecisionHead.forward` (`src/yoda/architecture/engine.py`), accept `candidate_states: BelnapState | None = None`.
  2. In `YodaDecisionEngine.forward` (`src/yoda/architecture/engine.py`), pass `candidate_states=k_state` when invoking `head(final_state, candidate_states=k_state)`.
  3. When `self.num_choices == 1` and `candidate_states` is supplied, pool candidate probes to `[B, 1, D]` and compute the bilinear Belnap inner product:
     $$\text{pos\_affinity} = \frac{q_{\text{pos}} c_{\text{pos}} + q_{\text{neg}} c_{\text{neg}}}{\sqrt{d}}, \quad \text{neg\_affinity} = \frac{q_{\text{pos}} c_{\text{neg}} + q_{\text{neg}} c_{\text{pos}}}{\sqrt{d}}$$
     Set logits to $(\text{pos\_affinity} - \text{neg\_affinity}) \times \text{knowledge}$.
  4. Verify that `tests/e2e/test_accuracy_ceiling_gate.py` passes (> 60% accuracy).

### Task 2: Milestone 3 — Requirement R4 (Dual-Channel Negation-Equivariant FFN)
- **TODO Slug:** `.docs/todos/dual-channel-negation-equivariant-ffn.md`
- **Impacted Files:** `src/yoda/architecture/belnap_transformer.py`, `tests/test_belnap_transformer.py`
- **Action Required:**
  In `BelnapFFN.forward`, avoid summing $w_1(e_{\text{pos}}) + w_2(e_{\text{neg}})$ into a single vector bottleneck. Maintain dual-channel positive and negative projections ($h_{\text{pos}}, h_{\text{neg}}$) with cross-channel gating to preserve strict $\mathcal{B}_4$ negation equivariance ($x.\text{negate}()$).

### Task 3: Milestone 4 — Requirement R5 (Context Encoding Deduplication)
- **TODO Slug:** `.docs/todos/independent-mode-context-encoding-cache.md`
- **Impacted Files:** `src/yoda/architecture/engine.py`, `tests/e2e/test_throughput_gate.py`
- **Action Required:**
  When `is_independent = True`, $B$ questions with $K$ choices are unrolled into $B \times K$ samples. Cache unique `(query, state)` encodings per `group_id` before Phase 3 cross-attention, eliminating redundant encoder forward passes. Verify that `tests/e2e/test_throughput_gate.py` reports $\ge 2.0\times$ throughput speedup.

### Task 4: Milestone 4 — Requirement R5 (Optuna Sweep Realignment)
- **Impacted Files:** `src/yoda/hyperopt/optuna.py`, `experiments/run_optuna_study.py`, `experiments/optuna_study_narrow.py`
- **Action Required:**
  Realign Optuna search bounds:
  - `conflation_weight`: $[0.01, 0.25]$ (log scale)
  - `residual_weight`: $[0.10, 0.25]$ (centered at $0.15$)
  - `decisiveness_weight`: $[0.001, 0.05]$ with warmup
  - Learning rate: $[1\text{e-}5, 5\text{e-}4]$

---

## 3. Verification Protocol for Mistral

Run the test suite at any point:
```bash
# Unit & regression suite (should be 100% passing)
uv run pytest tests/ -k "not e2e"

# E2E acceptance gates
uv run pytest tests/e2e/

# Lint and style checks
uv run ruff check src/ tests/ experiments/
```
