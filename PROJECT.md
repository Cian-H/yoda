# Project: Yoda Belnap System 1 Engine Architectural Hardening

## Architecture
Yoda is a Neuro-Symbolic (NeSy) System 1 decision engine based on the four-valued Belnap-Dunn bilattice $\mathcal{B}_4 = \{\mathbf{T}, \mathbf{F}, \mathbf{B}, \mathbf{N}\}$.
Representations propagate through bounded hypercubes $[0, 1]^d \times [0, 1]^d$ with decoupled positive evidence ($e_{\text{pos}}$) and negative evidence ($e_{\text{neg}}$).
- **Truth Coordinate:** $t = \frac{e_{\text{pos}} - e_{\text{neg}} + 1}{2} \in [0, 1]$
- **Knowledge Coordinate:** $k = \frac{e_{\text{pos}} + e_{\text{neg}}}{2} \in [0, 1]$
- **Core Modules:**
  - `src/yoda/architecture/encoders.py`: Text, Symbolic State, and Constraint encoders.
  - `src/yoda/architecture/attention.py` & `belnap_transformer.py`: Multihead Pooled Attention (MPA), BelnapAttention, BelnapFFN, BelnapTransformerBlock.
  - `src/yoda/architecture/engine.py`: `YodaDecisionEngine` & `BelnapDecisionHead`.
  - `src/yoda/nesy/ltn.py`: Logic Tensor Network (LTN) constraints, Łukasiewicz t-norm, XOR, OR, Contradiction losses.
  - `src/yoda/training/`: `losses.py`, `trainer.py`, `lightning.py`.
  - `src/yoda/hyperopt/`: Optuna search spaces and CLI runners.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Tokenizer Mask Propagation | `TextEncoder` preserves padding mask from tokenizer | M1 | Survey (`encoders.py:70`) |
| 2 | Attention Padding Masking | `BelnapAttention` masks padding keys ($s \to -1\text{e}9$) | M1 | Survey (`belnap_transformer.py:180`) |
| 3 | Active Mass Normalization | Scale-normalize attention aggregation by active sequence tokens | M1 | Survey (`plan.md:124`) |
| 4 | Contradiction Loss | Replace quadratic $p(1-p)$ with paraconsistent $e^+ \cdot e^-$ | M1 | Survey (`ltn.py:251`) |
| 5 | Decisiveness Warmup | Linear warmup schedule from 0 to $\lambda_{\text{target}}$ | M1 | Survey (`plan.md:129`) |
| 6 | Pre-Sigmoid LayerNorm | Insert LayerNorm before output sigmoids in attention and FFN | M1 | Survey (`belnap_transformer.py:202`) |
| 7 | Residual Highway $\alpha$ | Recalibrate default $\alpha$ to $0.15$ for $>35\%$ gradient retention | M1 | Survey (`belnap_transformer.py:289`) |
| 8 | Positive Support XOR/OR | Reformulate XOR/OR using $e^+$ to preserve unmentioned $Neither$ | M2 | Survey (`ltn.py:65`) |
| 9 | Coupled Value Routing | $2 \times 2$ bilattice aggregation preserving refutation signals | M2 | Survey (`belnap_transformer.py:195`) |
| 10 | Epistemic Logit Gating | Synchronize decision head logits with knowledge coordinate $k$ | M2 | Survey (`engine.py:96`) |
| 11 | Bilinear Alignment | Bilinear Belnap inner product scoring in decision head | M3 | Survey (`engine.py:400`) |
| 12 | Dual-Channel FFN | Negation-equivariant FFN with cross-polarity gating | M3 | Survey (`belnap_transformer.py:243`) |
| 13 | Context Deduplication | Cache unique query/state encodings per group across candidates | M4 | Survey (`engine.py:363`) |
| 14 | Hyperopt Realignment | Realign Optuna search bounds to safe numerical regimes | M4 | Survey (`optuna_study.py:46`) |
| 15 | E2E Requirements Test Suite | Tiers 1-4 acceptance tests (Gradient, Acc, Neither, Logits, Perf) | E2E | Survey (`ORIGINAL_REQUEST.md`) |
| 16 | Adversarial Hardening | Tier 5 white-box coverage hardening | Final | Survey (`plan.md`) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Suite | Design and implement complete opaque-box E2E test suites (Tiers 1-4) | none | DONE |
| M1 | Phase 1 Architectural Hardening | R1 (Mask propagation, padding mask, scale norm) + R2 (Contradiction loss, warmup, prenorm LayerNorm, $\alpha=0.15$) | none | DONE |
| M2 | Phase 2 NeSy Reformulation | R3 (Positive support XOR/OR, coupled value routing, epistemic logit gating) | M1 | PLANNED |
| M3 | Phase 3 Bilinear Head & Equivariance | R4 (Bilinear Belnap inner product scoring, dual-channel negation-equivariant FFN) | M2 | PLANNED |
| M4 | Phase 4 Context Deduplication & Sweep | R5 (Context encoding deduplication cache, Optuna sweep bounds realignment) | M3 | PLANNED |
| M5 | Final Acceptance & Adversarial Hardening | Phase 1 pass 100% E2E tests + Phase 2 adversarial coverage hardening | M0, M4 | PLANNED |

## Interface Contracts

### `TextEncoder` ↔ `BelnapMultiheadPooledAttention` / `BelnapAttention`
- `TextEncoder.forward(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor`: Expose `forward_with_mask` returning `(embeddings, attention_mask)` where `attention_mask` is shape `(B, L)` boolean or float tensor.
- `BelnapAttention.forward(q, kv, mask=None)`: Accepts optional `mask` shape `(B, L_kv)` or `(B, 1, 1, L_kv)`. Applies `masked_fill(~mask, -1e9)` to affinity scores and normalizes aggregation denominator by `mask.sum().clamp(min=1.0)`.

### `LTNConstraintLoss` ↔ `YodaDecisionEngine` & `YodaTrainer`
- `LTNConstraintLoss.forward(choice_pos, choice_neg, ...)`: Contradiction penalty computed as `(choice_pos * choice_neg).mean() * self.current_decisiveness_weight`.
- `LTNConstraintLoss.step_epoch(epoch)`: Updates `current_decisiveness_weight` according to linear warmup schedule.
- `_compute_xor(pos_support)` & `_compute_or(pos_support)`: Evaluates positive support, guaranteeing that $Neither$ $(e^+=0, e^-=0)$ incurs 0 loss.

### `YodaDecisionEngine` ↔ `BelnapDecisionHead`
- `BelnapDecisionHead.forward(final_state, candidate_states=None)`:
  - When `candidate_states` is supplied and `num_choices == 1`: computes bilinear inner product between context and candidate options.
  - Logits gated by knowledge: `logits = (affinity_diff) * knowledge` or `(raw_pos - raw_neg) * knowledge`.

## Code Layout
- `src/yoda/architecture/encoders.py`: Tokenizer mask extraction and propagation.
- `src/yoda/architecture/attention.py`: Masked multihead pooled attention.
- `src/yoda/architecture/belnap_transformer.py`: `BelnapAttention` (masking, scale norm, coupled routing, LayerNorm), `BelnapFFN` (dual-channel, LayerNorm), `BelnapTransformerBlock` ($\alpha=0.15$).
- `src/yoda/architecture/engine.py`: `BelnapDecisionHead` (bilinear scoring, knowledge gating), `YodaDecisionEngine` (context caching).
- `src/yoda/nesy/ltn.py`: `LTNConstraintLoss` (contradiction penalty, warmup schedule, positive support XOR/OR).
- `src/yoda/data/dataset.py`: Multi-choice grouping and unrolling.
- `src/yoda/hyperopt/optuna.py` & `experiments/optuna_study.py`: Parameter bounds.
- `tests/`: All unit, regression, integration, and E2E test suites.
