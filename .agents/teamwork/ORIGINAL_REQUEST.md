# Original User Request

## 2026-10-13T19:51:51Z

Implement the complete 4-phase architectural hardening and plateau resolution roadmap for Yoda (`.docs/plan.md`), eliminating the 36.5% Epoch 1 gradient collapse and the 50% validation accuracy ceiling across the Belnap System 1 decision engine.

Working directory: `/home/cianh/Projects/yoda`
Integrity mode: demo

Reference material:
- Plan & Mathematical Diagnoses: `.docs/plan.md`
- Work Packages: `.docs/todos/`
- Architecture Decision Records: `.docs/adrs/`

## Requirements

### R1. Attention Mask Propagation & Scale Normalization (Phase 1)
Ensure variable-length token sequences preserve padding masks through feature encoders into multihead pooled attention. Sigmoid attention affinities must mask out padding keys, and evidence aggregation must be scale-normalized by active sequence mass to keep intermediate activations bounded in $[0, 1]$ independent of sequence length.

### R2. Paraconsistent Contradiction Loss & Pre-Activation Normalization (Phase 1)
Eliminate Jacobian zeroing during early training by replacing the quadratic decisiveness penalty with a paraconsistent contradiction penalty ($Both$: $\text{choice\_pos} \times \text{choice\_neg}$) backed by linear weight warmup. Center pre-sigmoid activations via normalization layers prior to bounding activations in reasoning blocks, and recalibrate residual interpolation weights to preserve gradient highways through multi-stage reasoning.

### R3. NeSy Constraint Reformulation & Epistemic Logit Synchronization (Phase 2)
Reformulate multi-choice exclusivity constraints (XOR/OR) to evaluate positive support rather than truth coordinates, ensuring unmentioned candidate options at $Neither$ $(0, 0)$ do not incur false penalty. Implement coupled cross-polarity evidence routing in bilattice attention so refutation signals are preserved under coherent query alignment. Synchronize decision head logits with epistemic knowledge coordinates so zero-knowledge states cannot produce high classification confidence.

### R4. Bilinear Candidate-Query Alignment & Negation Equivariance (Phase 3)
Eliminate additive context cancellation during independent candidate evaluation by implementing explicit bilinear Belnap inner product scoring in the decision head, conditioning candidate ranking directly on context-query interaction. Refactor reasoning feed-forward layers to maintain dual-channel positive and negative projections with cross-gating to guarantee negation equivariance.

### R5. Context Encoding Deduplication & Hyperparameter Sweep Realignment (Phase 4)
Cache unique query and state encodings across unrolled candidate groups to remove redundant encoder forward passes. Update hyperparameter optimization search spaces and verification protocols to match the hardened architecture.

## Acceptance Criteria

### Test Suite & Code Verification
- [ ] All 181 existing tests in `tests/` pass via `uv run pytest`.
- [ ] New unit and regression tests added for attention masking, contradiction loss, coupled value routing, and bilinear head alignment all pass.
- [ ] Codebase passes lint and style validation via `uv run ruff check .`.

### Gradient & Training Health
- [ ] Average gradient norms across initial reasoning layers (`self.reasoning_layers[0]`) remain strictly positive ($> 1\text{e-}4$) throughout Epoch 1 backpropagation across 10 random initialization seeds.
- [ ] Multi-choice validation accuracy on independent choice evaluation benchmarks exceeds 60.0%, demonstrating resolution of the 50.0% ceiling.

### Paraconsistent Calibration
- [ ] Inputs containing unmentioned candidates retain epistemic coordinates $k < 0.20$ ($Neither$) without incurring constraint loss penalties.
- [ ] Candidates with low knowledge ($k \to 0$) do not produce high confidence classification logits.

### Performance & Throughput
- [ ] Independent candidate evaluation throughput increases by at least $2.0\times$ following context encoding deduplication.


## 2026-10-13T21:59:21Z

USER DIRECTIVE: Stop the project immediately upon concluding Milestone 2 (Phase 2 NeSy Reformulation & Epistemic Logits, R3). Do NOT dispatch Milestone 3 or Milestone 4. Wrap up, verify M2 gate, produce final handoff/summary, and terminate execution cleanly. The user will be handing off Milestone 3 and Milestone 4 to Mistral.


## Follow-up — 2026-10-13T21:59:25Z

USER DIRECTIVE: Stop the project immediately upon concluding Milestone 2 (Phase 2 NeSy Reformulation & Epistemic Logits, R3). Do NOT dispatch Milestone 3 or Milestone 4. Wrap up, verify M2 gate, produce final handoff/summary, and terminate execution cleanly. The user will be handing off Milestone 3 and Milestone 4 to Mistral.
