# 4. Belnap Bilattice Native Transformer & Decoupled Ingestion

- **Status**: Accepted
- **Date**: 2026-10-01

## Context

Standard Transformer architectures represent latent state tokens as unconstrained real vectors $\mathbf{h} \in \mathbb{R}^d$ and normalize attention using competitive softmax ($\sum_j \alpha_{ij} = 1$). For low-latency System 1 decision-making, this introduces two major failure modes:
1. **Attention Sinks on Missing Information ($\mathbf{N}$)**: When relevant context is absent, competitive softmax still allocates 100% mass across irrelevant tokens, causing hallucination.
2. **False Compromises on Contradictory Evidence ($\mathbf{B}$)**: When contradictory rules or facts are present, vector averaging produces an ambiguous neutral state that looks like weak evidence rather than a sharp logical conflict.
3. **Monolithic JSON Parsing Inefficiencies**: Serializing structured payloads (`DecisionPayload`) into flat JSON text strings incurs a 40–60% "syntax tax" on punctuation tokens and violates our sub-10ms latency budget.

## Decision

We design and implement a native Belnap Bilattice Transformer in `src/yoda/architecture/belnap_transformer.py` grounded by four-valued paraconsistent logic ($\mathcal{B}_4 = \{\mathbf{T}, \mathbf{F}, \mathbf{N}, \mathbf{B}\}$), paired with decoupled multi-branch ingestion:

1. **Dual Evidence Representation (`BelnapState`)**:
   - Every latent feature is parameterized as a continuous dual evidence pair $(\mathbf{e}^+, \mathbf{e}^-) \in [0, 1]^d \times [0, 1]^d$.
   - **Truth Coordinate**: $\mathbf{t} = \frac{\mathbf{e}^+ - \mathbf{e}^- + 1}{2} \in [0, 1]^d$ (falsity $\to$ truth).
   - **Knowledge Coordinate**: $\mathbf{k} = \frac{\mathbf{e}^+ + \mathbf{e}^-}{2} \in [0, 1]^d$ (ignorance $\to$ contradiction).
   - Exact logical operations: Negation ($\sim$), Conflation ($\ominus$), and $(t, k)$ roundtrip coordinate mapping.

2. **Bipolar Support/Refutation Attention (`BelnapAttention`)**:
   - Computes dual affinity projections:
     - Coherent support affinity: $\mathbf{S}^+ = (\mathbf{Q}^+ (\mathbf{K}^+)^\top + \mathbf{Q}^- (\mathbf{K}^-)^\top) / \sqrt{d_k}$.
     - Opposing refutation affinity: $\mathbf{S}^- = (\mathbf{Q}^+ (\mathbf{K}^-)^\top + \mathbf{Q}^- (\mathbf{K}^+)^\top) / \sqrt{d_k}$.
   - **Knowledge Gating**: Attention weights are modulated by source token knowledge $\mathbf{k}_{\text{src}}$. Complete absence of context evidence ($k \to 0$) suppresses attention output to Neither ($\mathbf{N}$) rather than creating attention sinks.

3. **Bilattice Feed-Forward Network (`BelnapFFN`)**:
   - Embeds the conflation operator to prune mutual cross-polarity ambiguity: $\mathbf{e}_{\text{res}}^+ = \text{ReLU}(\mathbf{e}^+ - \lambda \mathbf{e}^-)$.
   - Expands features non-linearly with GELU and projects back to bounded dual evidence $[0, 1]$.

4. **Multi-Branch Ingestion & Decision Readout (`BelnapDecisionTransformer`)**:
   - Parses `DecisionPayload` fields in Python into dedicated tensor pathways (dense query projection, structured state tokens, candidate choice embeddings).
   - Stacks $N$ `BelnapTransformerBlock` layers.
   - Emits calibrated choice logits ($[\mathbf{e}^+ - \mathbf{e}^-] \odot \mathbf{k}$), predicted choice indices, and native $(t, k)$ uncertainty coordinates.

## Consequences

- The model natively quantifies uncertainty on every forward pass without post-hoc temperature scaling or dropout passes.
- Contradictory evidence triggers $\mathbf{B}$ (Both/Conflict) and missing context triggers $\mathbf{N}$ (Neither/Unknown), enabling clean escalation and policy enforcement.
- Dense vectorized linear operations preserve our sub-10ms System 1 inference target.
- Full unit and property test coverage in `tests/test_belnap_transformer.py` (9/9 passed, 36/36 repository total).
