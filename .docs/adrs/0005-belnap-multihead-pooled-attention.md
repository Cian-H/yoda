# 5. Belnap Multihead Pooled Attention (Belnap-MPA)

- **Status**: Accepted
- **Date**: 2026-10-01

## Context

`MultiheadPooledAttention` (ADR 0002) aggregates variable-length sequences into fixed-dimensional latent vectors via learnable query tokens. When applied to structured decision payloads (`DecisionPayload`), which contain variable sets of constraints, rules, symbolic assertions, and historical events, standard attention exhibits two severe vulnerabilities:

1. **Softmax Attention Sinks on Vacuous Inputs**: When constraints or context are empty or uninformative, softmax attention is mathematically forced to sum to 1 ($\sum_j \alpha_{ij} = 1$), distributing 100% of its attention mass across irrelevant or padding tokens and hallucinating non-existent context relevance.
2. **Contradiction Cancellation**: When rules contradict each other (e.g. Rule A yields $+v$, Rule B yields $-v$), linear summation $+v + (-v) \approx 0$ collapses the conflict into a false neutral, destroying the signal that a logical contradiction occurred.
3. **Epistemic Indistinguishability**: The aggregated latent vector cannot distinguish between "no evidence was found" ($\mathbf{N}$) and "equally balanced strong opposing evidence" ($\mathbf{B}$).

## Decision

We implement `BelnapMultiheadPooledAttention` in `src/yoda/architecture/attention.py`, combining multihead cross-attention query pooling with four-valued Belnap bilattice logic:

1. **Learnable Epistemic Query Probes**:
   - Parameterizes $T$ learnable query tokens as dual evidence probes $(Q^+, Q^-) \in [0, 1]^{T \times D}$ bounded continuously via sigmoid.
   - Each query probe specializes in detecting specific logical conditions (e.g. constraint satisfaction, safety violation, intent match).

2. **Bipolar Cross-Attention Pooling**:
   - Leverages `BelnapAttention` to cross-attend between the query probes and incoming key-value tokens.
   - Computes separate coherent support ($S^+$) and refutation ($S^-$) affinity matrices.
   - Gates attention weights by source token knowledge $k_{\text{src}} = \frac{K^+ + K^-}{2}$.

3. **Zero-Knowledge Suppression & Contradiction Preservation**:
   - If input tokens have zero knowledge ($k \to 0$), cross-attention output evidence is strictly suppressed ($e^+, e^- \le 0.05 \implies \mathbf{N}$), eliminating softmax attention sinks.
   - If input tokens contain opposing evidence, positive and negative evidence are accumulated into separate channels ($e^+ \gg 0, e^- \gg 0 \implies \mathbf{B}$), preserving contradiction.

4. **Flexible Modality Adapters & Dual Readout**:
   - Accepts either native `BelnapState` objects or raw feature tensors $\mathbf{x} \in \mathbb{R}^{B \times S \times D}$ via a bounded input projection.
   - Emits a dual readout: a fixed dense latent vector $\mathbf{h} \in \mathbb{R}^{B \times D}$ for downstream neural fusion, and the structured `BelnapState` of shape $(B, T, D)$ exposing explicit truth and knowledge coordinates per query probe for neuro-symbolic rule auditing.

## Consequences

- Provides a dedicated, epistemically sound ingestion pathway for variable-length constraints and symbolic state in `DecisionPayload`.
- Eliminates attention sink hallucinations on sparse or missing context fields.
- Preserves full end-to-end differentiability and sub-millisecond pooling latency.
- Verified by unit, property, and gradient tests in `tests/test_belnap_mpa.py`.
