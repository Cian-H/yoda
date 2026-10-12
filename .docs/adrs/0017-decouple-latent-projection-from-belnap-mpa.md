# 17. Decouple Latent Projection from Belnap Multihead Pooled Attention

- **Status**: Accepted
- **Date**: 2026-10-12

## Context

ADR 0005 introduced `BelnapMultiheadPooledAttention` (Belnap-MPA) with a "dual readout" contract emitting both a dense 1D latent vector $\mathbf{h} \in \mathbb{R}^{B \times D}$ and a structured `BelnapState` of shape $(B, Q, D)$. The dense latent vector was produced by flattening the dual evidence probe vectors and projecting through an internal linear layer:
$$\mathbf{W}_{\text{proj}} \in \mathbb{R}^{(2 \cdot Q \cdot D) \times D}$$

As Yoda's architecture evolved (ADR 0004, ADR 0013, ADR 0014), the downstream reasoning core (`BelnapTransformerBlock`) and decision head (`BelnapDecisionHead`) transitioned into an end-to-end native Belnap pipeline that operates exclusively on `BelnapState`. Consequently, `YodaDecisionEngine.forward` discarded the dense latent vector (`_, q_state = self.q_mpa(q_emb)`).

When ADR 0014 adopted native BGE-large embedding dimensions ($D = 1024$) and hyperparameter sweeps scaled probe counts ($Q = 7$, $C = 14$, $K = 8$), `self.proj` scaled quadratically with $D$:
- `q_mpa`: $2 \times 7 \times 1024 \times 1024 = 14.68\text{M}$ parameters
- `c_mpa`: $2 \times 14 \times 1024 \times 1024 = 29.36\text{M}$ parameters
- `k_mpa`: $2 \times 8 \times 1024 \times 1024 = 16.78\text{M}$ parameters

This accumulated over $60\text{M}$ unutilized parameters consuming $240+\text{ MB}$ of dead optimizer state memory in GPU VRAM during training and inflating checkpoint sizes without contributing to the computation graph or receiving gradients.

## Decision

1. **Parameter Initialization Decoupling**:
   - `BelnapMultiheadPooledAttention.__init__` now accepts `include_projection: bool = True`.
   - When `include_projection=False`, `self.proj` is set to `None` and is not registered as a parameter or submodule.
   - Defaults to `True` for full backward compatibility with standalone usage and existing tests.

2. **Adaptive Forward Pass**:
   - `forward` adds a `return_latent: bool | None = None` parameter with strict `@overload` type annotations.
   - If `return_latent=False` (or when `return_latent is None` and `include_projection=False`), returns `BelnapState` directly without dual-evidence flattening or projection compute.
   - Emits explicit convenience helpers `forward_pooled` (returning `BelnapState`) and `forward_latent` (returning `torch.Tensor`).
   - Raises an explicit `RuntimeError` if `return_latent=True` is requested on an instance where `include_projection=False`.

3. **YodaDecisionEngine Integration**:
   - `YodaDecisionEngine` instantiates `q_mpa`, `c_mpa`, and `k_mpa` with `include_projection=False`.
   - `YodaDecisionEngine.forward` captures `BelnapState` directly without tuple unpack discards.

## Consequences

- **Zero Dead Parameters**: Eliminates $>60\text{M}$ idle parameters in `YodaDecisionEngine` under native BGE-large ($D = 1024$) embeddings.
- **Reduced Memory & Checkpoint Overhead**: Frees $>240\text{ MB}$ of optimizer state VRAM in AdamW and reduces model checkpoint size.
- **Preserved Backward Compatibility**: Standalone usages and existing unit tests expecting `(latent, pooled_state)` continue to function unchanged with `include_projection=True`.
