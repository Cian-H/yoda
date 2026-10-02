# 14. Native Full Embedding Dimension Directly into Belnap MPA

- **Status**: Accepted
- **Date**: 2026-10-02

## Context

Prior versions of `YodaDecisionEngine` defaulted `embed_dim` to 256. For standard sentence transformer backbones such as `BAAI/bge-small-en-v1.5` (native hidden dimension $d = 384$), this triggered `TextEncoder` to instantiate an intermediate `nn.Linear(384, 256)` adapter layer. 

Analysis identified two issues with this design:
1. **Redundant Linear Stacking**: `BelnapMultiheadPooledAttention` (MPA) already incorporates an internal linear bridge `self.input_proj = nn.Linear(embed_dim, 2 * embed_dim)` to project continuous token embeddings into positive support and negative refutation components. Stacking `nn.Linear(384, 256)` immediately before `nn.Linear(256, 512)` without an intervening activation function is mathematically redundant.
2. **Loss of High-Frequency Semantic Distinctions**: Forcing an intermediate bottleneck drops ~33% of the latent embedding space prior to attention pooling, discarding fine-grained symbolic invariants, syntactic negations, and subtle candidate differences that live in the full embedding representation.

## Decision

1. **Native Backbone Dimensions**:
   - `YodaDecisionEngine` now defaults `embed_dim: int | None = None`.
   - When `embed_dim is None`, `TextEncoder` initializes with `target_dim=None`, defaulting its internal adapter to `nn.Identity()`.
   - The engine automatically adopts `self.embed_dim = self.text_encoder.dim` (e.g. 384 for `BAAI/bge-small-en-v1.5`, or 256 for `dummy`).
2. **Direct Flow into MPA**:
   - Full native sequence embeddings flow directly from the backbone into the MPA's `input_proj`, mapping $d_{\text{native}} \to 2 d_{\text{native}}$ with zero intermediate compression.
   - Attention heads dynamically divide `embed_dim` cleanly (e.g., $384 / 4 = 96$ or $384 / 6 = 64$).

```mermaid
flowchart LR
    subgraph Previous Architecture (Bottlenecked)
        B1[Text Backbone: 384] -->|Linear Adapter| AD[Bottleneck: 256]
        AD -->|Input Proj| MPA1[Belnap MPA: 2 x 256]
    end

    subgraph Native Full Embedding Architecture
        B2[Text Backbone: 384] -->|Identity| MPA2[Belnap MPA: 2 x 384]
    end
```

## Consequences

- **Preserved Feature Space**: All 384 dimensions of semantic and syntactic features are retained for attention queries and candidate evaluation.
- **Cleaner Parameter Graph**: Eliminates an unnecessary linear projection layer from the text encoder.
- **Backward Compatibility**: If a user explicitly specifies `embed_dim: int`, the model continues to support custom target dimensions via `TextEncoder`'s linear adapter.
