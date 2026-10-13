# ADR 0020: Dual-Channel Negation-Equivariant Feed-Forward Network

## Status

Accepted

## Date

2026-10-13

## Context

`BelnapFFN` (ADR-0004) computed its hidden representation as a single shared vector
`h_mid = mish(w_1(e_pos) + w_2(e_neg))`, bottling both evidence polarities through one
representation before splitting back into dual output channels. This breaks negation
equivariance — the defining geometric property of the B4 bilattice under negation
(`x.negate()` swaps `e_pos` and `e_neg`): applying the FFN to a negated state produces an
arbitrary linear mixture instead of the swapped output, so paraconsistent semantics do not
survive non-linear feed-forward transformations.

## Decision

`BelnapFFN` maintains dual hidden channels with cross-polarity gating and tied output
parameters:

1. **Dual-channel hidden expansion**:
   `h_pos = mish(w_1(e_pos) + w_2(e_neg))` and `h_neg = mish(w_1(e_neg) + w_2(e_pos))`,
   where `w_1` transforms own-polarity evidence and `w_2` gates cross-polarity evidence.
2. **Tied output projection and normalization**: `w_o2` shares `w_o1`'s parameters and
   `norm_neg` shares `norm_pos`'s parameters (separate module objects, tied tensors), so
   both channels pass through the same output transformation.

Because the channel map is symmetric under polarity swap with shared parameters,
`f(negate(x)) == negate(f(x))` holds exactly (up to floating-point evaluation order), and
gradients from both channels accumulate into the shared output parameters.

## Consequences

- **Positive**: Strict negation equivariance across the feed-forward transformation,
  verified by exact `allclose` property tests; the bilattice polarity structure survives
  non-linear reasoning blocks.
- **Positive**: Each polarity keeps its own hidden channel, removing the information
  bottleneck of the summed intermediate.
- **Negative**: The output projection and normalization have half the parameter count of
  an untied dual-channel variant — an accepted constraint in exchange for the equivariance
  guarantee.
- **Note**: Pre-sigmoid LayerNorm hooks (`norm_pos`, `norm_neg`) remain separate module
  objects with identical parameters, preserving the R2 pre-activation normalization
  instrumentation.

## Related

- Extends [ADR-0004](./0004-belnap-bilattice-native-transformer.md) (bilattice-native
  transformer) and [ADR-0006](./0006-softexp-and-mish-activation-architecture.md)
  (softexp/Mish activation architecture).
