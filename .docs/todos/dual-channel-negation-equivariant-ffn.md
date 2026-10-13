# Preserve Bilattice Polarities via Dual-Channel Negation-Equivariant FFN

- **Area:** Bilattice Transformer Architecture (ADR-0004, ADR-0006)
- **Refs:** `../plan.md`, `../adrs/0004-belnap-bilattice-native-transformer.md`, `../adrs/0006-softexp-and-mish-activation-architecture.md`, `../../src/yoda/architecture/belnap_transformer.py`, `../../tests/test_belnap_transformer.py`

## Context

In `src/yoda/architecture/belnap_transformer.py`, `BelnapFFN.forward` resolves conflicting evidence via conflation and then computes intermediate activations as `h_mid = nn.functional.mish(self.w_1(x_res.e_pos) + self.w_2(x_res.e_neg))`. Summing linear projections of positive and negative evidence into a single shared vector `h_mid` bottlenecks both polarities through a unified representation. This breaks negation equivariance ($x.\text{negate}()$) because swapping $e_{\text{pos}}$ and $e_{\text{neg}}$ produces an arbitrary linear combination rather than an inverted polarity state. Refactoring `BelnapFFN` to maintain independent dual-channel projections ($h_{\text{pos}}, h_{\text{neg}}$) with cross-channel gating preserves the strict geometric properties of the $\mathcal{B}_4$ bilattice across non-linear feed-forward transformations.

## Deferred because

Refactoring `BelnapFFN` alters the internal linear layer weights and shapes of the feed-forward module in `BelnapTransformerBlock`. This architectural refinement is assigned to Phase 3 of the project plan.

## Revisit when

Next commit modifying `src/yoda/architecture/belnap_transformer.py` or when evaluating formal property-based negation equivariance tests.
