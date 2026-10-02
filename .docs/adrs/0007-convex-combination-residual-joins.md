# ADR 0007: Convex Combination Residual Joins for Epistemic Bilattices

- **Status**: Accepted
- **Date**: 2026-10-02
- **Author**: Antigravity & Cian
- **Deciders**: Engineering & Research Team

---

## 1. Context & Problem Statement

In the Belnap Bilattice Transformer (`BelnapTransformerBlock`), internal state representations $x = (e^+, e^-)$ must remain strictly bounded in $[0, 1]$ to maintain their semantic interpretation as positive and negative evidence coordinates.

Previously, `BelnapTransformerBlock._residual_join` used hard clamping to enforce these boundaries:
$$\text{res} = \text{clamp}(\text{base} + \Delta, 0.0, 1.0)$$

Diagnostic probing via Direct Logit Attribution (DLA) revealed a critical pathology:
1. **Saturation Dead-Zone**: The delta updates $\Delta$ from attention and FFN layers are predominantly non-negative ($> 0$). Successive residual additions quickly drive evidence values to the ceiling of $1.0$.
2. **Extinguished Gradients**: For any $x \ge 1.0$, the derivative of the clamp function is identically zero ($\frac{d}{dx}\text{clamp}(x) = 0$). This completely froze downstream reasoning layers.
3. **Dead Stage 2 (Constraint Reasoning)**: DLA probes showed Stage 2 attribution was exactly $+0.0000$, with Stage 1 and Stage 2 output logits being bit-for-bit identical.

Traditional alternatives like applying a sigmoid ($\sigma(\text{base} + \Delta)$) are unsuitable because sequential sigmoids suffer from $\sigma'(z) \le 0.25$, multiplying exponentially to near-zero across multiple cascade stages.

---

## 2. Decision

We replace hard clamping in `BelnapTransformerBlock._residual_join` with a **Convex Combination Residual Join**:

$$\text{res} = (1 - \alpha) \cdot \text{base} + \alpha \cdot \Delta$$

where $\alpha \in (0, 1)$ (parameterized by `residual_weight`, defaulting to $0.5$).

### Mathematical Guarantees:
1. **Invariant Preservation**: If $\text{base} \in [0, 1]$ and $\Delta \in [0, 1]$, then for any $\alpha \in [0, 1]$, their convex combination is mathematically guaranteed to stay in $[0, 1]$ without requiring squashing or clipping:
   $$0 \le (1 - \alpha) \cdot 0 + \alpha \cdot 0 \le \text{res} \le (1 - \alpha) \cdot 1 + \alpha \cdot 1 = 1$$
2. **Linear Gradient Flow**: The gradient with respect to the input representation is constant and non-zero:
   $$\frac{\partial\,\text{res}}{\partial\,\text{base}} = 1 - \alpha = 0.5$$
   This eliminates both saturation dead-zones and non-linear sigmoid compression.

---

## 3. Implementation Details

- Configured `residual_weight: float = 0.5` in `BelnapTransformerBlock`, `BelnapDecisionTransformer`, and `YodaDecisionEngine`.
- Updated `BelnapTransformerBlock._residual_join` to accept `alpha: float = 0.5`.
- Added unit tests in `tests/test_belnap_transformer.py` verifying that even when $\text{base} + \Delta > 1.0$, the output remains bounded and retains an exact non-zero gradient ($0.5$).

---

## 4. Consequences

### Positive
- **Active Deep Reasoning**: Stages downstream of self-attention and cross-attention receive unattenuated gradients, allowing constraint reasoning to contribute actively to decision logits.
- **Calibrated Epistemic Representation**: Prevents evidence saturation from inflating knowledge coordinates to artificial certainty ($k \to 1$).
- **Numerical Stability**: Requires no clipping, protecting against floating-point boundary anomalies.
