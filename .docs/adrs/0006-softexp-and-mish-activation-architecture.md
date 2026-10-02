# ADR 0006: SoftExp and Mish Activation Upgrades for Gradient Flow and Evidence Modeling

- **Status**: Accepted
- **Date**: 2026-10-02
- **Author**: Antigravity & Cian
- **Deciders**: Engineering & Research Team

---

## 1. Context & Problem Statement
Direct Logit Attribution (DLA) diagnostic probing across the reasoning cascade revealed severe gradient vanishing and epistemic collapse:
1. **Vanishing Attributions**: Context and constraint reasoning layers contributed near-zero marginal logit delta ($+0.0031$ and $+0.0010$) on the evaluation set.
2. **Sigmoid Saturation Bottleneck**: `BelnapAttention` output evidence was initialized with `init_bias = -3.0`, causing $\sigma'(-3.0) \approx 0.045$, attenuating over 95% of backpropagated gradients per layer.
3. **Multiplicative Gating Choke**: Attention weights were gated by $k_{\text{src}}$ via direct multiplication ($a = \sigma(s) \cdot k_{\text{src}}$), extinguishing gradients whenever source knowledge was low.
4. **Logit Zero-Out Trap**: `BelnapDecisionHead` computed `logits = (pos - neg) * knowledge`. The optimizer minimized cross-entropy by pushing $k \to 0$, creating flat logits ($0.0$) and a risk-free loss plateau at $\ln(5) \approx 1.6094$.

---

## 2. Decision

We introduce two complementary activation functions to restore continuous gradient flow while enhancing epistemic modeling:

1. **`SoftExp` (Soft Exponential Activation, Godfrey 2015)**:
   - Added as a new module in `src/yoda/architecture/activations.py`.
   - Parameterized by trainable $\alpha \in \mathbb{R}^D$ (per-channel or scalar):
     - Sub-linear / Logarithmic ($\alpha < 0$) for saturating evidence accumulation.
     - Linear identity ($\alpha \to 0$) via smooth Taylor expansion $x + \frac{1}{2}\alpha x^2$ ensuring non-zero gradients at initialization.
     - Super-linear / Exponential ($\alpha > 0$) for critical-mass thresholding.
   - Decomposed into distinct methods: `_linear_taylor`, `_logarithmic_branch`, `_exponential_branch` with numerical clamping against `NaN` and `Inf`.
   - Integrated into `BelnapMultiheadPooledAttention` (MPA) to map dense latent spaces into non-linear epistemic evidence curves.

2. **`Mish` Activation (`torch.nn.functional.mish`)**:
   - Integrated into `BelnapFFN` hidden layers in place of `torch.relu`.
   - Continuous, smooth, non-monotonic, unbounded above, eliminating the dead-neuron and gradient saturation issues.

3. **Attention & Decision Head Remediation**:
   - `BelnapAttention`: Re-center default `init_bias` from `-3.0` to `0.0`.
   - `BelnapAttention`: Modify knowledge gating to smooth residual scaling ($0.5 + 0.5 \cdot k_{\text{src}}$) to prevent zero-multiplier gradient cutoffs.
   - `BelnapDecisionHead`: Decouple classification logits from the $k \to 0$ trap by setting `logits = (pos_affinity - neg_affinity) / scale`, while retaining `choice_pos` and `choice_neg` in $[0, 1]$ for Belnap epistemic coordinates and `FuzzyBelnapLoss`.

---

## 3. Consequences

### Positive
- **Gradient Preservation**: Unbounded positive gradients via Mish and re-centered output biases eliminate the 95%+ attenuation per block.
- **Escape from Ignorance Shortcut**: Decoupling ranking logits from $k$ prevents the network from collapsing to uniform random choice to minimize loss.
- **Learnable Evidence Curvature**: `SoftExp` empowers the network to autonomously discover logarithmic or exponential evidence accumulation curves.

### Trade-offs & Mitigations
- **Branching in SoftExp**: Evaluated via vectorized `torch.where` with analytical Taylor series near $\alpha=0$ to ensure CUDA compatibility and prevent `NaN`s.
