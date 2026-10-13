# Implement Coupled Cross-Polarity Value Routing in Belnap Attention

- **Area:** Bilattice Attention & Transformer Architecture (ADR-0002, ADR-0004)
- **Refs:** `../plan.md`, `../adrs/0002-building-blocks-attention-and-belnap.md`, `../adrs/0004-belnap-bilattice-native-transformer.md`, `../../src/yoda/architecture/belnap_transformer.py`, `../../tests/test_belnap_transformer.py`

## Context

In `src/yoda/architecture/belnap_transformer.py`, `BelnapAttention.forward` decomposes attention affinity into coherent support ($s_{\text{pos}} \propto Q_+ K_+^\top + Q_- K_-^\top$) and opposite-polarity refutation ($s_{\text{neg}} \propto Q_+ K_-^\top + Q_- K_+^\top$). However, Step 4 aggregates values through isolated parallel channels: `agg_pos = a_pos @ v_pos` and `agg_neg = a_neg @ v_neg`. If a key token carries strong negative evidence ($v_{\text{neg}} \gg 0, v_{\text{pos}} \approx 0$) and a query aligns coherently with that key ($a_{\text{pos}} \gg 0, a_{\text{neg}} \approx 0$), `agg_pos` multiplies $a_{\text{pos}}$ by $v_{\text{pos}} \approx 0$, while `agg_neg` multiplies $a_{\text{neg}} \approx 0$ by $v_{\text{neg}}$. The key's negative refutation evidence is completely dropped from the aggregated state. To adhere strictly to $\mathcal{B}_4$ bilattice algebra, evidence aggregation should operate as a $2 \times 2$ coupled operator: $\text{agg}_{\text{pos}} = a_{\text{pos}} V_+ + a_{\text{neg}} V_-$ and $\text{agg}_{\text{neg}} = a_{\text{pos}} V_- + a_{\text{neg}} V_+$.

## Deferred because

Coupled value aggregation changes the forward aggregation step across all multihead attention layers in `BelnapAttention` and requires updating unit tests in `tests/test_belnap_transformer.py` and verifying head dimension handling. Staged under Phase 2 of the roadmap.

## Revisit when

Next commit modifying `src/yoda/architecture/belnap_transformer.py` or when evaluating refutation-heavy decision datasets.
