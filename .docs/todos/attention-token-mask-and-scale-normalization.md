# Preserve Tokenizer Padding Masks and Normalize Sigmoid Attention Scale

- **Area:** Attention Architecture & Feature Encoding (ADR-0002, ADR-0004, ADR-0005)
- **Refs:** `../plan.md`, `../adrs/0002-building-blocks-attention-and-belnap.md`, `../adrs/0005-belnap-multihead-pooled-attention.md`, `../../src/yoda/architecture/encoders.py`, `../../src/yoda/architecture/belnap_transformer.py`, `../../src/yoda/architecture/engine.py`

## Context

In `src/yoda/architecture/encoders.py`, `TextEncoder.forward` currently generates tokenizer outputs with `padding=True` and `truncation=True`, but discards `inputs["attention_mask"]` before returning hidden states through its adapter. Downstream in `src/yoda/architecture/engine.py`, `BelnapMultiheadPooledAttention` is called with `mask=None`, leaving all padding tokens active during epistemic pooling. Inside `BelnapAttention.forward` (`src/yoda/architecture/belnap_transformer.py`), attention weights $a_{\text{pos}} = \text{sigmoid}(s_{\text{pos}}) \cdot k_{\text{gate}}$ are elementwise in $[0, 1]$ rather than normalized via Softmax, giving unmasked padding tokens strictly positive weights ($\approx 0.375$). On variable-length batches with significant padding, padding tokens outnumber real tokens by up to 19-to-1, corrupting pooled evidence probes. Furthermore, unnormalized matrix multiplication $\text{agg} = A \cdot V$ sums across all sequence tokens $L_{\text{kv}}$, causing activations to scale linearly with sequence length and driving the output projection sigmoid into hard saturation ($0.0$ or $1.0$) on Step 1.

## Deferred because

This change requires modifying `TextEncoder.forward` to return attention masks alongside embeddings, propagating masks through `BelnapMPA` and `BelnapAttention`, and updating unit test fixtures in `tests/test_encoders.py` and `tests/test_belnap_transformer.py`. Implementing this along with normalization is scoped into Phase 1 of the architecture hardening plan.

## Revisit when

Next commit touching `src/yoda/architecture/encoders.py` or `src/yoda/architecture/belnap_transformer.py`, or before launching the next hyperparameter training run.
