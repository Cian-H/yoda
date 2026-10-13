# Cache Unique Context Encodings for Unrolled Independent Candidates

- **Area:** Performance, Training Throughput & Engine Architecture (ADR-0010, ADR-0013)
- **Refs:** `../plan.md`, `../adrs/0010-decoupled-columnar-parquet-dataset.md`, `../adrs/0013-independent-choice-assessment-architecture.md`, `../../src/yoda/architecture/engine.py`, `../../tests/test_independent_eval.py`

## Context

In `src/yoda/architecture/engine.py`, when independent candidate evaluation is active (`is_independent = True`), a batch of $B$ questions with $K$ choices each is unrolled into $B \times K$ flattened samples. Phase 1 encoding runs `self.text_encoder(queries)` and `self.state_encoder(states)` directly on these unrolled batches. Because each candidate in a question group shares the exact same query text and symbolic state dictionary, the text and state transformer backbones execute $K$ duplicate forward passes per question. For typical datasets where $K \in [3, 5]$, between 66% and 80% of all encoder forward passes are redundant computation. Caching or grouping unique `(query, state)` pairs per `group_id` before Phase 3 cross-attention eliminates duplicate backbone encoding, yielding an immediate $\sim K\times$ throughput speedup on multi-choice benchmarks.

## Deferred because

The immediate priority is resolving the mathematical loss collapse (36.5%) and accuracy ceiling (50%). Caching context encodings is an optimization scheduled for Phase 4 of the roadmap.

## Revisit when

Next commit optimizing throughput in `src/yoda/architecture/engine.py` or when scaling training runs to larger dataset sizes.
