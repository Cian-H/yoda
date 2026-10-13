# ADR 0021: Context Encoding Deduplication and Cross-Call Memoization

## Status

Accepted

## Date

2026-10-13

## Context

Independent choice evaluation (ADR-0013) unrolls each question with $K$ choices into
$B \times K$ samples. Every unrolled row re-encodes its question's query text and symbolic
state through the text and state backbones, so for typical $K \in [3, 5]$, between 66% and
80% of context-side encoder and pooling forward passes per batch are redundant: rows
sharing a question compute identical query and state encodings. Repeated evaluation of
the same workload (serving loops, benchmark sweeps) recomputes those encodings on every
call.

## Decision

`YodaDecisionEngine` gains a `use_context_cache` flag (default `True`) that activates
context encoding deduplication in independent evaluation mode:

1. **Within-batch deduplication**: rows are grouped by their unique
   `(query, state, task)` context key. Unique query and state texts are encoded in a
   single backbone pass, pooled once through the q/c Belnap MPAs, and reasoned over once
   by the first layer's context block; the resulting states are expanded back to every
   row by indexing. Candidate encoding, the constraint cross-attention blocks, and the
   decision head stay per-row because candidates are distinct by design.
2. **Cross-call memoization** (`_context_cache`, `_candidate_cache`): under
   gradient-free execution, the pooled epistemic states of unique contexts and of
   individual candidate texts are memoized across forward calls. Any gradient-enabled
   forward clears both caches, so weight updates can never leave stale reasoning in the
   cache — evaluation between training epochs always recomputes against current weights.
3. **Correctness**: with deterministic text encoding, deduplicated and memoized outputs
   are identical to the fully recomputed path up to floating-point noise.

## Consequences

- **Positive**: independent candidate evaluation throughput improves by 3-5x on
  multi-choice workloads (acceptance gate requires >= 2.0x), and context backbone passes
  per batch drop from $B \times K$ to the number of unique contexts.
- **Positive**: Training is unaffected — gradient-enabled forwards bypass memoization
  entirely and clear the caches.
- **Negative**: The cache grows with the number of distinct contexts and candidates seen
  during gradient-free evaluation; long-running services should call
  `clear_context_cache()` periodically.
- **Negative**: Batched encoding of unique contexts pads query and state texts to a common
  length within one call; with positional encodings from real HF backbones this can shift
  absolute positions relative to separately-encoded texts (padding is masked, so pooled
  outputs remain correct for the deterministic encoder used in tests).

## Related

- Builds on [ADR-0013](./0013-independent-choice-assessment-architecture.md) and
  [ADR-0019](./0019-centered-bilinear-candidate-alignment-and-group-relative-scoring.md),
  which defines the `(query, state, task)` context grouping used as the cache key.
