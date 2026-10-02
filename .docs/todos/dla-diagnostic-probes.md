# Direct Logit Attribution (DLA) Diagnostic Probes

- **Area:** Architecture & Interpretability (XAI)
- **Refs:** ../prompts/1790933295.shuffle_choices_and_dla_probe_design.md, ../../src/yoda/architecture/engine.py, ../../src/yoda/architecture/belnap_transformer.py

## Context

During early benchmarking of the YodaDecisionEngine on choice QA datasets, we recognized the need for mechanistic interpretability to verify that the cascading BelnapTransformerBlocks are actively performing evidence refinement rather than acting as a redundant identity path. In standard transformer models, Direct Logit Attribution (DLA) projects intermediate hidden states at each layer into the output vocabulary to observe the progression of logits and rank. For Yoda's continuous paraconsistent reasoning architecture, intermediate query evidence states can be fed directly to the BelnapDecisionHead alongside candidate options. This will measure the step-by-step evolution of target logit, choice rank, layer attribution delta, and epistemic knowledge ($k$) across post-pooling, post-context, and post-constraint stages.

## Deferred because

The user requested completing and committing the dataset choice shuffling first to solidify the data pipeline, followed by an alignment checkpoint on the proposed DLA probe design before modifying the core engine forward pass. Pausing here allows deliberate verification of the probe interface requirements without entangling data loader changes with model architecture modifications.

## Revisit when

The user reviews the DLA probe architectural design and gives input on whether to proceed with wiring `return_diagnostics` into `YodaDecisionEngine`.
