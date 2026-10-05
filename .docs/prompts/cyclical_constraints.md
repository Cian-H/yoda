Task: Implement staggered, oscillating cosine schedulers for constraint weights.

Changes:
- Added `CyclicalConstraintScheduler` to `yoda.training.trainer` using inverted cosine warm restarts.
- Initialized `ltn_scheduler` (starts at step 0) and `assertion_scheduler` (starts at T_0) in `YodaTrainer.fit`.
- Plumbed dynamic weights (`current_ltn_w`, `current_assertion_w`) into batch loss calculation and TensorBoard logs.
- Hardcoded evaluation to always use max constraint weights for strict metrics.
- Added comprehensive unit tests for scheduling mathematics and staggered starts.
