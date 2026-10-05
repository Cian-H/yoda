Task: Parameterize YodaDecisionEngine topology and add Dropout for Optuna tuning.

Changes:
- Added `dropout` parameter to Belnap FF, Attention, and Transformer blocks.
- Converted single-pass reasoning core into a configurable pipeline of `num_reasoning_blocks` length.
- Added `d_hidden_multiplier`, `conflation_weight`, and `dropout` to YodaDecisionEngine.
- Promoted all topological parameters to CLI flags in `train_yoda.py`.
- Updated test suite for multi-block forward passes, dropout stochasticity, and dynamic DLA stages.
