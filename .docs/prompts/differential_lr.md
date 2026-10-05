Task: Implement differential learning rates for text backbone.

Changes:
- Added `backbone_lr` support to `YodaTrainer` optimizer initialization.
- Split parameters into `backbone_params` (text_encoder.model) and `head_params`.
- Assigned distinct learning rates via PyTorch parameter groups.
- Changed default `--freeze-backbone` to False in `train_yoda.py`.
- Added `--backbone-lr` CLI flag (default 1e-5).
- Added test coverage for dual parameter group routing.
