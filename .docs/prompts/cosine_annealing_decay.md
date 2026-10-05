Task: Replace OneCycleLR with CosineAnnealingWarmRestarts with decaying amplitude and lengthening cycles.

Changes:
- Replaced OneCycleLR in YodaTrainer with CosineAnnealingWarmRestarts.
- Added T_0, T_mult, and lr_decay configs to Trainer and CLI arguments.
- Intercepted scheduler step to dynamically decay base_lrs at restart boundaries (T_cur == 0).
- Updated tests to verify scheduling and peak decay logic.
