Task: Create an Optuna study script for topological and embedder sweep.

Changes:
- Created `experiments/optuna_study.py` to coordinate trials.
- Configured search space for embedders, topology blocks, attention shapes, and learning dynamics.
- Implemented isolated subprocess execution to prevent PyTorch CUDA OOM fragmentation.
- Implemented `history.json` parsing to maximize final epoch validation accuracy.
