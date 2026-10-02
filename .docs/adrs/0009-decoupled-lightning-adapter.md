# ADR 0009: Decoupled PyTorch Lightning Adapter and Frozen Backbone Architecture

## Status
Accepted

## Date
2026-10-02

## Context
As the Yoda decision engine advances toward scaled multi-dataset experimentation, two architectural questions arose:
1. **Experiment Tracking & Training Framework**: How to leverage modern experiment tracking, multi-GPU scaling (DDP), automatic mixed precision (AMP), and checkpointing without locking the core decision engine into a third-party framework runtime.
2. **Text Encoder Freezing Policy**: In early experiments, the top transformer layer of `all-MiniLM-L6-v2` was unfrozen for semantic domain adaptation, adding 3.15M trainable parameters (out of 4.02M total). However, fine-tuning the text backbone on small-to-medium dataset samples risks semantic drift and catastrophic forgetting of general sentence semantics, while allowing the model to bypass the Belnap reasoning cascade by warping token embeddings directly.

## Decision
1. **Decoupled Lightning Adapter Pattern**:
   - `YodaDecisionEngine` remains a 100% standalone, framework-agnostic `torch.nn.Module`.
   - `YodaLightningAdapter(pl.LightningModule)` is implemented as a thin wrapper in `src/yoda/training/lightning.py` that delegates forward passes to `self.model: YodaDecisionEngine`.
   - The native lightweight `YodaTrainer` remains the default for fast local unit testing and quick DLA diagnostics, while `--use-lightning` is available for full-scale experiment runs.
2. **100% Frozen Text Backbone by Default**:
   - The pretrained text encoder backbone (`all-MiniLM-L6-v2`) is frozen 100% by default across all 6 transformer layers.
   - Trainable parameters are reduced from **4.02M down to ~860k** (~80% reduction), concentrating gradient updates entirely on the Belnap Bilattice reasoning cascade, multihead pooled attention (MPA), and LTN constraint projections.
   - An explicit opt-in flag (`--unfreeze-top-layer`) is retained for downstream domain adaptation if required.

## Consequences
- **Positive**: Zero vendor lock-in. Models trained with Lightning can be saved, exported (ONNX/Safetensors), or deployed using pure PyTorch without importing Lightning.
- **Positive**: Dramatic speedup and reduction in memory footprint during training, with no gradient buffers maintained across MiniLM transformer layers.
- **Positive**: Protects the pretrained semantic embedding manifold from distortion and forces the Epistemic Sieve and reasoning layers to solve the tasks.
- **Negative**: Slightly longer training scripts due to maintaining both native and Lightning execution paths.
