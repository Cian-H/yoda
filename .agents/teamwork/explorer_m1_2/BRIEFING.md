# BRIEFING — 2026-10-13T20:12:00Z

## Mission
Formulate exact implementation plan for R2: Paraconsistent Contradiction Loss & Linear Warmup Schedule across ltn.py, trainer.py, lightning.py, and test suites.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesizer
- Working directory: /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2
- Original parent: ead94b72-0198-425b-a453-dfb30ce62240
- Milestone: M1 (Phase 1 Architectural Hardening)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Target scope: R2 Paraconsistent Contradiction Loss & Linear Warmup Schedule
- Target files: src/yoda/nesy/ltn.py, src/yoda/training/trainer.py, src/yoda/training/lightning.py, tests/test_ltn.py, tests/test_contradiction_loss.py
- Deliverables: handoff.md in working directory and report back via send_message

## Current Parent
- Conversation ID: ead94b72-0198-425b-a453-dfb30ce62240
- Updated: not yet

## Investigation State
- **Explored paths**: .agents/teamwork/ORIGINAL_REQUEST.md, PROJECT.md, DISPATCH.md, .docs/plan.md, .docs/todos/ltn-contradiction-penalty-and-warmup.md, src/yoda/nesy/ltn.py, src/yoda/training/trainer.py, src/yoda/training/lightning.py, tests/test_ltn.py, tests/test_independent_eval.py, tests/test_lightning.py, tests/test_training.py.
- **Key findings**:
  1. `LTNConstraintLoss.forward` currently computes `choice_pos * (1 - choice_pos) + choice_neg * (1 - choice_neg)`, which minimizes the sigmoid activation Jacobian and causes early gradient death. Replacing with `(choice_pos * choice_neg).mean() * self.current_decisiveness_weight` penalizes true Belnap contradiction ($Both$) without penalizing $Neither$ $(0, 0)$ or consistent $True$ $(1, 0)$ / $False$ $(0, 1)$.
  2. Linear warmup schedule $\lambda(e) = \lambda_{\text{target}} \cdot \min(1.0, e / W)$ protects early reasoning layer representations during Epoch 1 backpropagation.
  3. `LTNConstraintLoss.step_epoch(epoch)` exposes the interface contract for per-epoch weight progression.
  4. Integration into `YodaTrainer.fit()` at epoch loop (`epoch - 1`) and `YodaLightningAdapter.on_train_epoch_start()` seamlessly controls weight during training.
  5. Existing `test_decisiveness_penalty` in `tests/test_ltn.py` must be adapted because its assertions assume quadratic $p(1-p)$ behavior (e.g. loss 0.1 at (0.5, 0.5) becoming 0.05, and negative gradient for values > 0.5 which now becomes positive).
  6. New `tests/test_contradiction_loss.py` will comprehensively cover Belnap corner states, symmetry, monotonicity, Jacobian health, mask propagation, warmup progression, and trainer integrations.
- **Unexplored areas**: None within the scope of R2.

## Key Decisions Made
- Parameterized `LTNConstraintLoss.__init__` with `decisiveness_weight: float = 0.1`, `warmup_epochs: int = 0`.
- Kept default `warmup_epochs = 0` for backwards compatibility so standalone criteria in existing tests evaluate at full target weight immediately unless warmup is explicitly requested.
- Implemented `step_epoch(epoch: int | None = None)` allowing both 0-indexed explicit epoch tracking and auto-increment.
- Mapped epoch indexing in `YodaTrainer.fit` (`epoch - 1`) and `YodaLightningAdapter` (`self.current_epoch`).

## Artifact Index
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/BRIEFING.md — Persistent working memory
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/DISPATCH.md — Task dispatch log
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/progress.md — Heartbeat and progress tracking
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/handoff.md — 5-component handoff report
