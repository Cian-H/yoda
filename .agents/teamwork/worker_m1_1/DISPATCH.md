# Dispatch for Worker M1_1 (Phase 1 Architectural Hardening Implementation)

**Task**: Implement Phase 1 Architectural Hardening: R1 (Attention Mask Propagation & Scale Normalization) and R2 (Paraconsistent Contradiction Loss, Warmup Schedule, Pre-Activation Layer Normalization, Residual Highway $\alpha=0.15$).

**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Working Directory**: `/home/cianh/Projects/yoda/.agents/teamwork/worker_m1_1`

**Input Reports & Specifications**:
- Explorer M1_1: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/handoff.md`, `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/proposed_changes.patch`, `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/proposed_test_attention_masking.py`
- Explorer M1_2: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/handoff.md`
- Explorer M1_3: `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/handoff.md`, `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/proposed_belnap_transformer.patch`, `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/proposed_optuna_study_narrow.patch`, `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/proposed_test_prenorm_highway.py`
- Domain Skill: `/home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md`

**Exclusive Write Ownership**:
- `src/yoda/architecture/encoders.py`
- `src/yoda/architecture/attention.py`
- `src/yoda/architecture/belnap_transformer.py`
- `src/yoda/architecture/engine.py`
- `src/yoda/nesy/ltn.py`
- `src/yoda/training/trainer.py`
- `src/yoda/training/lightning.py`
- `experiments/optuna_study_narrow.py`
- `tests/test_encoders.py`
- `tests/test_ltn.py`
- `tests/test_optuna_study_narrow.py`
- `tests/test_attention_masking.py`
- `tests/test_contradiction_loss.py`
- `tests/test_prenorm_highway.py`

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A forensic auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

**Instructions**:
1. Implement all changes specified by Explorers M1_1, M1_2, and M1_3.
2. Ensure active sequence mass normalization and pre-sigmoid LayerNorm are both cleanly combined in `BelnapAttention` and `BelnapFFN`.
3. Add the three new test suites (`tests/test_attention_masking.py`, `tests/test_contradiction_loss.py`, `tests/test_prenorm_highway.py`).
4. Update existing tests in `tests/test_ltn.py` and `tests/test_optuna_study_narrow.py` to match new paraconsistent and residual contracts.
5. Run full test suite (`uv run pytest`) and lint check (`uv run ruff check .`).
6. Write full handoff report to `/home/cianh/Projects/yoda/.agents/teamwork/worker_m1_1/handoff.md` and report completion back to orchestrator.

## 2026-10-13T20:14:47Z
You are Worker M1_1 for Phase 1 Architectural Hardening. Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/worker_m1_1.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/worker_m1_1/DISPATCH.md.
Domain skills to reference: /home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A forensic auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your exclusive write ownership:
- src/yoda/architecture/encoders.py
- src/yoda/architecture/attention.py
- src/yoda/architecture/belnap_transformer.py
- src/yoda/architecture/engine.py
- src/yoda/nesy/ltn.py
- src/yoda/training/trainer.py
- src/yoda/training/lightning.py
- experiments/optuna_study_narrow.py
- tests/test_encoders.py
- tests/test_ltn.py
- tests/test_optuna_study_narrow.py
- tests/test_attention_masking.py
- tests/test_contradiction_loss.py
- tests/test_prenorm_highway.py

Follow the detailed instructions, patches, and code diffs in:
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/handoff.md
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/handoff.md
- /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/handoff.md

Implement all changes for R1 and R2:
1. Tokenizer attention mask propagation in TextEncoder, SymbolicStateEncoder, ConstraintEncoder, with forward_with_mask and @overload.
2. Padding mask handling and active sequence mass normalization (agg / denom) in BelnapAttention.
3. Replace decisiveness penalty with paraconsistent Both contradiction penalty (choice_pos * choice_neg).mean() and linear warmup schedule in LTNConstraintLoss, YodaTrainer, and YodaLightningAdapter.
4. Insert LayerNorm before output sigmoids in BelnapAttention and BelnapFFN.
5. Recalibrate default residual_weight to 0.15 in BelnapTransformerBlock and _residual_join. Update FIXED_RESIDUAL_WEIGHT to 0.15 in optuna_study_narrow.py.
6. Update existing tests in tests/test_ltn.py and tests/test_optuna_study_narrow.py.
7. Add tests/test_attention_masking.py, tests/test_contradiction_loss.py, and tests/test_prenorm_highway.py.
8. Execute verification: run `uv run pytest` and `uv run ruff check .`.
9. Write complete handoff report to /home/cianh/Projects/yoda/.agents/teamwork/worker_m1_1/handoff.md and report back via send_message.


## 2026-10-13T20:40:13Z
**Context**: Liveness check
**Content**: What is your current execution status on Phase 1 implementation?
**Action**: Report current progress and status immediately.
