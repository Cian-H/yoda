# Dispatch for Explorer M1_2 (Phase 1: Contradiction Loss & Linear Warmup)

**Task**: Formulate implementation strategy for R2 (Paraconsistent Contradiction Loss & Linear Warmup).
**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Target Files**:
- `src/yoda/nesy/ltn.py`
- `src/yoda/training/trainer.py`
- `src/yoda/training/lightning.py`

**Instructions**:
1. Read `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/cianh/Projects/yoda/PROJECT.md`.
2. Formulate the replacement of $p(1-p) + q(1-q)$ with $e^+ \cdot e^-$ in `LTNConstraintLoss.forward`.
3. Design the linear warmup schedule $\lambda(e) = \lambda_{\text{target}} \cdot \min(1.0, e / W)$ and its integration in both `YodaTrainer` and `YodaLightningAdapter`.
4. Analyze how existing tests (`tests/test_ltn.py::test_decisiveness_penalty`) should be updated/migrated to validate the new paraconsistent contradiction contract.
5. Detail unit tests for `tests/test_contradiction_loss.py`.
6. Write your report to `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/handoff.md` and notify orchestrator.


## 2026-10-13T20:05:06Z
You are Explorer M1_2 for Phase 1. Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/DISPATCH.md.
Focus on R2: Paraconsistent Contradiction Loss & Linear Warmup Schedule.
Investigate src/yoda/nesy/ltn.py, src/yoda/training/trainer.py, src/yoda/training/lightning.py.
Formulate the exact implementation plan for:
1. Replacing p(1-p) + q(1-q) with paraconsistent Both penalty (choice_pos * choice_neg).mean().
2. Linear warmup schedule parameter and step/epoch progression.
3. Adapting existing tests in tests/test_ltn.py (e.g. test_decisiveness_penalty).
4. New unit tests in tests/test_contradiction_loss.py.
Write handoff report to /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_2/handoff.md and report back via send_message.
