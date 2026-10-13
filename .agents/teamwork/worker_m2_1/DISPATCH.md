# Dispatch for Worker M2_1 (Phase 2 NeSy Reformulation & Epistemic Logits Implementation)

**Task**: Implement Milestone 2: Phase 2 NeSy Reformulation & Epistemic Logits (Requirement R3).
**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Working Directory**: `/home/cianh/Projects/yoda/.agents/teamwork/worker_m2_1`
**Domain Skill**: `/home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md`

**Exclusive Write Ownership**:
- `src/yoda/nesy/ltn.py`
- `src/yoda/architecture/belnap_transformer.py`
- `src/yoda/architecture/engine.py`
- `tests/test_ltn.py`
- `tests/test_belnap_transformer.py`
- `tests/test_engine.py`
- `tests/test_nesy_reformulation.py`
- `tests/test_coupled_routing.py`
- `tests/test_knowledge_gating.py`

**MANDATORY INTEGRITY WARNING**:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A forensic auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

**Input Plans & Reference Artifacts**:
- Explorer M2_1:
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/handoff.md`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_ltn.py`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/ltn_positive_support.patch`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_ltn.py`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_nesy_reformulation.py`
- Explorer M2_2:
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_2/handoff.md`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_2/proposed_belnap_transformer.patch`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_2/proposed_test_belnap_transformer.patch`
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_2/proposed_test_coupled_routing.py`
- Explorer M2_3:
  - `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_3/handoff.md`
  - Section 4.1 code changes for `BelnapDecisionHead.forward` in `src/yoda/architecture/engine.py`
  - Section 4.2 tests for `tests/test_engine.py`
  - Section 4.3 new test file `tests/test_knowledge_gating.py`

**Instructions**:
1. Implement positive support XOR/OR in `src/yoda/nesy/ltn.py` (`_compute_xor`, `_compute_or`, and `forward` calls), ensuring $Neither$ $(0, 0)$ incurs zero penalty.
2. Implement $2 \times 2$ coupled cross-polarity evidence routing in `BelnapAttention.forward` (`src/yoda/architecture/belnap_transformer.py`), ensuring refutations are preserved and double negation symmetry holds.
3. Implement epistemic logit synchronization and knowledge gating in `BelnapDecisionHead.forward` (`src/yoda/architecture/engine.py`), ensuring logits are gated by candidate knowledge coordinates ($k$).
4. Update existing tests in `tests/test_ltn.py`, `tests/test_belnap_transformer.py`, and `tests/test_engine.py`.
5. Add the three new test suites (`tests/test_nesy_reformulation.py`, `tests/test_coupled_routing.py`, `tests/test_knowledge_gating.py`).
6. Run full verification: `uv run pytest tests/` and `uv run ruff check src/ tests/ experiments/`.
7. Deliver full handoff report to `/home/cianh/Projects/yoda/.agents/teamwork/worker_m2_1/handoff.md` and report completion back to orchestrator.

## 2026-10-13T21:21:13Z
You are Worker M2_1 for Milestone 2 (Phase 2 NeSy Reformulation & Epistemic Logits). Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/worker_m2_1.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/worker_m2_1/DISPATCH.md.
Domain skill: /home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md.
