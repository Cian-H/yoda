# Dispatch for Explorer M2_1 (Phase 2: Positive Support XOR/OR Reformulation)

**Task**: Formulate implementation strategy for R3: Positive Support LTN XOR/OR Reformulation.
**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Target Files**:
- `src/yoda/nesy/ltn.py`
- `tests/test_ltn.py`
- `tests/e2e/test_paraconsistent_gate.py`

**Instructions**:
1. Read `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/cianh/Projects/yoda/PROJECT.md`.
2. Analyze `src/yoda/nesy/ltn.py` (`_compute_xor`, `_compute_or`, `forward`).
3. Replace truth $t_i = 0.5(e^+ - e^- + 1)$ in XOR/OR with positive support $e_i^+$ (`choice_pos`):
   $$P(\text{XOR}) = \sum_{i=1}^K e_i^+ \prod_{j \neq i} (1 - e_j^+)$$
   $$P(\text{OR}) = 1 - \prod_{i=1}^K (1 - e_i^+)$$
4. Verify that unchosen candidate options at Neither $(0, 0)$ have $e_j^+ = 0 \implies (1 - e_j^+) = 1.0$, incurring 0 constraint penalty.
5. Formulate unit tests in `tests/test_ltn.py` and new `tests/test_nesy_reformulation.py`.
6. Write handoff report to `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/handoff.md` and notify orchestrator.

## 2026-10-13T20:58:23Z
You are Explorer M2_1 for Milestone 2. Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/DISPATCH.md.
Domain skill: /home/cianh/Projects/yoda/.agents/skills/nesy-constraint-check/SKILL.md.
Formulate implementation plan for R3: Positive Support LTN XOR/OR Reformulation in src/yoda/nesy/ltn.py.
Evaluate XOR/OR on positive support (e_pos / choice_pos) rather than truth coordinates (t), ensuring unmentioned Neither options incur 0 penalty.
Specify exact changes in src/yoda/nesy/ltn.py, tests/test_ltn.py, and new tests/test_nesy_reformulation.py.
Write handoff report to /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/handoff.md and report back via send_message.
