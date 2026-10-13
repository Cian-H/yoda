# BRIEFING — 2026-10-13T22:19:00Z

## Mission
Formulate implementation plan for R3: Positive Support LTN XOR/OR Reformulation in src/yoda/nesy/ltn.py, tests/test_ltn.py, and new tests/test_nesy_reformulation.py.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1
- Original parent: ead94b72-0198-425b-a453-dfb30ce62240
- Milestone: Milestone 2

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Evaluate XOR/OR on positive support (e_pos / choice_pos) rather than truth coordinates (t), ensuring unmentioned Neither options incur 0 penalty.
- Specify exact changes in src/yoda/nesy/ltn.py, tests/test_ltn.py, and new tests/test_nesy_reformulation.py.
- Write handoff report to /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/handoff.md and report back via send_message.

## Current Parent
- Conversation ID: ead94b72-0198-425b-a453-dfb30ce62240
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `PROJECT.md`, `.docs/plan.md`, `.docs/todos/ltn-xor-neither-support-reformulation.md`
  - `.agents/skills/nesy-constraint-check/SKILL.md`
  - `src/yoda/nesy/ltn.py`, `src/yoda/architecture/engine.py`, `src/yoda/training/trainer.py`
  - `tests/test_ltn.py`, `tests/test_contradiction_loss.py`, `tests/test_independent_eval.py`
  - `tests/e2e/test_paraconsistent_gate.py`, `tests/e2e/test_tier1_feature_coverage.py`, `tests/e2e/test_tier2_boundary_corner.py`, `tests/e2e/test_tier3_cross_feature.py`
- **Key findings**:
  - Identified root cause of unmentioned candidate penalties: truth coordinate $t_j = 0.5(e_j^+ - e_j^- + 1)$ evaluates to 0.5 under Neither $(0, 0)$, imposing false $-\log(1 - 0.5) = 0.69315$ loss per unmentioned option and driving network to fabricate false refutations $e^- \to 1.0$.
  - Positive support evaluation ($e_j^+ = 0$) ensures $(1 - e_j^+) = 1.0$, reducing penalty to exactly 0.0 while preserving gradient isolation ($\frac{\partial L}{\partial e^-} = 0$).
  - For unmentioned candidate slices (e.g. `test_paraconsistent_gate.py`), integrating non-assertion probability $p_{\text{none}} = \prod (1 - e_i^+)$ ensures pure Neither groups incur zero exclusivity penalty.
  - Updating XOR/OR from truth to `choice_pos` changes gradient paths in `tests/test_ltn.py`, requiring gradient assertions to check `choice_pos.grad` rather than `truth.grad`.
- **Unexplored areas**: None for R3 LTN scope.

## Key Decisions Made
- Fully specified exact changes for `src/yoda/nesy/ltn.py` (`_compute_xor`, `_compute_or`, `_compute_grouped_constraints`, `_compute_legacy_constraints`, `forward`, plus alias functions).
- Created exact drop-in proposed files: `proposed_ltn.py`, `proposed_test_ltn.py`, `proposed_test_nesy_reformulation.py`, and patch `ltn_positive_support.patch`.

## Artifact Index
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/DISPATCH.md` — Dispatch instructions
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/BRIEFING.md` — Working memory
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/progress.md` — Heartbeat / progress log
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_ltn.py` — Complete proposed `ltn.py`
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_ltn.py` — Complete proposed `test_ltn.py`
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/proposed_test_nesy_reformulation.py` — Complete new test suite
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/ltn_positive_support.patch` — Unified diff patch
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_1/handoff.md` — 5-component handoff report
