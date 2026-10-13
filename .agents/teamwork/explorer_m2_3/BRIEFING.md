# BRIEFING — 2026-10-13T21:12:00Z

## Mission
Formulate implementation plan for R3 Epistemic Logit Synchronization & Knowledge Gating in `BelnapDecisionHead.forward` (`src/yoda/architecture/engine.py`).

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesizer
- Working directory: /home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_3
- Original parent: ead94b72-0198-425b-a453-dfb30ce62240
- Milestone: Milestone 2 (Phase 2 NeSy Reformulation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Ensure candidate logits are gated by candidate knowledge $k$, preventing low knowledge ($k \to 0$) from generating high confidence
- Target files: `src/yoda/architecture/engine.py`, `tests/test_engine.py`, `tests/test_knowledge_gating.py`

## Current Parent
- Conversation ID: ead94b72-0198-425b-a453-dfb30ce62240
- Updated: 2026-10-13T21:12:00Z

## Investigation State
- **Explored paths**:
  - `src/yoda/architecture/engine.py`: `BelnapDecisionHead.forward` lines 29-145, `YodaDecisionEngine.forward` lines 324-445
  - `src/yoda/architecture/belnap_transformer.py`: `BelnapDecisionTransformer` lines 363-482, `BelnapAttention` knowledge gating lines 211-217
  - `tests/e2e/test_paraconsistent_gate.py`: Tier 4 acceptance tests lines 17-80
  - `tests/e2e/test_tier1_feature_coverage.py`: R3/R4 tests lines 332-371
  - `tests/e2e/test_tier2_boundary_corner.py`: R3 zero/maximal knowledge boundaries lines 222-241
  - `tests/e2e/test_tier3_cross_feature.py`: R3 x R4 cross-interaction lines 165-177
  - `tests/test_engine.py`: lines 14-64, lines 167-278, lines 409-459
  - Git log forensic commit `4f6d0ddcd`: historical context where `logits = (choice_pos - choice_neg) * knowledge` was replaced with ungated raw difference
- **Key findings**:
  - `BelnapDecisionHead.forward` currently computes ungated `logits = raw_pos - raw_neg` or `pos_affinity - neg_affinity`, failing to suppress logits when $k \to 0$.
  - When `num_choices == 1`, `BelnapDecisionHead.forward` currently returns early without supporting `candidate_states`, ignoring candidate probes in independent mode.
  - Linear layer projections `w_pos`/`w_neg` map zero evidence ($e_{\text{pos}}=0, e_{\text{neg}}=0$) to $\approx 0$ raw activations, which through standard sigmoid produce $\approx 0.5$ choice evidence ($k = 0.5$) rather than $k \to 0$.
  - Candidate knowledge gating via candidate intrinsic evidence $k_{\text{cand}} = \frac{e_{\text{pos}} + e_{\text{neg}}}{2}$ and gating factor $g = \min(1, 2 k_{\text{cand}})$ ensures low-evidence states smoothly decay to $k=0$ and $|logit| < 10^{-3}$, while preserving full $[0, 1]$ evidence and unattenuated logits for decisive states ($k = 0.5, t = 1.0$).
- **Unexplored areas**:
  - Downstream Phase 3 bilateral inner product scoring details for candidate caching deduplication.

## Key Decisions Made
- Harmonize logit formulation to $\text{logits} = (\text{raw\_diff}) \times \text{knowledge}$ across both linear head and bilinear candidate head modes.
- Compute candidate/input intrinsic knowledge coordinate $k_{\text{cand}}$ and gate choice evidence $e_{\text{pos}}, e_{\text{neg}}$ by $g = \text{clamp}(2 k_{\text{cand}}, \max=1.0)$ so zero-evidence inputs evaluate to $(e_{\text{pos}}=0, e_{\text{neg}}=0, k=0, \text{logits}=0)$.

## Artifact Index
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_3/DISPATCH.md`: Dispatch record from parent orchestrator.
- `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m2_3/handoff.md`: 5-component handoff report.
