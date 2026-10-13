# Dispatch for Explorer M1_3 (Phase 1: Pre-Activation LayerNorm & Residual Highway)

**Task**: Formulate implementation strategy for Pre-Activation Normalization and Residual Highway $\alpha=0.15$.
**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Target Files**:
- `src/yoda/architecture/belnap_transformer.py`
- `tests/test_optuna_study_narrow.py`
- `experiments/optuna_study_narrow.py`

**Instructions**:
1. Read `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/cianh/Projects/yoda/PROJECT.md`.
2. Formulate the insertion of `nn.LayerNorm(self.d_model)` before `torch.sigmoid(self.w_out_pos(...))` and `torch.sigmoid(self.w_out_neg(...))` in `BelnapAttention`, and in `BelnapFFN`.
3. Formulate recalibrating default `residual_weight: float = 0.15` in `BelnapTransformerBlock` and `_residual_join`.
4. Inspect `experiments/optuna_study_narrow.py` and `tests/test_optuna_study_narrow.py` to identify tests asserting fixed residual weights (e.g. 0.6356) and how they must be adjusted.
5. Detail unit tests for `tests/test_prenorm_highway.py`.
6. Write your report to `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/handoff.md` and notify orchestrator.


## 2026-10-13T20:05:06Z
sender=ead94b72-0198-425b-a453-dfb30ce62240 priority=MESSAGE_PRIORITY_HIGH
You are Explorer M1_3 for Phase 1. Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/DISPATCH.md.
Focus on R2: Pre-Activation Normalization & Residual Gradient Highway.
Investigate src/yoda/architecture/belnap_transformer.py, experiments/optuna_study_narrow.py, tests/test_optuna_study_narrow.py.
Formulate the exact implementation plan for:
1. Adding LayerNorm before output sigmoids in BelnapAttention and BelnapFFN.
2. Recalibrating default residual_weight to 0.15 in BelnapTransformerBlock and _residual_join.
3. Updating tests asserting fixed residual weights (tests/test_optuna_study_narrow.py).
4. New unit tests in tests/test_prenorm_highway.py.
Write handoff report to /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_3/handoff.md and report back via send_message.
