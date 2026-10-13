# Dispatch for Explorer M1_1 (Phase 1: Attention Masking & Scale Normalization)

**Task**: Formulate implementation strategy for R1 (Attention Mask Propagation & Scale Normalization).
**Authoritative Request**: `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md`
**Project Plan**: `/home/cianh/Projects/yoda/PROJECT.md`
**Target Files**:
- `src/yoda/architecture/encoders.py`
- `src/yoda/architecture/attention.py`
- `src/yoda/architecture/belnap_transformer.py`
- `src/yoda/architecture/engine.py`

**Instructions**:
1. Read `/home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md` and `/home/cianh/Projects/yoda/PROJECT.md`.
2. Analyze how to implement `TextEncoder` mask return while maintaining backwards compatibility with any callers expecting a single tensor.
3. Formulate the exact tensor operations for padding mask injection into `BelnapAttention.forward` ($s \leftarrow s.\text{masked\_fill}(\neg\text{mask}, -1\text{e}9)$) and active sequence mass normalization $\text{agg} / \text{denom}$.
4. Detail the unit tests to add in `tests/test_encoders.py` and `tests/test_attention_masking.py`.
5. Write your report to `/home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/handoff.md` and notify orchestrator.

## 2026-10-13T20:05:06Z
You are Explorer M1_1 for Phase 1. Your working directory is /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1.
Read /home/cianh/Projects/yoda/.agents/teamwork/ORIGINAL_REQUEST.md first.
Read /home/cianh/Projects/yoda/PROJECT.md.
Read /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/DISPATCH.md.
Focus on R1: Attention Mask Propagation & Scale Normalization.
Investigate src/yoda/architecture/encoders.py, attention.py, belnap_transformer.py, engine.py.
Formulate the exact implementation plan for:
1. Returning mask from TextEncoder while keeping backward compatibility.
2. Padding mask handling in BelnapAttention.
3. Active sequence mass normalization in BelnapAttention aggregation.
4. Unit tests to verify in tests/test_encoders.py and new tests/test_attention_masking.py.
Write handoff report to /home/cianh/Projects/yoda/.agents/teamwork/explorer_m1_1/handoff.md and report back via send_message.
