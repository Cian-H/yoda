# Technical Debt Payoff

**Prompt:** "Ok. Very informative! Let's pay down any technical debt here before we get too deep into our design here"

**Actions Taken:**
- Reviewed the Jev AST function quality scan output (`function_smells_report.json`).
- Refactored 5 functions in `src/yoda/data/ingest.py` flagged with `minor_smells`.
- Extracted nested JSON parsing in `_normalize_state` into separate helper methods (`_try_parse_json_dict`, `_try_parse_jsonl_events`), reducing cyclomatic complexity.
- Eliminated 4 redundant streaming methods (`_stream_nimble`, `_stream_kev`, `_stream_dwidlee`, `_stream_synth_and_scenarios`) and merged them into a single data-driven `_stream_datasets` method using explicit mapping tuples.
- Resolved all linter and formatting issues (`E501`, `W293`).
- Validated functionality with `just check` (All 43 tests pass).

**Follow-up Prompt:** "Can you give a quick look at the ones that were hovering on the borderline please too? ... I actually meant take a look at the ones in the attention.py file that were minority smelly"

**Follow-up Actions Taken:**
- Reviewed `fetch_raw_datasets` in `ingest.py` (which had a ~40% minor smell probability) and refactored the massive duplicate list extension block into a declarative `downloads` list.
- Reviewed `parse_nimble_record`, `parse_kev_record`, `parse_n4ze3m_record`, `parse_mghafiri_record` and extracted the duplicated dictionary-building logic into a central `_build_payload` helper.
- Reviewed `MultiheadPooledAttention.forward` in `attention.py` and removed the hacky extraction of `in_proj_weight` from an unused `nn.MultiheadAttention` component, replacing it with native `nn.Linear` layers for clean PyTorch idiom.
- Re-ran tests, confirming `just check` passes perfectly.
- Committed all changes.
