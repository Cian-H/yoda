---
name: system1-eval
description: Benchmark and evaluate System 1 decision engine latency, heuristic accuracy, and throughput against baselines (Jev, Laya, CLM, Julia-1). Use when profiling low-latency inference, running benchmark sweeps, or comparing decision paths.
---

# System 1 Evaluation & Benchmarking

Guide and procedure for running low-latency benchmark sweeps and evaluating the Yoda System 1 decision engine against reference baselines.

## Objectives

- Profile single-query and batched inference latency (target: sub-10ms heuristic evaluation).
- Measure heuristic decision agreement and calibrate epistemic uncertainty.
- Compare decisions and prune paths against baselines (e.g. Jev, Laya, CLM, Julia-1).

## Procedure

1. **Verify Environment**:
   Ensure `devenv` shell is active or run through `uv`:
   ```bash
   uv run pytest tests/
   ```

2. **Benchmarking Ingestion & Execution**:
   - Construct benchmark datasets or mock evaluation payloads using `yoda.architecture.schema.DecisionPayload`.
   - Run the benchmark harness in `experiments/`:
     ```bash
     uv run python experiments/benchmark_latency.py
     ```
   - If using Marimo for interactive exploration:
     ```bash
     uv run marimo edit notebooks/workbench.py
     ```

3. **Baseline Comparison via MCP (`jev`)**:
   When evaluating decision agreement or comparing Yoda's candidate pruning against baseline decision semantics, invoke the `jev` MCP tools:
   - Call `jev_decide` passing the same candidate set, evidence, and priorities to obtain Jev's reference choice distribution and per-candidate requirement satisfaction.
   - Use `jev_compare` to measure semantic consistency between Yoda's generated decision rationales and reference baseline reasoning traces.

4. **Metrics Collection**:
   Record:
   - P50, P95, and P99 latency (ms).
   - Constraint satisfaction rate (%).
   - Uncertainty distribution (epistemic vs aleatoric).
   - Baseline decision agreement rate (% vs Jev/Laya).

5. **Telemetry & Audit**:
   - Verify that all benchmark runs emit structured log events (`benchmark.run.start`, `benchmark.run.complete`).
   - Store results under `experiments/` or `data/` using `polars` / `safetensors`.
