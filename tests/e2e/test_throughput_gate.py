"""Tier 4 Acceptance Gate: Throughput Benchmark Gate.

Measures independent candidate evaluation throughput speedup >= 2.0x
achieved via context encoding deduplication caching.
"""

import time
from typing import Any

import torch

from yoda.architecture.engine import YodaDecisionEngine


def _benchmark_evaluation_throughput(
    engine: YodaDecisionEngine,
    queries: list[str],
    states: list[dict[str, Any]],
    candidates: list[str],
    iterations: int = 15,
) -> float:
    """Measures average throughput (candidate evaluations per second)."""
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            engine(queries=queries, states=states, candidates=candidates)

    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            engine(queries=queries, states=states, candidates=candidates)
    elapsed = time.perf_counter() - start_time

    total_evals = len(candidates) * iterations
    return total_evals / max(elapsed, 1e-6)


def test_independent_candidate_evaluation_throughput_speedup() -> None:
    """Acceptance Gate: Independent candidate evaluation throughput speedup must be >= 2.0x."""
    # Workload: 8 groups with 5 choices each (40 candidate evaluations per pass)
    num_groups = 8
    choices_per_group = 5

    queries: list[str] = []
    states: list[dict[str, Any]] = []
    candidates: list[str] = []

    for g in range(num_groups):
        q = f"Telemetry question {g} requiring multi-stage reasoning and verification"
        s = {"sensor_group": g, "calibration_level": 4, "mode": "autonomous"}
        for c in range(choices_per_group):
            queries.append(q)
            states.append(s)
            candidates.append(f"Action proposal {c} for question {g}")

    # Baseline uncached engine: redundant backbone encoding
    uncached_engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=64,
        num_choices=1,
        n_heads=2,
        num_reasoning_blocks=1,
        use_context_cache=False,
    )

    # Hardened cached engine
    cached_engine = YodaDecisionEngine(
        text_model_name="dummy",
        embed_dim=64,
        num_choices=1,
        n_heads=2,
        num_reasoning_blocks=1,
    )

    uncached_throughput = _benchmark_evaluation_throughput(
        uncached_engine,
        queries=queries,
        states=states,
        candidates=candidates,
        iterations=15,
    )

    cached_throughput = _benchmark_evaluation_throughput(
        cached_engine,
        queries=queries,
        states=states,
        candidates=candidates,
        iterations=15,
    )

    # Check if context deduplication cache is implemented on the engine
    assert hasattr(cached_engine, "_context_cache") or hasattr(cached_engine, "cached_forward"), (
        "YodaDecisionEngine must implement context encoding deduplication cache"
    )

    # Measure speedup against baseline where each unrolled sample executes its own encoding pass
    speedup = cached_throughput / max(uncached_throughput, 1e-6)
    assert speedup >= 2.0, (
        f"Independent candidate evaluation speedup reached {speedup:.2f}x (required >= 2.0x)."
    )
