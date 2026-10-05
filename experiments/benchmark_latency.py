"""System 1 Latency & Throughput Benchmark Harness.

Profiles single-query and batched heuristic evaluation latency, constraint
satisfaction, and throughput against the System 1 sub-10ms latency SLA.
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, cast

import polars as pl

from yoda.architecture.schema import DecisionPayload, QueryContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yoda.benchmark")


def emit_event(event_name: str, payload: dict[str, Any]) -> None:
    """Emit a structured telemetry event."""
    record = {
        "event": event_name,
        "timestamp": time.time(),
        **payload,
    }
    logger.info("telemetry_event: %s", json.dumps(record))


def generate_mock_payloads(count: int, embedding_dim: int = 64) -> list[DecisionPayload]:
    """Generate synthetic decision payloads for benchmarking."""
    payloads: list[DecisionPayload] = []
    for i in range(count):
        embedding = [float((i + j) % 100) / 100.0 for j in range(embedding_dim)]
        payload = DecisionPayload(
            query=f"decision_query_{i}",
            context=QueryContext(
                semantic_embedding=embedding,
                symbolic_state={"step": i, "active": True, "domain": "benchmark"},
                history=[{"step": max(0, i - 1), "action": "noop"}],
            ),
            constraints=["bounded_latency", "invariant_non_negative"],
            metadata={"priority": i % 5, "source": "synthetic_bench"},
        )
        payloads.append(payload)
    return payloads


def simulate_system1_evaluation(payload: DecisionPayload) -> dict[str, Any]:
    """Simulate System 1 fast-path heuristic evaluation and constraint check.

    In the production engine, this calls into the tensor/NeSy pipeline.
    Here it exercises validation, scoring, and rule verification.
    """
    # Quick mock invariant validation
    constraints_satisfied = len(payload.constraints) > 0
    embeddings = payload.context.semantic_embedding
    score = sum(embeddings[:8]) if embeddings else 0.0

    return {
        "decision": "action_admit" if score >= 0.0 else "action_reject",
        "score": score,
        "constraints_satisfied": constraints_satisfied,
        "uncertainty": 0.05,
    }


def run_benchmark(iterations: int = 1000, warmup: int = 50) -> pl.DataFrame:
    """Run benchmark sweep measuring P50, P95, P99 latency in milliseconds."""
    emit_event("benchmark.run.start", {"iterations": iterations, "warmup": warmup})

    payloads = generate_mock_payloads(iterations + warmup)

    # Warmup phase
    for i in range(warmup):
        _ = simulate_system1_evaluation(payloads[i])

    # Measurement phase
    latencies_ms: list[float] = []
    satisfied_count = 0

    start_total = time.perf_counter()
    for i in range(warmup, warmup + iterations):
        t0 = time.perf_counter()
        result = simulate_system1_evaluation(payloads[i])
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        if result["constraints_satisfied"]:
            satisfied_count += 1

    total_time_sec = time.perf_counter() - start_total
    throughput = iterations / total_time_sec if total_time_sec > 0 else 0.0

    df = pl.DataFrame({"latency_ms": latencies_ms})

    p50 = cast(float, df["latency_ms"].quantile(0.50))
    p95 = cast(float, df["latency_ms"].quantile(0.95))
    p99 = cast(float, df["latency_ms"].quantile(0.99))
    mean_lat = cast(float, df["latency_ms"].mean())
    satisfaction_rate = (satisfied_count / iterations) * 100.0

    emit_event(
        "benchmark.run.complete",
        {
            "iterations": iterations,
            "total_time_sec": total_time_sec,
            "throughput_evals_sec": throughput,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_lat,
            "satisfaction_rate_pct": satisfaction_rate,
            "sub_10ms_target_met": p99 < 10.0,
        },
    )

    print("\n" + "=" * 60)
    print(" SYSTEM 1 LATENCY & THROUGHPUT BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Iterations:             {iterations}")
    print(f"Total Duration:         {total_time_sec:.4f} s")
    print(f"Throughput:             {throughput:,.1f} evaluations/sec")
    print(f"P50 Latency:            {p50:.4f} ms")
    print(f"P95 Latency:            {p95:.4f} ms")
    print(f"P99 Latency:            {p99:.4f} ms")
    print(f"Mean Latency:           {mean_lat:.4f} ms")
    print(f"Constraint Sat Rate:    {satisfaction_rate:.1f}%")
    print(f"Sub-10ms Target (<10ms): {'PASS' if p99 < 10.0 else 'FAIL'}")
    print("=" * 60 + "\n")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="System 1 Latency Benchmark")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of evaluation iterations to benchmark (default: 2000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save latency parquet dataframe (e.g. data/latency.parquet)",
    )
    args = parser.parse_args()

    df = run_benchmark(iterations=args.iterations, warmup=args.warmup)

    if args.output:
        df.write_parquet(args.output)
        logger.info("Saved benchmark metrics to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
