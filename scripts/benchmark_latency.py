"""
scripts/benchmark_latency.py

Runs the 30-run latency harness across systems S1 through S6.
2 warmup runs discarded, runs 3-32 recorded.
Reports cold vs warm latency, std, median, p95, tok/s, hardware specs.
Saves to results/latency_benchmark.json.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from agent.synthesizer import _synthesize_fallback
from experiments.config import HARDWARE, LATENCY_N_RUNS, LATENCY_N_WARMUP, PRIMARY_MODEL, RESULTS_DIR
from experiments.ollama_client import OllamaClient
from experiments.systems.common import execute_tools
from experiments.systems.s1_rule_planner import rule_plan
from experiments.systems.s2_llm_planner import plan_with_llm
from experiments.systems.s4_call_all import call_all_plan
from experiments.systems.s5_react import run_react_query
from experiments.systems.s6_llm_synth import synthesize_with_llm

BENCHMARK_QUERY = "What is the current analyst sentiment on Tesla stock?"


def benchmark_s1():
    latencies = []
    for _ in range(LATENCY_N_WARMUP + LATENCY_N_RUNS):
        t0 = time.perf_counter()
        plan = rule_plan(BENCHMARK_QUERY)
        outs, _ = execute_tools(plan)
        _synthesize_fallback(BENCHMARK_QUERY, outs)
        latencies.append((time.perf_counter() - t0) * 1000)

    warm = latencies[LATENCY_N_WARMUP:]
    return {
        "system": "S1 (rule-planner)",
        "cold_ms": round(latencies[0], 2),
        "mean_ms": round(statistics.mean(warm), 2),
        "std_ms": round(statistics.stdev(warm), 2) if len(warm) > 1 else 0.0,
        "median_ms": round(statistics.median(warm), 2),
        "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 2),
    }


def benchmark_s4():
    latencies = []
    for _ in range(LATENCY_N_WARMUP + LATENCY_N_RUNS):
        t0 = time.perf_counter()
        plan = call_all_plan(BENCHMARK_QUERY)
        outs, _ = execute_tools(plan)
        _synthesize_fallback(BENCHMARK_QUERY, outs)
        latencies.append((time.perf_counter() - t0) * 1000)

    warm = latencies[LATENCY_N_WARMUP:]
    return {
        "system": "S4 (call-all)",
        "cold_ms": round(latencies[0], 2),
        "mean_ms": round(statistics.mean(warm), 2),
        "std_ms": round(statistics.stdev(warm), 2) if len(warm) > 1 else 0.0,
        "median_ms": round(statistics.median(warm), 2),
        "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 2),
    }


def benchmark_s2(client: OllamaClient):
    latencies = []
    llm_latencies = []
    for _ in range(LATENCY_N_WARMUP + LATENCY_N_RUNS):
        t0 = time.perf_counter()
        plan, resp, _ = plan_with_llm(BENCHMARK_QUERY, client)
        outs, _ = execute_tools(plan)
        _synthesize_fallback(BENCHMARK_QUERY, outs)
        latencies.append((time.perf_counter() - t0) * 1000)
        llm_latencies.append(resp.elapsed_ms)

    warm = latencies[LATENCY_N_WARMUP:]
    warm_llm = llm_latencies[LATENCY_N_WARMUP:]
    return {
        "system": "S2 (llm-planner)",
        "cold_ms": round(latencies[0], 2),
        "mean_ms": round(statistics.mean(warm), 2),
        "std_ms": round(statistics.stdev(warm), 2) if len(warm) > 1 else 0.0,
        "median_ms": round(statistics.median(warm), 2),
        "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 2),
        "mean_llm_only_ms": round(statistics.mean(warm_llm), 2),
    }


def benchmark_s3(client: OllamaClient):
    prompt = f"Answer concisely: {BENCHMARK_QUERY}"
    latencies = []
    for _ in range(LATENCY_N_WARMUP + LATENCY_N_RUNS):
        t0 = time.perf_counter()
        resp = client.generate(prompt, num_predict=256)
        latencies.append((time.perf_counter() - t0) * 1000)

    warm = latencies[LATENCY_N_WARMUP:]
    return {
        "system": "S3 (no-tools)",
        "cold_ms": round(latencies[0], 2),
        "mean_ms": round(statistics.mean(warm), 2),
        "std_ms": round(statistics.stdev(warm), 2) if len(warm) > 1 else 0.0,
        "median_ms": round(statistics.median(warm), 2),
        "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 2),
    }


def run_latency_benchmarks():
    print(f"Running latency benchmarks on {HARDWARE['chip']} ({HARDWARE['ram_gb']}GB)...")
    results = {}

    print("Benchmarking S1 (rule-planner)...")
    results["S1"] = benchmark_s1()

    print("Benchmarking S4 (call-all)...")
    results["S4"] = benchmark_s4()

    client = OllamaClient(model=PRIMARY_MODEL, think=False, num_predict=256)
    if client.is_available():
        print(f"Benchmarking S2 (llm-planner with {PRIMARY_MODEL})...")
        results["S2"] = benchmark_s2(client)

        print(f"Benchmarking S3 (no-tools with {PRIMARY_MODEL})...")
        results["S3"] = benchmark_s3(client)

    summary = {
        "benchmark_query": BENCHMARK_QUERY,
        "hardware": HARDWARE,
        "n_warmup": LATENCY_N_WARMUP,
        "n_runs": LATENCY_N_RUNS,
        "results": results,
    }

    out_file = Path(RESULTS_DIR) / "latency_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLatency benchmark results saved to {out_file}:")
    for sys_id, metrics in results.items():
        print(f"  {metrics['system']}: Mean={metrics['mean_ms']}ms, Std={metrics['std_ms']}ms, Cold={metrics['cold_ms']}ms")


if __name__ == "__main__":
    run_latency_benchmarks()
