"""
experiments/metrics/latency.py

30-run latency harness for all systems.

Protocol:
  - 2 warm-up runs (results discarded)
  - 30 timed runs (runs 3–32)
  - Cold start: first call after process start (model not in Ollama's cache)
  - Warm: runs 3–32 (model loaded, KV cache cleared between runs via new session)
  - Reports: mean, std, median, p95, cold separately
  - Token counts reported for cost / efficiency comparison ($0 budget)

IMPORTANT: Only call this harness when the model is already loaded (after
a warm-up call has completed). Do not call for offline systems (S1, S4)
since they have sub-millisecond latency; just report the measured wall time.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from experiments.config import LATENCY_N_RUNS, LATENCY_N_WARMUP, OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# Timing result
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    system_id:       str
    model:           str
    n_runs:          int
    cold_ms:         float       # First call after model load
    mean_ms:         float       # Mean of 30 warm runs
    std_ms:          float
    median_ms:       float
    p95_ms:          float
    mean_prompt_tps: float       # Prompt tokens / sec (warm)
    mean_gen_tps:    float       # Generation tokens / sec (warm)
    mean_gen_tokens: float       # Mean output tokens per call
    all_ms:          list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "system_id":       self.system_id,
            "model":           self.model,
            "n_runs":          self.n_runs,
            "cold_ms":         self.cold_ms,
            "mean_ms":         self.mean_ms,
            "std_ms":          self.std_ms,
            "median_ms":       self.median_ms,
            "p95_ms":          self.p95_ms,
            "mean_prompt_tps": self.mean_prompt_tps,
            "mean_gen_tps":    self.mean_gen_tps,
            "mean_gen_tokens": self.mean_gen_tokens,
        }


# ---------------------------------------------------------------------------
# Core timing function
# ---------------------------------------------------------------------------

def time_ollama_call(
    model: str,
    prompt: str,
    think: bool = False,
    num_predict: int = 256,
    temperature: float = 0,
    seed: int = 42,
    base_url: str = OLLAMA_BASE_URL,
) -> tuple[float, dict]:
    """
    Time a single Ollama generate call. Returns (elapsed_ms, raw_response_dict).
    elapsed_ms is wall-clock time; raw contains Ollama's internal timing fields.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        },
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read())
    elapsed_ms = (time.monotonic() - t0) * 1000
    return elapsed_ms, raw


def _percentile(xs: list[float], p: float) -> float:
    """p-th percentile of xs (0.0–100.0)."""
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Latency harness
# ---------------------------------------------------------------------------

def run_latency_harness(
    system_id: str,
    model: str,
    prompt: str,
    n_warmup: int = LATENCY_N_WARMUP,
    n_runs: int = LATENCY_N_RUNS,
    think: bool = False,
    num_predict: int = 256,
    verbose: bool = True,
) -> LatencyResult:
    """
    Run the latency harness for an LLM-based system.

    Steps:
      1. n_warmup warm-up calls (results discarded)
      2. n_runs timed calls
      3. Compute statistics and return LatencyResult

    Note: This does NOT measure cold-start (model-load) time.
    Cold start is measured separately after an explicit model unload.
    Pass cold_ms from an earlier measurement or set to 0 if not available.
    """
    if verbose:
        print(f"  Latency harness: {system_id} ({model})")
        print(f"  Warm-up: {n_warmup} runs | Timed: {n_runs} runs")

    # Warm-up (discard)
    for i in range(n_warmup):
        time_ollama_call(model, prompt, think=think, num_predict=num_predict)
        if verbose:
            print(f"  Warm-up {i+1}/{n_warmup} done")

    # Timed runs
    elapsed_times: list[float] = []
    prompt_tps_list: list[float] = []
    gen_tps_list: list[float] = []
    gen_token_counts: list[float] = []

    for i in range(n_runs):
        ms, raw = time_ollama_call(model, prompt, think=think, num_predict=num_predict)
        elapsed_times.append(ms)

        eval_ns = raw.get("eval_duration", 1)
        p_eval_ns = raw.get("prompt_eval_duration", 1)
        gen_tks = raw.get("eval_count", 0)
        p_tks = raw.get("prompt_eval_count", 0)

        prompt_tps_list.append(p_tks / (p_eval_ns / 1e9) if p_eval_ns else 0)
        gen_tps_list.append(gen_tks / (eval_ns / 1e9) if eval_ns else 0)
        gen_token_counts.append(gen_tks)

        if verbose and (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{n_runs}: {ms:.0f}ms "
                  f"({gen_tks} tok, {gen_tps_list[-1]:.1f} tok/s)")

    mean_ms  = sum(elapsed_times) / len(elapsed_times)
    std_ms   = _stddev(elapsed_times)
    sorted_t = sorted(elapsed_times)
    median   = _percentile(elapsed_times, 50)
    p95      = _percentile(elapsed_times, 95)

    result = LatencyResult(
        system_id       = system_id,
        model           = model,
        n_runs          = n_runs,
        cold_ms         = 0.0,  # Set externally from cold-start measurement
        mean_ms         = round(mean_ms, 1),
        std_ms          = round(std_ms, 1),
        median_ms       = round(median, 1),
        p95_ms          = round(p95, 1),
        mean_prompt_tps = round(sum(prompt_tps_list) / len(prompt_tps_list), 1),
        mean_gen_tps    = round(sum(gen_tps_list) / len(gen_tps_list), 1),
        mean_gen_tokens = round(sum(gen_token_counts) / len(gen_token_counts), 1),
        all_ms          = [round(x, 1) for x in elapsed_times],
    )

    if verbose:
        print(f"  Results: mean={result.mean_ms}ms std={result.std_ms}ms "
              f"p95={result.p95_ms}ms gen={result.mean_gen_tps}tok/s")

    return result


def measure_cold_start(
    model: str,
    prompt: str,
    base_url: str = OLLAMA_BASE_URL,
) -> float:
    """
    Measure cold-start time: unload model, wait 2s, then time first call.
    Returns elapsed_ms for the cold call.

    IMPORTANT: This will unload the model from Ollama's cache.
    Call this BEFORE the warm harness, not after.
    """
    # Unload by sending keep_alive=0
    payload = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps({"model": model, "keep_alive": "0s", "prompt": "", "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass  # If it fails, the model may not be loaded — that's fine

    time.sleep(2)  # Let Ollama finish unloading

    # Cold call
    ms, _ = time_ollama_call(model, prompt, num_predict=32)
    return round(ms, 1)


def save_latency_result(result: LatencyResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = result.as_dict()
    data["all_ms"] = result.all_ms  # Include raw timings for reproducibility
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Offline system timing (S1, S4 — no LLM)
# ---------------------------------------------------------------------------

def time_offline_system(
    system_id: str,
    run_fn: Callable[[dict], dict],
    queries: list[dict],
    n_runs: int = LATENCY_N_RUNS,
) -> LatencyResult:
    """
    Time an offline system (no LLM — just function calls).
    Uses the same harness structure but calls run_fn directly.
    """
    # Warm up with first 2 queries
    for q in queries[:LATENCY_N_WARMUP]:
        run_fn(q)

    elapsed_times = []
    for i in range(n_runs):
        q = queries[i % len(queries)]
        t0 = time.monotonic()
        run_fn(q)
        elapsed_times.append((time.monotonic() - t0) * 1000)

    mean_ms = sum(elapsed_times) / len(elapsed_times)
    return LatencyResult(
        system_id       = system_id,
        model           = "none (offline)",
        n_runs          = n_runs,
        cold_ms         = 0.0,
        mean_ms         = round(mean_ms, 3),
        std_ms          = round(_stddev(elapsed_times), 3),
        median_ms       = round(_percentile(elapsed_times, 50), 3),
        p95_ms          = round(_percentile(elapsed_times, 95), 3),
        mean_prompt_tps = 0.0,
        mean_gen_tps    = 0.0,
        mean_gen_tokens = 0.0,
        all_ms          = [round(x, 3) for x in elapsed_times],
    )
