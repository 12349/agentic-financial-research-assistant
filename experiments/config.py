"""
experiments/config.py

Single source of truth for all experiment constants.
Every experiment imports from here — no magic numbers in experiment files.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
OLLAMA_TEMPERATURE = 0      # Deterministic output
OLLAMA_SEED = 42            # Passed to Ollama options.seed

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "qwen2.5:7b-instruct"
# Architecture: dense Qwen2.5 7B, 7.6B params
# Quantization: Q4_K_M
# Size on disk: ~4.7 GB
# Vendor family: Alibaba/Qwen
# Roles: planner (S2), no-tools (S3), ReAct (S5), synthesizer (S6)
# NOTE: fits entirely in 16GB RAM (~20-25 tok/s). Extended thinking disabled.

SECOND_MODEL = "llama3.1:8b"
# Architecture: dense Llama 3.1, 8B params
# Quantization: Q4_K_M
# Size on disk: ~4.7 GB
# Vendor family: Meta/Llama (DIFFERENT from PRIMARY_MODEL; satisfies independence requirement)
# Roles: external query generation, second-rater label pass
# STATED LIMITATION: this is a model rater, not a human rater

# ---------------------------------------------------------------------------
# Ollama server
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Hardware (M4 Mac mini, 16 GB unified memory)
# ---------------------------------------------------------------------------
HARDWARE = {
    "model": "Mac16,10",
    "chip": "Apple M4",
    "ram_gb": 16,
    "os": "macOS 27.0",
    "note": (
        "qwen3.6 (23GB) uses NVMe-backed unified memory mapping. "
        "Cold-start load: 3-5 min. Warm inference: ~8-15 tok/s. "
        "llama3.1:8b (4.7GB) fits entirely in RAM. ~25-35 tok/s."
    ),
}

# ---------------------------------------------------------------------------
# Evaluation splits
# ---------------------------------------------------------------------------
EVAL_DIR = "eval"
QUERIES_TEST  = f"{EVAL_DIR}/queries_test.jsonl"    # n=47  — held-out, touch once
QUERIES_DEV   = f"{EVAL_DIR}/queries_dev.jsonl"     # n=45  — tuning/ablations
QUERIES_TRAIN = f"{EVAL_DIR}/queries_train.jsonl"   # n=137 — development only
QUERIES_ALL   = f"{EVAL_DIR}/queries.jsonl"         # n=229

EXTERNAL_QUERIES = f"{EVAL_DIR}/external_queries/external_queries.jsonl"   # n≥50
EXTERNAL_QUERIES_PROMPT = f"{EVAL_DIR}/external_queries/generation_prompt.txt"
LABELING_PROMPT = f"{EVAL_DIR}/external_queries/labeling_prompt.txt"

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
METADATA_FILE = f"{RESULTS_DIR}/metadata.json"

# Per-system raw output directories
RAW_DIR = {
    "s1": f"{RESULTS_DIR}/s1_rule_planner/raw",
    "s2": f"{RESULTS_DIR}/s2_llm_planner/raw",
    "s3": f"{RESULTS_DIR}/s3_no_tools/raw",
    "s4": f"{RESULTS_DIR}/s4_call_all/raw",
    "s5": f"{RESULTS_DIR}/s5_react/raw",
    "s6": f"{RESULTS_DIR}/s6_llm_synth/raw",
}

# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------
KNOWN_TOOLS = frozenset({"search_news", "get_ratings", "get_guidance", "get_earnings"})
MAX_TOOL_CALLS = 4          # Hard cap, matches agent/orchestrator.py

# ---------------------------------------------------------------------------
# Synthesis metrics
# ---------------------------------------------------------------------------
# Tolerance for numeric faithfulness check (e.g. 4.37 vs 4.370 → match)
NUMERIC_TOLERANCE_PCT = 0.5   # 0.5% relative tolerance

# ---------------------------------------------------------------------------
# Latency harness
# ---------------------------------------------------------------------------
LATENCY_N_WARMUP = 2         # Discarded warm-up runs
LATENCY_N_RUNS   = 30        # Timed runs (runs 3–32)

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
CI_ALPHA = 0.05              # 95% CIs

# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------
REACT_MAX_STEPS = 4          # Matches MAX_TOOL_CALLS
REACT_STOP_TOKEN = "Final Answer:"

# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------
FINANCEBENCH_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/"
    "financebench_open_source.json"
)
FINANCEBENCH_LOCAL = f"{RESULTS_DIR}/financebench/financebench_oss.json"
FINANCEBENCH_LICENSE = "Apache 2.0"
FINANCEBENCH_CITE = "Islam et al., arXiv:2311.11944"

# Tickers in our fixture data — for FinanceBench in-scope filter
FIXTURE_TICKERS = frozenset({"NVDA", "TSLA", "JPM", "XOM"})
