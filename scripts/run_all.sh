#!/usr/bin/env bash
# scripts/run_all.sh
# End-to-end reproducible evaluation pipeline for Phase 2 experiments.
#
# Usage:
#   bash scripts/run_all.sh                 # Full evaluation (offline + LLM)
#   OLLAMA_SKIP_LLM=1 bash scripts/run_all.sh # Offline systems & ablations only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export PYTHONPATH="."
export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"

echo "=========================================================="
echo " Agentic Financial Research Assistant — Phase 2 Pipeline"
echo "=========================================================="
echo "Root: $ROOT_DIR"
echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# 1. Offline Systems (S1, S4)
echo "--- [1/6] Running S1 (Rule Planner) on Dev & Test ---"
python3 -m experiments.systems.s1_rule_planner --split dev
python3 -m experiments.systems.s1_rule_planner --split test

echo "--- [2/6] Running S4 (Call-All Baseline) on Dev & Test ---"
python3 -m experiments.systems.s4_call_all --split dev
python3 -m experiments.systems.s4_call_all --split test

# 2. Offline Ablations
echo "--- [3/6] Running Offline Ablations (A1, A3, A5) ---"
python3 -m experiments.ablations.a1_tool_cap
python3 -m experiments.ablations.a3_sentiment_mode
python3 -m experiments.ablations.a5_search_engine

# 3. FinanceBench External Anchor
echo "--- [4/6] Running FinanceBench Benchmark Evaluation ---"
python3 scripts/benchmark_financebench.py

# 4. LLM Systems (if not skipped)
if [ "${OLLAMA_SKIP_LLM:-0}" -eq 1 ]; then
    echo "--- Skipping LLM systems (OLLAMA_SKIP_LLM=1) ---"
else
    echo "--- [5/6] Running LLM Systems (S2, S3, S5, S6) ---"
    python3 -m experiments.systems.s2_llm_planner --split test
    python3 -m experiments.systems.s3_no_tools --split test
    python3 -m experiments.systems.s6_llm_synth --split test
    python3 -m experiments.systems.s5_react --split test

    echo "--- Running Latency Benchmarks (30 warm runs + cold start) ---"
    python3 scripts/benchmark_latency.py
fi

# 5. Table & Report Generation
echo "--- [6/6] Generating Publication Tables ---"
python3 experiments/generate_tables.py

echo ""
echo "=========================================================="
echo " Pipeline Complete! All artifacts saved to results/"
echo "=========================================================="
