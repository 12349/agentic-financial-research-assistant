#!/usr/bin/env bash
# scripts/run_all.sh
#
# Phase 2.95 frozen, reproducible evaluation pipeline.
#
# REQUIREMENTS:
#   - Working tree must be CLEAN (no uncommitted changes). Script refuses otherwise.
#   - Ollama must be running with both models pulled:
#       ollama pull qwen2.5:7b-instruct
#       ollama pull llama3.1:8b
#
# WHAT THIS RUNS (Phase 2.95 — test split n=47, dev where noted):
#   S1  Rule Planner         dev+test  both models (N/A — rule-based, model-independent)
#   S2  LLM Planner          dev+test  qwen2.5:7b-instruct + llama3.1:8b
#   S4  Call-All Baseline    dev+test  model-independent
#   S5  ReAct (production)   dev+test  qwen2.5:7b-instruct + llama3.1:8b
#   S6  LLM Synthesis        dev+test  qwen2.5:7b-instruct + llama3.1:8b
#   S7  Hybrid Synthesis     dev+test  qwen2.5:7b-instruct + llama3.1:8b
#   A1  Cap ablation (S5 production prompt, cap=4)  test  qwen2.5:7b-instruct
#   Label-noise sensitivity  all 50 human-labeled queries  S1+S2
#   Stats                    frozen family definition from experiments/stats_family.json
#   Independent checker      S6+S7 test answers
#   Tables                   from results only
#
# RUNTIME ESTIMATE (Apple M4 16GB, Ollama, greedy decoding, as of Phase 2.95):
#   S1 dev+test          ~2 min   (deterministic, fixture only)
#   S4 dev+test          ~1 min   (deterministic)
#   S2 Qwen  dev+test    ~45 min  (1558 ms/query × 92 queries)
#   S2 Llama dev+test    ~90 min  (estimated 2× Qwen latency for Llama)
#   S5 Qwen  dev+test    ~26 h    (17,140 ms/query × 92 queries)   ← DOMINANT
#   S5 Llama dev+test    ~30 h    (estimated 1.2× Qwen for Llama)  ← DOMINANT
#   S6 Qwen  dev+test    ~15 min  (9,574 ms/query × 92 queries)
#   S6 Llama dev+test    ~30 min  (10,383 ms/query × 92 queries)
#   S7 Qwen  dev+test    ~60 min  (revision loop ~2–3× S6)
#   S7 Llama dev+test    ~90 min
#   A1 cap ablation      ~25 min  (S5 with cap=4 on dev+test)
#   Label noise          ~2 h     (S1+S2 on 40 train + 10 dev queries)
#   Stats + tables       ~5 min
#
#   TOTAL ESTIMATE: ~60–70 hours wall time for full S5 both families.
#   WITH OLLAMA_SKIP_S5=1: ~6–8 hours (skip S5 both families).
#
# Usage:
#   bash scripts/run_all.sh                  # Full pipeline
#   OLLAMA_SKIP_S5=1 bash scripts/run_all.sh # Skip S5 (saves ~55h)
#   OLLAMA_SKIP_LLM=1 bash scripts/run_all.sh # Offline only (S1, S4, tables)
#   DRY_RUN=1 bash scripts/run_all.sh        # Print commands without running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export PYTHONPATH="."
export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"

# ── PHASE 2.95 ITEM 1: DIRTY TREE GUARD ──────────────────────────────────────
# Every result JSON will be stamped with git_commit by common.py.
# A dirty tree makes that commit hash meaningless.
DIRTY_CHECK=$(git status --porcelain 2>/dev/null | head -1)
if [ -n "$DIRTY_CHECK" ]; then
    echo "ERROR: Working tree is dirty. Commit or stash all changes before running." >&2
    echo "  Dirty files:" >&2
    git status --short >&2
    exit 1
fi

COMMIT=$(git rev-parse HEAD)
METRIC_BLOB=$(git hash-object experiments/metrics/synthesis_quality.py)

echo "=========================================================="
echo " Agentic Financial Research Assistant — Phase 2.95 Pipeline"
echo "=========================================================="
echo "Root:         $ROOT_DIR"
echo "Git commit:   $COMMIT"
echo "Metric blob:  $METRIC_BLOB  (experiments/metrics/synthesis_quality.py)"
echo "Date:         $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

_run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

# ── [1] OFFLINE SYSTEMS ───────────────────────────────────────────────────────
echo "--- [1] S1 (Rule Planner) dev + test ---"
_run python3 -m experiments.systems.s1_rule_planner --split dev
_run python3 -m experiments.systems.s1_rule_planner --split test

echo "--- [2] S4 (Call-All Baseline) dev + test ---"
_run python3 -m experiments.systems.s4_call_all --split dev
_run python3 -m experiments.systems.s4_call_all --split test

if [ "${OLLAMA_SKIP_LLM:-0}" = "1" ]; then
    echo "--- Skipping all LLM systems (OLLAMA_SKIP_LLM=1) ---"
else

# ── [2] LLM PLANNER (S2) ──────────────────────────────────────────────────────
echo "--- [3] S2 LLM Planner — qwen2.5:7b-instruct — dev + test ---"
_run python3 -m experiments.systems.s2_llm_planner --split dev   --model qwen2.5:7b-instruct
_run python3 -m experiments.systems.s2_llm_planner --split test  --model qwen2.5:7b-instruct

echo "--- [4] S2 LLM Planner — llama3.1:8b — dev + test ---"
_run python3 -m experiments.systems.s2_llm_planner --split dev   --model llama3.1:8b
_run python3 -m experiments.systems.s2_llm_planner --split test  --model llama3.1:8b

# ── [3] S6 LLM SYNTHESIS ──────────────────────────────────────────────────────
echo "--- [5] S6 LLM Synthesis — qwen2.5:7b-instruct — dev + test ---"
_run python3 -m experiments.systems.s6_llm_synth --split dev   --model qwen2.5:7b-instruct
_run python3 -m experiments.systems.s6_llm_synth --split test  --model qwen2.5:7b-instruct

echo "--- [6] S6 LLM Synthesis — llama3.1:8b — dev + test ---"
_run python3 -m experiments.systems.s6_llm_synth --split dev   --model llama3.1:8b
_run python3 -m experiments.systems.s6_llm_synth --split test  --model llama3.1:8b

# ── [4] S7 HYBRID SYNTHESIS ───────────────────────────────────────────────────
echo "--- [7] S7 Hybrid — qwen2.5:7b-instruct — dev + test ---"
_run python3 -m experiments.systems.s7_hybrid_synth --split dev   --model qwen2.5:7b-instruct
_run python3 -m experiments.systems.s7_hybrid_synth --split test  --model qwen2.5:7b-instruct

echo "--- [8] S7 Hybrid — llama3.1:8b — dev + test ---"
_run python3 -m experiments.systems.s7_hybrid_synth --split dev   --model llama3.1:8b
_run python3 -m experiments.systems.s7_hybrid_synth --split test  --model llama3.1:8b

# ── [5] REACT (S5) ─────────────────────────────────────────────────────────────
if [ "${OLLAMA_SKIP_S5:-0}" = "1" ]; then
    echo "--- Skipping S5 ReAct (OLLAMA_SKIP_S5=1) ---"
else
    echo "--- [9] S5 ReAct (production prompt) — qwen2.5:7b-instruct — dev + test ---"
    _run python3 -m experiments.systems.s5_react --split dev   --model qwen2.5:7b-instruct
    _run python3 -m experiments.systems.s5_react --split test  --model qwen2.5:7b-instruct

    echo "--- [10] S5 ReAct (production prompt) — llama3.1:8b — dev + test ---"
    _run python3 -m experiments.systems.s5_react --split dev   --model llama3.1:8b
    _run python3 -m experiments.systems.s5_react --split test  --model llama3.1:8b

    # ── [6] A1 CAP ABLATION (production S5 code, cap=4) ──────────────────────
    echo "--- [11] A1 Cap Ablation (production S5, cap=4) — qwen2.5:7b-instruct ---"
    _run python3 -m experiments.ablations.a1_tool_cap_live \
        --model qwen2.5:7b-instruct --cap 4 --split dev
    _run python3 -m experiments.ablations.a1_tool_cap_live \
        --model qwen2.5:7b-instruct --cap 4 --split test
fi

# ── [7] LABEL-NOISE SENSITIVITY (S1 + S2 on all 50 human-labeled) ─────────────
echo "--- [12] Label-noise sensitivity (S1 + S2, all 50 human-labeled queries) ---"
_run python3 experiments/label_noise_sensitivity.py

# ── [8] INDEPENDENT CHECKER ──────────────────────────────────────────────────
echo "--- [13] Independent metric checker (S6 + S7 test answers) ---"
_run python3 experiments/independent_checker.py

fi  # end OLLAMA_SKIP_LLM block

# ── [9] STATISTICAL TESTS ────────────────────────────────────────────────────
echo "--- [14] Statistical tests (family defined in experiments/stats_family.json) ---"
_run python3 -m experiments.compute_statistical_tests

# ── [10] TABLES ──────────────────────────────────────────────────────────────
echo "--- [15] Generating publication tables from results/ ---"
_run python3 experiments/generate_tables.py

echo ""
echo "=========================================================="
echo " Pipeline Complete! All artifacts saved to results/"
echo " Commit: $COMMIT"
echo " Metric: $METRIC_BLOB"
echo "=========================================================="
