#!/usr/bin/env bash
# scripts/run_all.sh
#
# Phase 2.95 frozen, reproducible evaluation pipeline.
# Ordered by model to avoid Ollama model-reload overhead.
# Resumable: skips any step whose summary JSON already carries the current git_commit.
# Logs to logs/run_TIMESTAMP.log.
#
# DIRTY-TREE GUARD: refuses to run if working tree has uncommitted changes.
# Every result JSON is stamped with git_commit + metric_blob_hash by code (common.py).
#
# RUNTIME ESTIMATE (from logged per-query latencies, Apple M4 16 GB):
#
#   Source data and arithmetic:
#     S1  rule-based:  mean 34 ms/q (dev+test=92q)  92×0.034s  =   3 s
#     S4  call-all:    mean 22 ms/q (92q)            92×0.022s  =   2 s
#     S2  Qwen:        mean 1,558 ms/q (92q)         92×1.558   = 143 s  (2.4 min)
#     S2  Llama:       mean 2,846 ms/q (92q)         92×2.846   = 262 s  (4.4 min)
#     S5  Qwen:        mean 17,140 ms/q (92q)        92×17.140  =1,577 s (26 min)
#     S5  Llama:       estimated 1.1× S5 Qwen        92×18.854  =1,735 s (29 min)
#     S6  Qwen:        mean 12,983 ms/q dev (45q),   45×12.983  = 584 s
#                      mean  9,643 ms/q test (47q)   47× 9.643  = 453 s  =17 min total
#     S6  Llama:       mean 10,383 ms/q (92q)        92×10.383  = 955 s  (16 min)
#     S7  Qwen:        mean 14,685 ms/q (92q)        92×14.685  =1,351 s (23 min)
#     S7  Llama:       mean 18,964 ms/q (92q)        92×18.964  =1,745 s (29 min)
#     A1  Qwen cap=4:  2 runs × 92q × 17,140 ms      184×17.140 =3,154 s (53 min)
#     Label noise S2:  50q × 1,558 ms                50×1.558   =  78 s  ( 1 min)
#
#   TOTAL ESTIMATE: ~3.3 hours
#     Qwen steps: S1+S4+S2Q+S5Q+S6Q+S7Q+A1 ≈ 2.0 h
#     Llama steps: S2L+S5L+S6L+S7L         ≈ 1.3 h
#     (Previous 55h estimate was wrong — confused seconds with milliseconds)
#
# Usage:
#   bash scripts/run_all.sh          # Full pipeline (~3.3 h)
#   DRY_RUN=1 bash scripts/run_all.sh # Print commands without running
#   RESUME=1 bash scripts/run_all.sh  # Skip steps whose summary has current commit
#
# NOTE: S5 must run. There is no skip option for S5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export PYTHONPATH="."
export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"

# ── DIRTY TREE GUARD ─────────────────────────────────────────────────────────
DIRTY_CHECK=$(git status --porcelain 2>/dev/null | head -1)
if [ -n "$DIRTY_CHECK" ]; then
    echo "ERROR: Working tree is dirty. Commit or stash all changes before running." >&2
    git status --short >&2
    exit 1
fi

COMMIT=$(git rev-parse HEAD)
METRIC_BLOB=$(git hash-object experiments/metrics/synthesis_quality.py)

# ── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u '+%Y%m%dT%H%M%SZ')
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
# Tee all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo " Agentic Financial Research Assistant — Phase 2.95 Pipeline"
echo "=========================================================="
echo "Root:         $ROOT_DIR"
echo "Git commit:   $COMMIT"
echo "Metric blob:  $METRIC_BLOB"
echo "Date:         $TIMESTAMP"
echo "Log:          $LOG_FILE"
echo ""

# ── HELPERS ──────────────────────────────────────────────────────────────────
_run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    echo "[RUN] $*"
    "$@"
}

# Check if a summary JSON already carries the current commit stamp.
# $1 = path to summary JSON file
_is_done() {
    local f="$1"
    if [ "${RESUME:-0}" != "1" ]; then
        return 1  # always re-run when not in resume mode
    fi
    if [ ! -f "$f" ]; then
        return 1  # file doesn't exist yet
    fi
    local stamped
    stamped=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('git_commit',''))" 2>/dev/null || echo "")
    if [ "$stamped" = "$COMMIT" ]; then
        echo "  SKIP (already stamped with $COMMIT): $f"
        return 0
    fi
    return 1
}

_step_header() {
    echo ""
    echo "━━━ $1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: MODEL-INDEPENDENT (S1, S4) — no Ollama required
# ══════════════════════════════════════════════════════════════════════════════

_step_header "[1/16] S1 Rule Planner — dev"
if ! _is_done "results/s1_rule_planner/summary_dev.json"; then
    _run python3 -m experiments.systems.s1_rule_planner --split dev
fi

_step_header "[2/16] S1 Rule Planner — test"
if ! _is_done "results/s1_rule_planner/summary_test.json"; then
    _run python3 -m experiments.systems.s1_rule_planner --split test
fi

_step_header "[3/16] S4 Call-All Baseline — dev + test"
if ! _is_done "results/s4_call_all/summary_dev.json"; then
    _run python3 -m experiments.systems.s4_call_all --split dev
fi
if ! _is_done "results/s4_call_all/summary_test.json"; then
    _run python3 -m experiments.systems.s4_call_all --split test
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: QWEN2.5:7B-INSTRUCT — all steps, model loaded once
# Est: S2 2.4 min + S5 26 min + S6 17 min + S7 23 min + A1 53 min = ~2.0 h
# ══════════════════════════════════════════════════════════════════════════════

_step_header "[4/16] S2 LLM Planner (Qwen) — dev"
if ! _is_done "results/s2_llm_planner/summary_dev.json"; then
    _run python3 -m experiments.systems.s2_llm_planner --split dev --model qwen2.5:7b-instruct
fi

_step_header "[5/16] S2 LLM Planner (Qwen) — test"
if ! _is_done "results/s2_llm_planner/summary_test.json"; then
    _run python3 -m experiments.systems.s2_llm_planner --split test --model qwen2.5:7b-instruct
fi

_step_header "[6/16] S5 ReAct production prompt (Qwen) — dev"
if ! _is_done "results/s5_react/summary_dev.json"; then
    _run python3 -m experiments.systems.s5_react --split dev --model qwen2.5:7b-instruct
fi

_step_header "[7/16] S5 ReAct production prompt (Qwen) — test"
if ! _is_done "results/s5_react/summary_test.json"; then
    _run python3 -m experiments.systems.s5_react --split test --model qwen2.5:7b-instruct
fi

_step_header "[8/16] A1 Cap ablation via production S5 (Qwen, cap=4 + unlimited, dev + test)"
if ! _is_done "results/ablations/a1_cap_via_s5.json"; then
    _run python3 -m experiments.ablations.a1_cap_via_s5 --model qwen2.5:7b-instruct --cap 4 --split dev
    _run python3 -m experiments.ablations.a1_cap_via_s5 --model qwen2.5:7b-instruct --cap 4 --split test
fi

_step_header "[9/16] S6 LLM Synthesis (Qwen) — dev"
if ! _is_done "results/s6_llm_synth/summary_dev.json"; then
    _run python3 -m experiments.systems.s6_llm_synth --split dev --model qwen2.5:7b-instruct
fi

_step_header "[10/16] S6 LLM Synthesis (Qwen) — test"
if ! _is_done "results/s6_llm_synth/summary_test.json"; then
    _run python3 -m experiments.systems.s6_llm_synth --split test --model qwen2.5:7b-instruct
fi

_step_header "[11/16] S7 Hybrid Synthesis (Qwen) — dev"
if ! _is_done "results/s7_hybrid_synth/summary_dev.json"; then
    _run python3 -m experiments.systems.s7_hybrid_synth --split dev --model qwen2.5:7b-instruct
fi

_step_header "[12/16] S7 Hybrid Synthesis (Qwen) — test"
if ! _is_done "results/s7_hybrid_synth/summary_test.json"; then
    _run python3 -m experiments.systems.s7_hybrid_synth --split test --model qwen2.5:7b-instruct
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C: LLAMA3.1:8B — all steps, model loaded once
# Est: S2 4.4 min + S5 29 min + S6 16 min + S7 29 min = ~1.3 h
# ══════════════════════════════════════════════════════════════════════════════

_step_header "[13/16] S2 LLM Planner (Llama) — dev + test"
if ! _is_done "results/s2_llama3.1_8b/summary_dev.json"; then
    _run python3 -m experiments.systems.s2_llm_planner --split dev  --model llama3.1:8b
fi
if ! _is_done "results/s2_llama3.1_8b/summary_test.json"; then
    _run python3 -m experiments.systems.s2_llm_planner --split test --model llama3.1:8b
fi

_step_header "[14/16] S5 ReAct production prompt (Llama) — dev + test"
if ! _is_done "results/s5_react_llama3.1_8b/summary_dev.json"; then
    _run python3 -m experiments.systems.s5_react --split dev  --model llama3.1:8b
fi
if ! _is_done "results/s5_react_llama3.1_8b/summary_test.json"; then
    _run python3 -m experiments.systems.s5_react --split test --model llama3.1:8b
fi

_step_header "[15/16] S6 LLM Synthesis (Llama) — dev + test"
if ! _is_done "results/s6_llama3.1_8b/summary_dev.json"; then
    _run python3 -m experiments.systems.s6_llm_synth --split dev  --model llama3.1:8b
fi
if ! _is_done "results/s6_llama3.1_8b/summary_test.json"; then
    _run python3 -m experiments.systems.s6_llm_synth --split test --model llama3.1:8b
fi

_step_header "[16/16] S7 Hybrid Synthesis (Llama) — dev + test"
if ! _is_done "results/s7_hybrid_llama3.1_8b/summary_dev.json"; then
    _run python3 -m experiments.systems.s7_hybrid_synth --split dev  --model llama3.1:8b
fi
if ! _is_done "results/s7_hybrid_llama3.1_8b/summary_test.json"; then
    _run python3 -m experiments.systems.s7_hybrid_synth --split test --model llama3.1:8b
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE D: POST-RUN ANALYSIS (offline, no model required)
# ══════════════════════════════════════════════════════════════════════════════

_step_header "[Post] Label-noise sensitivity (S1 offline + S2 Qwen)"
_run python3 experiments/label_noise_sensitivity.py --model qwen2.5:7b-instruct

_step_header "[Post] Independent checker (S6 + S7 test answers)"
_run python3 experiments/independent_checker.py

_step_header "[Post] Statistical tests (family: experiments/stats_family.json)"
_run python3 -m experiments.compute_statistical_tests

_step_header "[Post] Publication tables"
_run python3 experiments/generate_tables.py

_step_header "[Post] Old-vs-new comparison"
_run python3 scripts/compare_results.py

echo ""
echo "=========================================================="
echo " Pipeline Complete — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo " Commit: $COMMIT"
echo " Metric: $METRIC_BLOB"
echo " Log:    $LOG_FILE"
echo "=========================================================="
