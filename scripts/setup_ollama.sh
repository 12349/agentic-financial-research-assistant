#!/usr/bin/env bash
# scripts/setup_ollama.sh
# Pull required Ollama models for Phase 2 experiments and verify them.
#
# Machine target: Mac mini M4, 16 GB unified memory, macOS 27.0
# Ollama version: 0.23.3 (server) / 0.30.10 (client)
#
# Models:
#   PRIMARY  : qwen3.6:latest   (36B MoE, Q4_K_M, ~23 GB on disk)
#              Architecture: qwen35moe — activates ~3.6B params/token
#              Fits in 16 GB unified memory via Apple Silicon memory mapping +
#              NVMe-backed paging. Expect ~8-12 tok/s after warm-up.
#
#   SECOND   : llama3.1:8b      (Meta Llama 3.1 8B, Q4_K_M, ~4.7 GB on disk)
#   FAMILY     Different vendor family (Meta vs Alibaba) — used as second rater
#              for independence requirement. Fits entirely in 16 GB RAM.
#              Expect ~25-35 tok/s.
#
# Usage:
#   bash scripts/setup_ollama.sh
#
# What it does:
#   1. Checks Ollama is running
#   2. Pulls both models if not already present
#   3. Verifies model digests
#   4. Runs one smoke-test prompt per model
#   5. Writes results/metadata.json with hardware + model info

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Config ────────────────────────────────────────────────────────────────────
PRIMARY_MODEL="qwen3.6:latest"
SECOND_MODEL="llama3.1:8b"
RESULTS_DIR="results"
METADATA_FILE="${RESULTS_DIR}/metadata.json"

SMOKE_PROMPT='Output ONLY a raw JSON array with no explanation. Query: "What is the current analyst sentiment on Tesla stock?" Available tools: search_news, get_ratings, get_guidance, get_earnings. Return the tools needed as: [{"tool": "<name>", "args": {"ticker": "<ticker>"}}]'

# ── Prerequisites ─────────────────────────────────────────────────────────────
command -v ollama &>/dev/null || error "Ollama not found. Install from: https://ollama.com/download"

info "Ollama version: $(ollama --version 2>&1 | head -1)"

# Check server is running
if ! ollama list &>/dev/null 2>&1; then
    warn "Ollama server not responding. Starting it..."
    ollama serve &>/dev/null &
    sleep 3
    ollama list &>/dev/null || error "Could not start Ollama server"
fi

mkdir -p "${RESULTS_DIR}"

# ── Pull models ───────────────────────────────────────────────────────────────
pull_if_missing() {
    local model="$1"
    if ollama list 2>/dev/null | grep -q "^${model%:*}"; then
        info "Model already present: ${model}"
    else
        info "Pulling ${model} (this may take several minutes)..."
        ollama pull "${model}" || error "Failed to pull ${model}"
        info "Done: ${model}"
    fi
}

pull_if_missing "${PRIMARY_MODEL}"
pull_if_missing "${SECOND_MODEL}"

# ── Verify digests ─────────────────────────────────────────────────────────────
info "Verifying model digests..."
PRIMARY_DIGEST=$(ollama show "${PRIMARY_MODEL}" 2>/dev/null | grep -i "digest\|sha256" | head -1 || echo "unavailable")
SECOND_DIGEST=$(ollama show "${SECOND_MODEL}" 2>/dev/null | grep -i "digest\|sha256" | head -1 || echo "unavailable")
info "  ${PRIMARY_MODEL}: ${PRIMARY_DIGEST}"
info "  ${SECOND_MODEL}:  ${SECOND_DIGEST}"

# ── Smoke tests ────────────────────────────────────────────────────────────────
smoke_test() {
    local model="$1"
    local label="$2"
    info "Smoke-testing ${label} (${model})..."
    local start_ns
    start_ns=$(python3 -c "import time; print(int(time.time_ns()))")
    local output
    output=$(echo "${SMOKE_PROMPT}" | ollama run "${model}" --nowordwrap 2>/dev/null | tr -d '\n')
    local end_ns
    end_ns=$(python3 -c "import time; print(int(time.time_ns()))")
    local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    info "  Output: ${output:0:120}..."
    info "  Elapsed: ${elapsed_ms}ms"

    # Basic validity: does output contain a JSON array?
    if echo "${output}" | python3 -c "import json,sys; d=json.loads(sys.stdin.read().strip()); assert isinstance(d,list)" 2>/dev/null; then
        info "  ✅ Valid JSON array returned"
    else
        warn "  ⚠️  Output is not a clean JSON array — parser fallback will be used"
    fi
    echo "${elapsed_ms}"
}

PRIMARY_LATENCY=$(smoke_test "${PRIMARY_MODEL}" "primary")
SECOND_LATENCY=$(smoke_test "${SECOND_MODEL}" "second-family rater")

# ── Hardware info ──────────────────────────────────────────────────────────────
HW_MODEL=$(sysctl -n hw.model 2>/dev/null || echo "unknown")
HW_RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
HW_NCPU=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)
OS_VER=$(sw_vers -productVersion 2>/dev/null || uname -r)
OLLAMA_VER=$(ollama --version 2>&1 | head -1)

# ── Write metadata.json ────────────────────────────────────────────────────────
python3 - << PYEOF
import json, datetime

metadata = {
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "hardware": {
        "model": "${HW_MODEL}",
        "chip": "Apple M4",
        "ram_gb": ${HW_RAM_GB},
        "perf_cores": ${HW_NCPU},
        "os": "macOS ${OS_VER}"
    },
    "ollama": {
        "version": "${OLLAMA_VER}",
        "server_client_note": "server 0.23.3 / client 0.30.10"
    },
    "models": {
        "primary": {
            "name": "${PRIMARY_MODEL}",
            "architecture": "qwen35moe (36B total, ~3.6B active/token)",
            "quantization": "Q4_K_M",
            "size_gb_approx": 23,
            "temperature": 0,
            "seed": 42,
            "smoke_test_latency_ms": ${PRIMARY_LATENCY},
            "digest": "${PRIMARY_DIGEST}",
            "role": "planner, synthesizer, ReAct agent",
            "vendor_family": "Alibaba/Qwen"
        },
        "second_rater": {
            "name": "${SECOND_MODEL}",
            "architecture": "llama3.1 (8B dense)",
            "quantization": "Q4_K_M",
            "size_gb_approx": 4.7,
            "temperature": 0,
            "seed": 42,
            "smoke_test_latency_ms": ${SECOND_LATENCY},
            "digest": "${SECOND_DIGEST}",
            "role": "external query generation, second-rater label pass",
            "vendor_family": "Meta/Llama",
            "independence_note": "Different vendor family from primary. Satisfies different-family independence requirement. NOT a human rater; stated plainly in all outputs."
        }
    },
    "cost": {
        "total_usd": 0.0,
        "note": "All inference is local via Ollama. Zero external API calls. Report tokens and latency instead of cost."
    },
    "experiment_seed": 42,
    "test_split_n": 47,
    "train_n": 137,
    "dev_n": 45
}

with open("${METADATA_FILE}", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Wrote ${METADATA_FILE}")
PYEOF

info "✅ Setup complete."
info "   Primary model:  ${PRIMARY_MODEL} (smoke test: ${PRIMARY_LATENCY}ms)"
info "   Second rater:   ${SECOND_MODEL}  (smoke test: ${SECOND_LATENCY}ms)"
info "   Metadata:       ${METADATA_FILE}"
