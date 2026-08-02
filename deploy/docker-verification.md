# Docker Build Verification

**Date verified:** 2026-08-01  
**Platform:** macOS arm64 (Apple Silicon)  
**Docker runtime:** Colima 0.x (free, open-source Docker daemon for macOS) + Docker CLI 29.7.1  
**Image tag:** `financial-research-agent:latest`

> **Note (2026-08-02 security pass):** The Dockerfile was updated after this initial verification run to:
> (1) pin both `FROM` stages to the exact SHA256 digest of `python:3.12-slim` verified here, and
> (2) add a non-root `appuser` via `adduser` + `USER appuser`. The curl outputs below are from the first build and remain valid for the application behaviour; rebuild with the current Dockerfile to get the hardened image.

---


## Environment Setup

Docker Desktop's Homebrew cask requires `sudo` for a symlink step that cannot run
non-interactively. Instead, we used the Docker CLI formula (`brew install docker`) and
Colima as the daemon (`brew install colima && colima start`), which is the recommended
free alternative for macOS CI/non-interactive setups.

```bash
brew install docker docker-buildx docker-compose
brew install colima
colima start --cpu 2 --memory 4
# colima sets DOCKER_HOST automatically; can also be set explicitly:
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
```

---

## docker build

```
$ docker build -t financial-research-agent .

DEPRECATED: The legacy builder is deprecated and will be removed in a future release.
            Install the buildx component to build images with BuildKit:
            https://docs.docker.com/go/buildx/
Sending build context to Docker daemon  271.4MB
Step 1/23 : FROM python:3.12-slim AS builder
3.12-slim: Pulling from library/python
...
Status: Downloaded newer image for python:3.12-slim
 ---> 57cd7c3a7a27
Step 2/23 : WORKDIR /app
Step 3/23 : RUN pip install --no-cache-dir --upgrade pip
Successfully installed pip-26.2
Step 4/23 : COPY requirements.txt .
Step 5/23 : RUN pip install --no-cache-dir -r requirements.txt
Collecting flask>=3.0.0
Collecting openai>=1.0.0
Collecting anthropic>=0.30.0
Collecting requests>=2.31.0
...
Successfully installed annotated-types-0.8.0 anthropic-0.120.2 anyio-4.14.2
  blinker-1.9.0 certifi-2026.7.22 charset_normalizer-3.4.9 click-8.4.2
  distro-1.9.0 docstring-parser-0.18.0 flask-3.1.3 h11-0.16.0 httpcore-1.0.9
  httpx-0.28.1 idna-3.18 itsdangerous-2.2.0 jinja2-3.1.6 jiter-0.16.0
  markupsafe-3.0.3 openai-2.52.0 pydantic-2.13.4 pydantic-core-2.46.4
  requests-2.34.2 sniffio-1.3.1 tqdm-4.70.0 typing-extensions-4.16.0
  typing-inspection-0.4.2 urllib3-2.7.0 werkzeug-3.1.8
Step 6/23 : FROM python:3.12-slim
Step 7/23 : WORKDIR /app
Step 8/23 : COPY --from=builder /usr/local/lib/python3.12/site-packages ...
Step 9/23 : COPY --from=builder /usr/local/bin ...
Step 10/23 : COPY agent/       ./agent/
Step 11/23 : COPY classifier/  ./classifier/
Step 12/23 : COPY data/        ./data/
Step 13/23 : COPY logger/      ./logger/
Step 14/23 : COPY planner/     ./planner/
Step 15/23 : COPY static/      ./static/
Step 16/23 : COPY tools/       ./tools/
Step 17/23 : COPY app.py       .
Step 18/23 : RUN mkdir -p logs
Step 19/23 : ENV PORT=5001
Step 20/23 : ENV PYTHONPATH=/app
Step 21/23 : EXPOSE 5001
Step 22/23 : HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 ...
Step 23/23 : CMD ["sh", "-c", "python3 app.py"]
Successfully built 3d637ec7012e
Successfully tagged financial-research-agent:latest
```

**Result: ✅ Build succeeded. All 23 steps clean, zero errors.**

---

## docker run

```bash
$ docker run -d --name fra-test -p 5001:5001 -e PORT=5001 financial-research-agent
6d65092d4518b060c5cedf343f4b0d314b739df2c7b1e754e3bdfc172008cdd4
```

---

## curl /health

```bash
$ curl -s http://localhost:5001/health
```

```json
{
  "anthropic_api": false,
  "financial_data_api": false,
  "llm": "rule-based-fallback",
  "mode": "offline",
  "status": "ok"
}
```

**Result: ✅ Health endpoint responding correctly. Offline mode confirmed (no env vars set).**

---

## curl /query

```bash
$ curl -s -X POST http://localhost:5001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How did Nvidia'\''s data center business perform last quarter?"}'
```

```json
{
  "answer": "[Source: news-nvda-001] Nvidia Data Center Revenue Hits Record $22.6B in Q3 FY2025, Surpassing Analyst Expectations\nNvidia reported record data center revenue of $22.6 billion for Q3 FY2025, up 112% year-over-year. The result exceeded Wall Street consensus of $21.1 billion. CEO Jensen Huang cited explosive demand for H100 and H200 GPUs from hyperscalers and cloud providers. The company guided Q4 data center revenue above $37 billion.\n[Source: news-nvda-003] Additionally: Nvidia Q4 Guidance Tops Estimates; Analysts Raise Price Targets Across the Board",
  "mode": "fallback",
  "query": "How did Nvidia's data center business perform last quarter?",
  "sources": [
    {"channel": "news", "record_id": "news-nvda-001", "ticker": "NVDA"},
    {"channel": "news", "record_id": "news-nvda-003", "ticker": "NVDA"}
  ],
  "tool_calls_made": 1,
  "trace": [
    {"step": 0, "tool": "__planner__", "latency_ms": 0.1, "success": true},
    {"step": 1, "tool": "search_news",  "latency_ms": 2.9, "success": true}
  ]
}
```

**Result: ✅ Agent running correctly inside Docker. Grounded answer, correct sources, correct trace.**

---

## Summary

| Check | Result |
|---|---|
| `docker build` — 23 steps | ✅ Pass |
| `GET /health` | ✅ `{"status":"ok","mode":"offline"}` |
| `POST /query` (NVDA) | ✅ Grounded answer, 2 sources, 1 tool call |
| No env vars needed | ✅ Offline-first confirmed |

Build verified locally on 2026-08-01 using Docker CLI 29.7.1 + Colima (free macOS daemon).
