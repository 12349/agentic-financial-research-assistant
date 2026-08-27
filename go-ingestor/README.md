# go-ingestor — Standalone Ingestion Stub

> **Not wired to the Flask app in this build.** Read this before running it.

## What it does

`go-ingestor/main.go` demonstrates a streaming ingestion pattern for a financial-data pipeline:

- **Offline demo mode** (`-sample-file path/to/fixture.json`): reads a local JSON fixture, normalises each entry into a `NewsDoc` struct, and POSTs the batch to a configurable RAG service endpoint (`--rag-url`).
- **Live mode** (`FINANCIAL_NEWS_API_KEY` set): polls a REST financial news API on a configurable interval (`--interval`, default 30 s), normalises the response, and forwards to the same endpoint.

The struct and normalisation logic mirror what a production ingestion service would feed into a vector store.

## What it does NOT do

**The Flask app (`app.py`) has no `/ingest` endpoint.** The Go ingestor's default target (`http://localhost:8000/ingest`) does not exist in this project. Running it against the Flask app would return 404 on every push.

This is intentional: wiring a live ingestion pipeline to the FAISS index requires designing thread-safe index mutation, deduplication logic, and a persistence layer — non-trivial engineering decisions that deserve proper design, not a rushed stub. The ingestor is committed to show the ingestion *pattern*, not as a working end-to-end pipeline.

## Running it (against a real RAG service)

`
cd go-ingestor
go run main.go --rag-url http://your-rag-service/ingest --sample-file ../data/news_fixtures.json --once
`

`
# Live mode (requires FINANCIAL_NEWS_API_KEY and NEWS_API_URL)
FINANCIAL_NEWS_API_KEY=your_key go run main.go --rag-url http://your-rag-service/ingest --tickers NVDA,TSLA,JPM,XOM
`

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rag-url` | `http://localhost:8000/ingest` | Target ingest endpoint |
| `--interval` | `30s` | Polling interval (live mode) |
| `--sample-file` | `""` | Path to a local JSON fixture (offline mode) |
| `--tickers` | `""` | Comma-separated tickers to filter (live mode only) |
| `--once` | `false` | Run a single pass and exit |
