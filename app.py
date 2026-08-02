"""
app.py — Flask API for the Multi-Agent Financial Research Assistant

Endpoints:
  POST /query           Run a research query through the agent
  GET  /health          Health check
  GET  /tools           List available tools and their descriptions

Usage:
  PYTHONPATH=. python3 app.py
  Then: curl -X POST http://localhost:5000/query -H 'Content-Type: application/json' \
        -d '{"query": "How did Nvidia perform last quarter?"}'
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from agent.orchestrator import run as agent_run

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.after_request
def add_cors(response):
    """Add CORS headers so the UI can call the API from any origin."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    """Serve the web UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:p>", methods=["OPTIONS"])
def options_handler(p=None):  # noqa: ARG001
    """Handle pre-flight CORS requests."""
    return "", 204


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    api_key_set = bool(os.environ.get("FINANCIAL_DATA_API_KEY"))
    llm_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return jsonify({
        "status": "ok",
        "mode": "online" if api_key_set else "offline",
        "llm": "anthropic" if llm_key_set else "rule-based-fallback",
        "financial_data_api": api_key_set,
        "anthropic_api": llm_key_set,
    })


@app.route("/tools", methods=["GET"])
def list_tools():
    """List the 4 available tool functions."""
    return jsonify({
        "tools": [
            {
                "name": "search_news",
                "description": "Semantic search over financial news articles (TF-IDF offline, live API online)",
                "args": {"query": "str", "ticker": "str | None"},
            },
            {
                "name": "get_ratings",
                "description": "Most recent analyst ratings and price targets for a ticker",
                "args": {"ticker": "str"},
            },
            {
                "name": "get_guidance",
                "description": "Most recent company guidance figures (revenue, EPS forecasts)",
                "args": {"ticker": "str"},
            },
            {
                "name": "get_earnings",
                "description": "Most recent earnings report: actual vs. estimated revenue and EPS",
                "args": {"ticker": "str"},
            },
        ],
        "max_tool_calls_per_query": 4,
    })


@app.route("/query", methods=["POST"])
def query():
    """
    Run a financial research query through the agent.

    Request body (JSON):
      {"query": "Your research question here"}

    Response (JSON):
      {
        "query":          str,
        "answer":         str,
        "sources":        [{ticker, channel, record_id}, ...],
        "trace":          [{step, tool, input, output, latency_ms, success, error, timestamp_utc}, ...],
        "mode":           "llm" | "fallback" | "hybrid",
        "tool_calls_made": int,
      }
    """
    body = request.get_json(silent=True)
    if not body or "query" not in body:
        return jsonify({
            "error": "Request body must be JSON with a 'query' field.",
            "example": {"query": "How did Nvidia's data center business perform last quarter?"}
        }), 400

    user_query = str(body["query"]).strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        result = agent_run(user_query)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        # §3.5: never crash on a bad response — return a partial answer
        return jsonify({
            "query": user_query,
            "answer": f"An unexpected error occurred while processing your query: {exc}",
            "sources": [],
            "trace": [],
            "mode": "error",
            "tool_calls_made": 0,
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting Financial Research Agent on port {port}")
    print(f"  FINANCIAL_DATA_API_KEY: {'set' if os.environ.get('FINANCIAL_DATA_API_KEY') else 'NOT SET (offline mode)'}")
    print(f"  ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET (rule-based fallback)'}")
    app.run(host="0.0.0.0", port=port, debug=debug)
