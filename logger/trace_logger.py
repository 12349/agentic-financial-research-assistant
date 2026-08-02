"""
logger/trace_logger.py

Structured JSON trace logger for the agent.

Each tool call is logged as a trace entry:
  {
    "step": int,
    "tool": str,
    "input": dict,
    "output": dict,          # tool return value (full)
    "latency_ms": float,
    "success": bool,
    "error": str | None,     # if tool raised an exception
    "timestamp_utc": str,
  }

The full trace is accumulated in memory and written to a JSON file at
the end of each query (if a log_dir is configured).

Usage:
    logger = TraceLogger(log_dir="logs/")
    logger.log_call(step=1, tool="search_news", input={...}, output={...}, latency_ms=12.3)
    trace = logger.get_trace()
    logger.save(query="...", answer="...")
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TraceLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self._trace: list[dict] = []
        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def reset(self):
        """Clear trace for a new query."""
        self._trace = []

    def log_call(
        self,
        step: int,
        tool: str,
        input: dict,
        output: Optional[dict],
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a single tool call in the trace."""
        entry = {
            "step": step,
            "tool": tool,
            "input": input,
            "output": output,
            "latency_ms": latency_ms,
            "success": error is None,
            "error": error,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._trace.append(entry)

    def get_trace(self) -> list[dict]:
        """Return the current trace entries."""
        return list(self._trace)

    def save(self, query: str, answer: str, sources: list[dict], mode: str) -> Optional[str]:
        """
        Write the full trace + answer to a JSON log file.

        Returns the path written, or None if no log_dir configured.
        """
        if not self._log_dir:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        filename = self._log_dir / f"trace_{ts}.json"
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "mode": mode,
            "answer": answer,
            "sources": sources,
            "trace": self._trace,
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        return str(filename)


if __name__ == "__main__":
    logger = TraceLogger(log_dir="/tmp/fin_agent_test_logs")
    logger.log_call(
        step=1,
        tool="search_news",
        input={"query": "test", "ticker": "NVDA"},
        output={"results": [], "mode": "offline"},
        latency_ms=5.2,
    )
    print(json.dumps(logger.get_trace(), indent=2))
    path = logger.save(
        query="test query",
        answer="test answer",
        sources=[],
        mode="fallback",
    )
    print(f"Saved to: {path}")
