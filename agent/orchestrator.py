"""
agent/orchestrator.py

The main agent loop.

Behavior (per §7):
  1. Receive user query.
  2. Planner decides which tools to call (LLM primary, rule-based fallback).
  3. Call each tool; log input/output/latency. Hard cap: MAX_TOOL_CALLS = 4.
  4. If a tool fails: log the error, record gap, continue with partial results.
  5. Pass all outputs to synthesizer.
  6. Return {answer, sources, trace, mode}.

This module enforces the max-call cap and never lets the loop run unbounded.
"""

import os
import time
from typing import Any, Optional

from tools import search_news, get_ratings, get_guidance, get_earnings
from planner.llm_planner import plan as llm_plan
from agent.synthesizer import synthesize
from logger.trace_logger import TraceLogger

# Hard cap on tool calls per query (§3.4)
MAX_TOOL_CALLS = 4

# Tool registry: maps tool name → callable
_TOOL_REGISTRY: dict[str, Any] = {
    "search_news": search_news,
    "get_ratings": get_ratings,
    "get_guidance": get_guidance,
    "get_earnings": get_earnings,
}

# Default log directory (can be overridden)
_LOG_DIR = os.environ.get("AGENT_LOG_DIR", "logs")


def run(query: str, log_dir: Optional[str] = None) -> dict:
    """
    Run the agent for a single query.

    Args:
        query:   The user's research question.
        log_dir: Directory to write trace logs to (uses AGENT_LOG_DIR env or 'logs/' default).

    Returns:
        {
          "answer":     str,           # the grounded answer
          "sources":    list[dict],    # [{ticker, channel, record_id}, ...]
          "trace":      list[dict],    # per-call trace entries
          "mode":       str,           # "llm" | "fallback" (planner mode)
          "query":      str,
          "tool_calls_made": int,
        }
    """
    effective_log_dir = log_dir or _LOG_DIR
    logger = TraceLogger(log_dir=effective_log_dir)
    logger.reset()

    # ------------------------------------------------------------------ #
    # Step 2/3: Plan — LLM primary, rule-based fallback
    # ------------------------------------------------------------------ #
    plan_t0 = time.time()
    try:
        tool_calls, planner_mode = llm_plan(query)
    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrator] Planning failed entirely ({exc}), defaulting to news search.")
        tool_calls = [{"tool": "search_news", "args": {"query": query, "ticker": None}}]
        planner_mode = "fallback"
    plan_latency = round((time.time() - plan_t0) * 1000, 1)

    # Log the planning step
    logger.log_call(
        step=0,
        tool="__planner__",
        input={"query": query},
        output={"plan": tool_calls, "planner_mode": planner_mode},
        latency_ms=plan_latency,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Execute tool calls (capped at MAX_TOOL_CALLS)
    # ------------------------------------------------------------------ #
    tool_outputs: list[dict] = []
    call_count = 0

    for call_spec in tool_calls:
        if call_count >= MAX_TOOL_CALLS:
            # Hard cap reached — log and stop
            logger.log_call(
                step=call_count + 1,
                tool="__cap__",
                input={"skipped": call_spec},
                output={"reason": f"MAX_TOOL_CALLS ({MAX_TOOL_CALLS}) reached"},
                latency_ms=0.0,
                error="cap_exceeded",
            )
            break

        tool_name = call_spec.get("tool", "")
        tool_args = call_spec.get("args", {})
        call_count += 1

        tool_fn = _TOOL_REGISTRY.get(tool_name)
        if not tool_fn:
            # Unknown tool — log as error, skip
            logger.log_call(
                step=call_count,
                tool=tool_name,
                input=tool_args,
                output=None,
                latency_ms=0.0,
                error=f"Unknown tool: {tool_name}",
            )
            tool_outputs.append({
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}",
            })
            continue

        # Call the tool
        t0 = time.time()
        error_msg: Optional[str] = None
        output: Optional[dict] = None

        try:
            output = tool_fn(**tool_args)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            output = {
                "tool": tool_name,
                "error": error_msg,
                "results": [],
                "ratings": [],
                "guidance": [],
                "earnings": [],
            }

        latency = round((time.time() - t0) * 1000, 1)

        # Attach tool name to output for synthesizer
        if output is not None and "tool" not in output:
            output["tool"] = tool_name

        logger.log_call(
            step=call_count,
            tool=tool_name,
            input=tool_args,
            output=output,
            latency_ms=latency,
            error=error_msg,
        )

        if output:
            tool_outputs.append(output)

    # ------------------------------------------------------------------ #
    # Step 5/6: Synthesize answer
    # ------------------------------------------------------------------ #
    synthesis = synthesize(query, tool_outputs)

    # ------------------------------------------------------------------ #
    # Step 7: Build response
    # ------------------------------------------------------------------ #
    trace = logger.get_trace()

    # Determine overall mode (planner + synthesizer)
    synthesis_mode = synthesis.get("mode", "fallback")
    if planner_mode == "llm" and synthesis_mode == "llm":
        overall_mode = "llm"
    elif planner_mode == "llm" or synthesis_mode == "llm":
        overall_mode = "hybrid"
    else:
        overall_mode = "fallback"

    result = {
        "query": query,
        "answer": synthesis["answer"],
        "sources": synthesis["sources"],
        "trace": trace,
        "mode": overall_mode,
        "tool_calls_made": call_count,
    }

    # Save trace to file
    logger.save(
        query=query,
        answer=synthesis["answer"],
        sources=synthesis["sources"],
        mode=overall_mode,
    )

    return result


if __name__ == "__main__":
    import json
    result = run("How did Nvidia's data center business perform last quarter?")
    print(json.dumps(result, indent=2))
