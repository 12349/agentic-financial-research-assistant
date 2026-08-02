"""
planner/llm_planner.py

LLM-based planner — provider-agnostic.

Provider selection via LLM_PROVIDER env var:
  - "groq"       → Groq free-tier API (OpenAI-compatible), model llama-3.3-70b-versatile
                    Requires: GROQ_API_KEY
  - "anthropic"  → Anthropic Claude, model claude-3-haiku-20240307
                    Requires: ANTHROPIC_API_KEY
  - "none" / unset → rule-based fallback (zero dependencies, always works)

Falls back to rule_based.plan() on any provider failure or missing key.

The LLM is asked to return a JSON list of tool call specs:
  [{"tool": "search_news", "args": {"query": ..., "ticker": ...}}, ...]

Available tools:
  - search_news(query, ticker=None)
  - get_ratings(ticker)
  - get_guidance(ticker)
  - get_earnings(ticker)
"""

import os
import json
import re
from typing import Optional
from .rule_based import plan as fallback_plan

# Tool definitions for the prompt
_TOOL_DESCRIPTIONS = """
You have access to 4 tools for financial research:

1. search_news(query: str, ticker: str | None)
   - Searches financial news articles using TF-IDF similarity
   - Use for: recent news, performance narratives, macro events, outlook
   - Example: search_news("NVDA data center performance", ticker="NVDA")

2. get_ratings(ticker: str)
   - Returns the most recent analyst ratings and price targets for a ticker
   - Use for: analyst sentiment, consensus, buy/sell/hold counts, price targets
   - Example: get_ratings("TSLA")

3. get_guidance(ticker: str)
   - Returns the most recent company-issued guidance (revenue, EPS forecasts)
   - Use for: what the company predicted for an upcoming period
   - Example: get_guidance("JPM")

4. get_earnings(ticker: str)
   - Returns the most recent earnings report: actual vs. estimated revenue and EPS
   - Use for: did results beat or miss, quarterly financials
   - Example: get_earnings("JPM")
"""

_SYSTEM_PROMPT = """You are a financial research assistant that decides which tools to call to answer a user query.
Given a user question, output a JSON array of tool calls. Each call is:
  {"tool": "<tool_name>", "args": {<args>}}

Rules:
- Use ONLY the 4 tools listed. Do not invent tools.
- Emit at most 2 tool calls per query.
- For compound questions needing both guidance AND earnings, include both.
- For news/narrative/outlook questions, use search_news.
- For analyst opinion questions, use get_ratings.
- Always extract the ticker from the query.
- Output ONLY the JSON array — no explanation, no markdown fences, just the raw JSON.
""".strip()


def _parse_llm_response(text: str) -> Optional[list[dict]]:
    """Extract a JSON array from the LLM's raw text response."""
    # Try direct parse first
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from within the text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _validate_plan(plan: list[dict]) -> list[dict]:
    """
    Validate and sanitize a plan from the LLM.
    - Only allow known tools.
    - Cap at 2 tool calls.
    """
    valid_tools = {"search_news", "get_ratings", "get_guidance", "get_earnings"}
    validated = []
    for call in plan:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool")
        args = call.get("args", {})
        if tool not in valid_tools:
            continue
        if not isinstance(args, dict):
            args = {}
        validated.append({"tool": tool, "args": args})
    return validated[:2]


def _build_user_message(query: str) -> str:
    return (
        f"User query: {query}\n\n"
        f"Available tools:\n{_TOOL_DESCRIPTIONS}\n\n"
        "Return the JSON array of tool calls to answer this query."
    )


def _plan_via_groq(query: str) -> Optional[list[dict]]:
    """Call Groq's OpenAI-compatible API (llama-3.3-70b-versatile) to plan tool calls."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # noqa: PLC0415
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query)},
            ],
            max_tokens=512,
            temperature=0,
        )
        raw_text = response.choices[0].message.content or ""
        return _parse_llm_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_planner] Groq call failed ({exc}), will fall back.")
        return None


def _plan_via_anthropic(query: str) -> Optional[list[dict]]:
    """Call Anthropic Claude (claude-3-haiku-20240307) to plan tool calls."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query)}],
        )
        raw_text = response.content[0].text if response.content else ""
        return _parse_llm_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_planner] Anthropic call failed ({exc}), will fall back.")
        return None


def plan(query: str) -> tuple[list[dict], str]:
    """
    Plan tool calls for a query using an LLM.

    Provider is selected via LLM_PROVIDER env var:
      "groq"      → Groq free-tier (llama-3.3-70b-versatile) — requires GROQ_API_KEY
      "anthropic" → Anthropic Claude (claude-3-haiku) — requires ANTHROPIC_API_KEY
      unset/"none"→ rule-based fallback (zero dependencies)

    Returns:
        (plan: list[dict], mode: str)
        where mode is "llm-groq", "llm-anthropic", or "fallback"
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect if LLM_PROVIDER not set but a key is present
    if not provider:
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"

    parsed = None

    if provider == "groq":
        parsed = _plan_via_groq(query)
        mode_label = "llm-groq"
    elif provider == "anthropic":
        parsed = _plan_via_anthropic(query)
        mode_label = "llm-anthropic"
    else:
        # Explicit "none" or no key available
        return fallback_plan(query), "fallback"

    if not parsed:
        print(f"[llm_planner] Could not parse LLM response from {provider}, falling back.")
        return fallback_plan(query), "fallback"

    validated = _validate_plan(parsed)
    if not validated:
        print(f"[llm_planner] {provider} plan was empty after validation, falling back.")
        return fallback_plan(query), "fallback"

    return validated, mode_label


if __name__ == "__main__":
    # Test with no API key (should use fallback)
    queries = [
        "How did Nvidia's data center business perform last quarter?",
        "Did JPMorgan's actual earnings beat or miss their own guidance?",
    ]
    for q in queries:
        result, mode = plan(q)
        print(f"Q: {q}")
        print(f"Mode: {mode}")
        print(f"Plan: {json.dumps(result, indent=2)}")
        print()
