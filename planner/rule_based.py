"""
planner/rule_based.py

Rule-based fallback planner: maps keywords in the query to one or more tool calls.
This is the safety net — no LLM, no API key, no network required.

Returns a list of tool call specs:
  [{"tool": "search_news", "args": {"query": ..., "ticker": ...}}, ...]

Design:
  1. Extract tickers mentioned in the query (uppercase 2-5 letter words matching known tickers).
  2. Match keywords to tools using a priority-ordered rule table.
  3. Return de-duplicated list of tool calls.

Max tool calls is enforced by the orchestrator (not here), but this planner
will never emit more than 2 tool calls per query — covering the hardest
case (guidance + earnings comparison).
"""

import re
from typing import Optional

# Known tickers in our fixture data — for extraction heuristics
_KNOWN_TICKERS = {"NVDA", "TSLA", "JPM", "XOM"}

# Broad ticker pattern (2–5 uppercase letters): used as a fallback
_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Keyword → tool mapping (ordered: more-specific rules first)
# Each rule: (keywords_required_ANY, tool_name)
# NOTE: 'perform/performance/business/how did' are NEWS signals, not earnings
_RULES: list[tuple[list[str], str]] = [
    # guidance + earnings comparison (multi-tool) — check these FIRST
    # Phrases that imply both a forward-looking forecast AND an actual result
    (
        [
            "beat or miss", "actual earnings", "earnings beat", "earnings miss",
            "vs their", "versus their", "compare to guidance",
            "vs what they guided", "reporting vs", "vs guidance",
            "beat their guidance", "miss their guidance",
        ],
        "get_guidance",
    ),
    (
        [
            "beat or miss", "actual earnings", "earnings beat", "earnings miss",
            "vs their", "versus their", "compare to guidance",
            "vs what they guided", "reporting vs", "vs guidance",
            "beat their guidance", "miss their guidance",
        ],
        "get_earnings",
    ),
    # single-tool: earnings (hard numeric results)
    (
        [
            "quarterly results", "eps", "net income", "profit", "revenue beat",
            "revenue miss", "earnings results", "reported earnings", "earnings report",
            "beat estimate", "missed estimate", "quarterly earnings",
            "earn last quarter", "earned last quarter", "how much did", "how much earn",
            "earn this quarter", "earn per share",
        ],
        "get_earnings",
    ),
    # single-tool: guidance (forward-looking company forecasts)
    (
        [
            "guided", "guidance range", "raised guidance", "lowered guidance",
            "forecast revenue", "projected revenue", "company forecast",
        ],
        "get_guidance",
    ),
    # single-tool: analyst ratings
    (
        [
            "rating", "ratings", "analyst", "analysts", "sentiment", "price target",
            "upgrade", "downgrade", "overweight", "underweight", "outperform",
            "underperform",
        ],
        "get_ratings",
    ),
    # news is the broadest — also used as default/catch-all
    (
        [
            "news", "article", "report", "story", "recent", "latest",
            "opec", "meeting", "outlook", "trend", "said", "says", "announced",
            "perform", "performance", "business", "how did", "what happened",
        ],
        "search_news",
    ),
]

# Tools that are NOT query-keyword driven but ticker-only
_TICKER_ONLY_TOOLS: dict[str, str] = {
    # If we only see a ticker and no specific keywords, use news as default
}


# Words that look like tickers but are NOT tickers (stop-words for ticker extraction)
_TICKER_STOPWORDS = {
    "OPEC", "CEO", "CFO", "COO", "IPO", "ETF", "GDP", "CPI", "FED",
    "SEC", "NYSE", "NASDAQ", "US", "USA", "UK", "EU", "AI", "ML",
    "YOY", "QOQ", "EPS", "NII", "FY", "Q3", "Q4", "API",
}


def _extract_ticker(query: str) -> Optional[str]:
    """
    Extract the most likely ticker from the query.
    Priority:
      1. Company name → ticker mapping (most reliable)
      2. Known fixture tickers found verbatim in query
      3. Uppercase 2–5 letter words not in stopword list
    """
    # Priority 1: company name resolution
    company_ticker = _extract_ticker_from_company(query)
    if company_ticker:
        return company_ticker

    words = query.split()
    # Priority 2: known fixture tickers
    for word in words:
        cleaned = re.sub(r"[^A-Za-z]", "", word).upper()
        if cleaned in _KNOWN_TICKERS:
            return cleaned
    # Priority 3: first uppercase-only token of 2-5 chars not in stopwords
    for word in words:
        m = _TICKER_RE.match(word)
        if m:
            candidate = m.group(1)
            if candidate not in _TICKER_STOPWORDS and len(candidate) <= 5:
                return candidate
    return None


def _extract_ticker_from_company(query: str) -> Optional[str]:
    """Map well-known company names to tickers."""
    q_lower = query.lower()
    name_map = {
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "j.p. morgan": "JPM",
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "exxon mobil": "XOM",
    }
    for name, ticker in name_map.items():
        if name in q_lower:
            return ticker
    return None


def plan(query: str) -> list[dict]:
    """
    Given a user query, return a list of tool call specs.

    Each spec: {"tool": str, "args": dict}

    Rules:
    - At most 2 distinct tools per call (multi-tool queries like guidance+earnings)
    - Default to search_news if no keyword matches
    """
    q_lower = query.lower()

    # Extract ticker
    ticker = _extract_ticker(query)

    # Match rules
    matched_tools: list[str] = []
    for keywords, tool in _RULES:
        if any(kw in q_lower for kw in keywords):
            if tool not in matched_tools:
                matched_tools.append(tool)

    # Special case: if both guidance AND earnings keywords appear, include both
    has_guidance_kw = any(kw in q_lower for kw in ["guidance", "forecast", "projected", "estimate"])
    has_earnings_kw = any(kw in q_lower for kw in ["actual", "beat", "miss", "earnings", "revenue", "results"])
    if has_guidance_kw and has_earnings_kw:
        for t in ["get_guidance", "get_earnings"]:
            if t not in matched_tools:
                matched_tools.insert(0, t)

    # Default: if nothing matched, fall back to news search
    if not matched_tools:
        matched_tools = ["search_news"]

    # De-duplicate while preserving order, cap at 2
    seen: set[str] = set()
    deduped: list[str] = []
    for t in matched_tools:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    deduped = deduped[:2]

    # Build tool call specs
    calls = []
    for tool_name in deduped:
        if tool_name == "search_news":
            calls.append({
                "tool": "search_news",
                "args": {"query": query, "ticker": ticker},
            })
        else:
            # Ticker-required tools: skip if no ticker found
            if ticker:
                calls.append({"tool": tool_name, "args": {"ticker": ticker}})
            else:
                # No ticker: fall back to news search
                calls.append({
                    "tool": "search_news",
                    "args": {"query": query, "ticker": None},
                })

    return calls


if __name__ == "__main__":
    import json
    tests = [
        "How did Nvidia's data center business perform last quarter?",
        "What's the current analyst sentiment on Tesla?",
        "Did JPMorgan's actual earnings beat or miss their own guidance?",
        "What's the outlook for Exxon given the upcoming OPEC meeting?",
        "What is ZZZZ doing?",
    ]
    for q in tests:
        print(f"Q: {q}")
        print(f"Plan: {json.dumps(plan(q), indent=2)}")
        print()
