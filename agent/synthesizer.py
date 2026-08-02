"""
agent/synthesizer.py

Synthesizes a grounded answer from tool outputs.

Provider selection via LLM_PROVIDER env var:
  - "groq"       → Groq free-tier API (OpenAI-compatible), model llama-3.3-70b-versatile
                    Requires: GROQ_API_KEY
  - "anthropic"  → Anthropic Claude, model claude-3-5-sonnet-20241022
                    Requires: ANTHROPIC_API_KEY
  - unset/"none" → deterministic rule-based synthesis (no LLM required)

CRITICAL GROUNDING RULE (§3.1, §7.6):
  Every sentence in the answer must be traceable to a specific tool output.
  The fallback synthesizer enforces this by constructing the answer directly
  from tool output fields — it cannot hallucinate because it never generates
  free text; it only formats structured data.

Returns:
  {
    "answer":   str,            # the final text answer
    "sources":  list[dict],     # [{ticker, channel, record_id}, ...]
    "mode":     "llm-groq"|"llm-anthropic"|"fallback"
  }
"""

import os
import json
from typing import Optional


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------

def _extract_sources(tool_outputs: list[dict]) -> list[dict]:
    """
    Walk all tool outputs and collect every source_ref found.
    Deduplicates by record_id.
    """
    seen: set[str] = set()
    sources: list[dict] = []

    for output in tool_outputs:
        tool = output.get("tool", "")

        if tool == "search_news":
            for article in output.get("results", []):
                ref = article.get("source_ref")
                if ref and ref.get("record_id") not in seen:
                    seen.add(ref["record_id"])
                    sources.append(ref)

        elif tool == "get_ratings":
            for rating in output.get("ratings", []):
                ref = rating.get("source_ref")
                if ref and ref.get("record_id") not in seen:
                    seen.add(ref["record_id"])
                    sources.append(ref)

        elif tool == "get_guidance":
            for guidance in output.get("guidance", []):
                ref = guidance.get("source_ref")
                if ref and ref.get("record_id") not in seen:
                    seen.add(ref["record_id"])
                    sources.append(ref)

        elif tool == "get_earnings":
            for earnings in output.get("earnings", []):
                ref = earnings.get("source_ref")
                if ref and ref.get("record_id") not in seen:
                    seen.add(ref["record_id"])
                    sources.append(ref)

    return sources


def _format_dollars(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# Fallback (rule-based) synthesizer
# ---------------------------------------------------------------------------

def _synthesize_fallback(query: str, tool_outputs: list[dict]) -> dict:
    """
    Construct a grounded answer from structured tool output fields.
    Every sentence maps directly to a specific data field — no free generation.

    GROUNDING CONTRACT: sources[] contains ONLY records whose record_id appears
    inline in the answer text via [Source: <id>]. No record is added to sources
    unless it was actually cited.
    """
    paragraphs: list[str] = []
    gaps: list[str] = []
    used_sources: list[dict] = []    # only records actually cited in the answer
    seen_ids: set[str] = set()       # dedup guard

    def _add_source(source_ref: dict) -> None:
        rid = source_ref.get("record_id")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            used_sources.append(source_ref)

    for output in tool_outputs:
        tool = output.get("tool", "")
        error = output.get("error")

        # ---- search_news ----
        if tool == "search_news":
            results = output.get("results", [])
            if error or not results:
                gaps.append(f"No news data was available for this query (tool: search_news).")
                continue
            top = results[0]
            lines = [f"[Source: {top['source_ref']['record_id']}] {top['headline']}"]
            _add_source(top["source_ref"])
            # Add body snippet (first 300 chars) to ground the answer
            body_snip = top["body"][:400].rstrip()
            if not body_snip.endswith("."):
                body_snip = body_snip + "..."
            lines.append(body_snip)
            if len(results) > 1:
                second = results[1]
                lines.append(
                    f"[Source: {second['source_ref']['record_id']}] Additionally: {second['headline']}"
                )
                _add_source(second["source_ref"])
            paragraphs.append("\n".join(lines))

        # ---- get_ratings ----
        elif tool == "get_ratings":
            ratings = output.get("ratings", [])
            summary = output.get("summary")
            if error or not ratings or not summary:
                gaps.append(f"No analyst ratings data was available for {output.get('ticker', 'the requested ticker')}.")
                continue
            ticker = output.get("ticker", "")
            consensus = summary.get("consensus", "N/A")
            bullish = summary.get("bullish", 0)
            neutral = summary.get("neutral", 0)
            bearish = summary.get("bearish", 0)
            total = summary.get("total_ratings", 0)
            avg_pt = summary.get("avg_price_target")
            pt_str = f"${avg_pt:.0f}" if avg_pt else "N/A"

            lines = [
                f"Analyst sentiment on {ticker} is currently {consensus.upper()}.",
                f"[Source: {ratings[0]['source_ref']['record_id']}] Of {total} recent ratings: "
                f"{bullish} bullish, {neutral} neutral, {bearish} bearish. "
                f"Average price target: {pt_str}.",
            ]
            _add_source(ratings[0]["source_ref"])
            # Add top 2 individual ratings
            for r in ratings[:2]:
                action_str = f"{r.get('action', 'rates')} at" if r.get("action") else "rates"
                lines.append(
                    f"[Source: {r['source_ref']['record_id']}] "
                    f"{r['analyst_firm']} ({r.get('analyst_name', '')}) "
                    f"{action_str} {r['rating']}, PT ${r.get('price_target', 'N/A')} "
                    f"(from ${r.get('prior_price_target', 'N/A')}). "
                    f"{r.get('notes', '')}"
                )
                _add_source(r["source_ref"])
            paragraphs.append("\n".join(lines))

        # ---- get_guidance ----
        elif tool == "get_guidance":
            guidance_list = output.get("guidance", [])
            if error or not guidance_list:
                gaps.append(f"No guidance data was available for {output.get('ticker', 'the requested ticker')}.")
                continue
            g = guidance_list[0]
            ticker = output.get("ticker", "")
            val_str = _format_dollars(g.get("guidance_value"))
            est_str = _format_dollars(g.get("analyst_estimate_prior"))
            lines = [
                f"[Source: {g['source_ref']['record_id']}] "
                f"{ticker} issued {g.get('metric', 'guidance')} guidance of {val_str} "
                f"for {g.get('period', 'the period')} (issued {g.get('issued_date', 'N/A')}). "
                f"Prior analyst estimate was {est_str}.",
                f"  Notes: {g.get('notes', '')}",
            ]
            _add_source(g["source_ref"])
            paragraphs.append("\n".join(lines))

        # ---- get_earnings ----
        elif tool == "get_earnings":
            earnings_list = output.get("earnings", [])
            if error or not earnings_list:
                gaps.append(f"No earnings data was available for {output.get('ticker', 'the requested ticker')}.")
                continue
            e = earnings_list[0]
            ticker = output.get("ticker", "")
            rev_actual = _format_dollars(e.get("revenue_actual"))
            rev_est = _format_dollars(e.get("revenue_estimate"))
            rev_bm = e.get("revenue_beat_miss", "N/A")
            rev_surp = e.get("revenue_surprise_pct")
            rev_surp_str = f" ({rev_surp:+.1f}%)" if rev_surp is not None else ""
            eps_actual = e.get("eps_actual")
            eps_est = e.get("eps_estimate")
            eps_bm = e.get("eps_beat_miss", "N/A")
            eps_surp = e.get("eps_surprise_pct")
            eps_surp_str = f" ({eps_surp:+.1f}%)" if eps_surp is not None else ""
            yoy = e.get("yoy_revenue_growth_pct")
            yoy_str = f" Revenue grew {yoy:.1f}% year-over-year." if yoy is not None else ""
            lines = [
                f"[Source: {e['source_ref']['record_id']}] "
                f"{ticker} {e.get('period', '')} earnings (reported {e.get('report_date', 'N/A')}): "
                f"Revenue {rev_actual} vs. estimate {rev_est} — {rev_bm.upper()}{rev_surp_str}. "
                f"EPS ${eps_actual} vs. estimate ${eps_est} — {eps_bm.upper()}{eps_surp_str}.{yoy_str}",
                f"  Notes: {e.get('notes', '')}",
            ]
            _add_source(e["source_ref"])
            paragraphs.append("\n".join(lines))

    # Assemble final answer
    if not paragraphs and gaps:
        answer = "No data was found to answer this query.\n\n" + "\n".join(gaps)
    elif paragraphs:
        answer = "\n\n".join(paragraphs)
        if gaps:
            answer += "\n\n⚠ Data gaps: " + " ".join(gaps)
    else:
        answer = "No data was found to answer this query."

    return {"answer": answer, "sources": used_sources, "mode": "fallback"}


# ---------------------------------------------------------------------------
# LLM synthesizer — shared prompt + response handling
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """You are a financial research assistant. Your job is to write a grounded, cited answer.

CRITICAL RULES:
1. Every factual claim you make MUST be directly supported by data in the tool_outputs provided.
2. Cite each claim using [Source: <record_id>] inline.
3. Do NOT invent numbers, quotes, or facts not present in the tool_outputs.
4. If a tool returned no data or an error, state that explicitly in your answer.
5. Be concise. 3-5 sentences per tool output is sufficient.
6. End with a "Sources used:" list.
""".strip()


def _build_synthesis_message(query: str, tool_outputs: list[dict]) -> str:
    tool_data = json.dumps(tool_outputs, indent=2)
    return (
        f"User question: {query}\n\n"
        f"Tool outputs (use ONLY these to construct your answer):\n{tool_data}"
    )


def _synthesize_via_groq(query: str, tool_outputs: list[dict]) -> dict:
    """Use Groq's OpenAI-compatible API to generate a grounded answer."""
    try:
        from openai import OpenAI  # noqa: PLC0415
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": _build_synthesis_message(query, tool_outputs)},
            ],
            max_tokens=1024,
            temperature=0,
        )
        answer_text = response.choices[0].message.content or ""
        sources = _extract_sources(tool_outputs)
        return {"answer": answer_text, "sources": sources, "mode": "llm-groq"}
    except Exception as exc:  # noqa: BLE001
        print(f"[synthesizer] Groq synthesis failed ({exc}), falling back to rule-based.")
        result = _synthesize_fallback(query, tool_outputs)
        result["mode"] = "fallback"
        return result


def _synthesize_via_anthropic(query: str, tool_outputs: list[dict]) -> dict:
    """Use Anthropic Claude to generate a grounded answer."""
    try:
        import anthropic  # noqa: PLC0415
        api_key = os.environ["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=_SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": _build_synthesis_message(query, tool_outputs)}],
        )
        answer_text = response.content[0].text if response.content else ""
        sources = _extract_sources(tool_outputs)
        return {"answer": answer_text, "sources": sources, "mode": "llm-anthropic"}
    except Exception as exc:  # noqa: BLE001
        print(f"[synthesizer] Anthropic synthesis failed ({exc}), falling back to rule-based.")
        result = _synthesize_fallback(query, tool_outputs)
        result["mode"] = "fallback"
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(query: str, tool_outputs: list[dict]) -> dict:
    """
    Generate a grounded, sourced answer from tool outputs.

    Provider selected via LLM_PROVIDER env var:
      "groq"      → Groq free-tier (llama-3.3-70b-versatile) — requires GROQ_API_KEY
      "anthropic" → Anthropic Claude                          — requires ANTHROPIC_API_KEY
      unset/none  → rule-based fallback (zero dependencies)

    Returns:
        {"answer": str, "sources": list[dict], "mode": "llm-groq"|"llm-anthropic"|"fallback"}
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect if LLM_PROVIDER not set but a key is present
    if not provider:
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"

    if provider == "groq":
        return _synthesize_via_groq(query, tool_outputs)
    if provider == "anthropic":
        return _synthesize_via_anthropic(query, tool_outputs)
    return _synthesize_fallback(query, tool_outputs)
