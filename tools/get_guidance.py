"""
tools/get_guidance.py

Returns the most recent company guidance figures for a given ticker.

Offline mode (default):  loads data/guidance_fixtures.json
Online mode:             calls the financial data API when FINANCIAL_DATA_API_KEY is set

Each guidance record has a 'source_ref': {ticker, channel, record_id}
"""

import os
import json
import time
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_FIXTURES_PATH = _DATA_DIR / "guidance_fixtures.json"


def _load_fixtures() -> list[dict]:
    with open(_FIXTURES_PATH, "r") as f:
        return json.load(f)


def _get_guidance_offline(ticker: str, limit: int) -> dict:
    all_guidance = _load_fixtures()
    ticker_upper = ticker.upper()
    records = [g for g in all_guidance if g["ticker"].upper() == ticker_upper]

    if not records:
        return {
            "ticker": ticker_upper,
            "guidance": [],
            "error": f"No guidance data found for {ticker_upper}",
        }

    # Sort by issued_date descending
    records_sorted = sorted(records, key=lambda r: r["issued_date"], reverse=True)[:limit]

    for r in records_sorted:
        r["source_ref"] = {
            "ticker": r["ticker"],
            "channel": "guidance",
            "record_id": r["id"],
        }

    return {
        "ticker": ticker_upper,
        "guidance": records_sorted,
    }


# Set GUIDANCE_API_URL (and FINANCIAL_DATA_API_KEY) to point at your data provider.
# Example: GUIDANCE_API_URL=https://api.yourprovider.com/v2.1/calendar/guidance
_GUIDANCE_API_URL = os.environ.get("GUIDANCE_API_URL", "")


def _get_guidance_online(ticker: str, limit: int) -> dict:
    """Call the financial data API for guidance. Falls back offline on error."""
    try:
        import requests  # noqa: PLC0415
        api_key = os.environ["FINANCIAL_DATA_API_KEY"]
        params = {
            "token": api_key,
            "parameters[tickers]": ticker.upper(),
            "parameters[pageSize]": limit,
        }
        resp = requests.get(
            _GUIDANCE_API_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        items = raw.get("guidance", raw) if isinstance(raw, dict) else raw

        records = []
        for item in items[:limit]:
            record = {
                "id": str(item.get("id", "")),
                "ticker": ticker.upper(),
                "period": item.get("period", ""),
                "issued_date": item.get("date", ""),
                "metric": item.get("name", "Revenue"),
                "guidance_value": item.get("eps_est") or item.get("revenue_est"),
                "guidance_unit": "USD",
                "guidance_low": item.get("eps_est_low") or item.get("revenue_low"),
                "guidance_high": item.get("eps_est_high") or item.get("revenue_high"),
                "analyst_estimate_prior": None,
                "notes": item.get("importance", ""),
                "source_ref": {
                    "ticker": ticker.upper(),
                    "channel": "guidance",
                    "record_id": str(item.get("id", "")),
                },
            }
            records.append(record)

        if not records:
            return _get_guidance_offline(ticker, limit)

        return {"ticker": ticker.upper(), "guidance": records}
    except Exception as exc:  # noqa: BLE001
        print(f"[get_guidance] Online fetch failed ({exc}), falling back to offline.")
        return _get_guidance_offline(ticker, limit)


def get_guidance(ticker: str, limit: int = 3) -> dict:
    """
    Get the most recent company guidance for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'JPM').
        limit:  Maximum number of guidance records to return (default 3).

    Returns:
        {
          "ticker":     str,
          "guidance":   list of guidance dicts (with source_ref),
          "mode":       "offline" | "online",
          "tool":       "get_guidance",
          "latency_ms": float,
        }
    """
    t0 = time.time()
    mode = "online" if os.environ.get("FINANCIAL_DATA_API_KEY") else "offline"

    if mode == "online":
        result = _get_guidance_online(ticker, limit)
    else:
        result = _get_guidance_offline(ticker, limit)

    result["mode"] = mode
    result["tool"] = "get_guidance"
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


if __name__ == "__main__":
    out = get_guidance("JPM")
    print(json.dumps(out, indent=2))
