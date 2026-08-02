"""
tools/get_earnings.py

Returns the most recent earnings report summary for a given ticker.

Offline mode (default):  loads data/earnings_fixtures.json
Online mode:             calls the financial data API when FINANCIAL_DATA_API_KEY is set

Each earnings record has a 'source_ref': {ticker, channel, record_id}
"""

import os
import json
import time
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_FIXTURES_PATH = _DATA_DIR / "earnings_fixtures.json"


def _load_fixtures() -> list[dict]:
    with open(_FIXTURES_PATH, "r") as f:
        return json.load(f)


def _get_earnings_offline(ticker: str, limit: int) -> dict:
    all_earnings = _load_fixtures()
    ticker_upper = ticker.upper()
    records = [e for e in all_earnings if e["ticker"].upper() == ticker_upper]

    if not records:
        return {
            "ticker": ticker_upper,
            "earnings": [],
            "error": f"No earnings data found for {ticker_upper}",
        }

    # Sort by report_date descending — most recent first
    records_sorted = sorted(records, key=lambda r: r["report_date"], reverse=True)[:limit]

    for r in records_sorted:
        r["source_ref"] = {
            "ticker": r["ticker"],
            "channel": "earnings",
            "record_id": r["id"],
        }

    return {
        "ticker": ticker_upper,
        "earnings": records_sorted,
    }


# Set EARNINGS_API_URL (and FINANCIAL_DATA_API_KEY) to point at your data provider.
# Example: EARNINGS_API_URL=https://api.yourprovider.com/v2.1/calendar/earnings
_EARNINGS_API_URL = os.environ.get("EARNINGS_API_URL", "")


def _get_earnings_online(ticker: str, limit: int) -> dict:
    """Call the financial data API for earnings. Falls back to offline on any error."""
    try:
        import requests  # noqa: PLC0415
        api_key = os.environ["FINANCIAL_DATA_API_KEY"]
        params = {
            "token": api_key,
            "parameters[tickers]": ticker.upper(),
            "parameters[pageSize]": limit,
        }
        resp = requests.get(
            _EARNINGS_API_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        items = raw.get("earnings", raw) if isinstance(raw, dict) else raw

        records = []
        for item in items[:limit]:
            eps_actual = item.get("eps")
            eps_est = item.get("eps_est")
            rev_actual = item.get("revenue")
            rev_est = item.get("revenue_est")

            eps_beat = None
            if eps_actual is not None and eps_est is not None:
                try:
                    eps_beat = "beat" if float(eps_actual) >= float(eps_est) else "miss"
                except (ValueError, TypeError):
                    pass

            rev_beat = None
            if rev_actual is not None and rev_est is not None:
                try:
                    rev_beat = "beat" if float(rev_actual) >= float(rev_est) else "miss"
                except (ValueError, TypeError):
                    pass

            record = {
                "id": str(item.get("id", "")),
                "ticker": ticker.upper(),
                "period": item.get("period", ""),
                "report_date": item.get("date", ""),
                "revenue_actual": rev_actual,
                "revenue_estimate": rev_est,
                "revenue_beat_miss": rev_beat,
                "revenue_surprise_pct": None,
                "eps_actual": eps_actual,
                "eps_estimate": eps_est,
                "eps_beat_miss": eps_beat,
                "eps_surprise_pct": None,
                "net_income_actual": None,
                "data_center_revenue": None,
                "yoy_revenue_growth_pct": None,
                "notes": item.get("importance", ""),
                "source_ref": {
                    "ticker": ticker.upper(),
                    "channel": "earnings",
                    "record_id": str(item.get("id", "")),
                },
            }
            records.append(record)

        if not records:
            return _get_earnings_offline(ticker, limit)

        return {"ticker": ticker.upper(), "earnings": records}
    except Exception as exc:  # noqa: BLE001
        print(f"[get_earnings] Online fetch failed ({exc}), falling back to offline.")
        return _get_earnings_offline(ticker, limit)


def get_earnings(ticker: str, limit: int = 2) -> dict:
    """
    Get the most recent earnings report(s) for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'NVDA').
        limit:  Maximum number of earnings records to return (default 2).

    Returns:
        {
          "ticker":     str,
          "earnings":   list of earnings dicts (with source_ref),
          "mode":       "offline" | "online",
          "tool":       "get_earnings",
          "latency_ms": float,
        }
    """
    t0 = time.time()
    mode = "online" if os.environ.get("FINANCIAL_DATA_API_KEY") else "offline"

    if mode == "online":
        result = _get_earnings_online(ticker, limit)
    else:
        result = _get_earnings_offline(ticker, limit)

    result["mode"] = mode
    result["tool"] = "get_earnings"
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


if __name__ == "__main__":
    out = get_earnings("JPM")
    print(json.dumps(out, indent=2))
