"""Polygon.io Options Advanced client for the options page.

Pure data layer — no Streamlit imports. Mirrors the equities-bot's REST
patterns (reference endpoint for chain discovery, defensive pagination,
`next_url` cursor needs the apiKey re-appended). Quotes/greeks come from the
snapshot endpoints, which carry bid/ask, implied_volatility, and
delta/gamma/theta/vega in one response.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.polygon.io"
PAGE_LIMIT = 250
MAX_PAGES = 10
TIMEOUT = 15


class PolygonError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise PolygonError("POLYGON_API_KEY not set (env / .env / Streamlit secrets)")
    return key


def _get(path_or_url: str, **params: Any) -> dict:
    url = path_or_url if path_or_url.startswith("http") else f"{BASE}{path_or_url}"
    params = {k: v for k, v in params.items() if v is not None}
    params["apiKey"] = _api_key()
    res = requests.get(url, params=params, timeout=TIMEOUT)
    if res.status_code != 200:
        raise PolygonError(f"HTTP {res.status_code} for {url.split('?')[0]}: {res.text[:200]}")
    data = res.json()
    if data.get("status") not in ("OK", "DELAYED", None):
        raise PolygonError(f"API status={data.get('status')} error={data.get('error')}")
    return data


def build_occ(ticker: str, expiry_iso: str, contract_type: str, strike: float) -> str:
    """OCC symbol with Polygon's `O:` prefix, e.g. O:AAPL260918C00230000."""
    yymmdd = expiry_iso.replace("-", "")[2:]
    cp = "C" if contract_type.lower().startswith("c") else "P"
    return f"O:{ticker.upper()}{yymmdd}{cp}{int(round(strike * 1000)):08d}"


def list_expirations(ticker: str) -> list[str]:
    """Distinct unexpired expiration dates for an underlying, ascending.

    Calls-only (put expiries are identical) at the endpoint's 1000/page max —
    date discovery needs far fewer requests than paging the full chain.
    """
    expiries: set[str] = set()
    url = "/v3/reference/options/contracts"
    params: dict[str, Any] = {
        "underlying_ticker": ticker.upper(),
        "expired": "false",
        "contract_type": "call",
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
    }
    for _ in range(MAX_PAGES):
        data = _get(url, **params)
        for c in data.get("results", []):
            d = c.get("expiration_date")
            if d and d >= date.today().isoformat():
                expiries.add(d)
        nxt = data.get("next_url")
        if not nxt:
            break
        url, params = nxt, {}
    return sorted(expiries)


def _row_from_snapshot(r: dict) -> dict:
    """Flatten one snapshot result into a display row."""
    details = r.get("details", {}) or {}
    quote = r.get("last_quote", {}) or {}
    greeks = r.get("greeks", {}) or {}
    day = r.get("day", {}) or {}
    return {
        "occ": details.get("ticker"),
        "expiry": details.get("expiration_date"),
        "type": details.get("contract_type"),
        "strike": details.get("strike_price"),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "ivm": r.get("implied_volatility"),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "vega": greeks.get("vega"),
        "theta": greeks.get("theta"),
        "open_interest": r.get("open_interest"),
        "volume": day.get("volume"),
        "underlying_price": (r.get("underlying_asset") or {}).get("price"),
    }


def chain_snapshot(ticker: str, expiry_iso: str, contract_type: str) -> list[dict]:
    """All contracts for (underlying, expiry, call|put) with quotes + greeks."""
    rows: list[dict] = []
    url = f"/v3/snapshot/options/{ticker.upper()}"
    params: dict[str, Any] = {
        "expiration_date": expiry_iso,
        "contract_type": contract_type.lower(),
        "limit": PAGE_LIMIT,
        "order": "asc",
        "sort": "strike_price",
    }
    for _ in range(MAX_PAGES):
        data = _get(url, **params)
        rows.extend(_row_from_snapshot(r) for r in data.get("results", []))
        nxt = data.get("next_url")
        if not nxt:
            break
        url, params = nxt, {}
    return rows


def contract_snapshot(ticker: str, occ: str) -> dict | None:
    """Quote + greeks for one contract; None if not found."""
    try:
        data = _get(f"/v3/snapshot/options/{ticker.upper()}/{occ}")
    except PolygonError as e:
        if "HTTP 404" in str(e):
            return None
        raise
    result = data.get("results")
    return _row_from_snapshot(result) if result else None
