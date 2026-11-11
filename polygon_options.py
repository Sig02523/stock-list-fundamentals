"""Polygon.io Options Advanced client for the options page.

Pure data layer — no Streamlit imports. Mirrors the equities-bot's REST
patterns (reference endpoint for chain discovery, defensive pagination,
`next_url` cursor needs the apiKey re-appended). Quotes/greeks come from the
snapshot endpoints, which carry bid/ask, implied_volatility, and
delta/gamma/theta/vega in one response.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import blackscholes

load_dotenv()

logger = logging.getLogger("stocklist")

BASE = "https://api.polygon.io"
PAGE_LIMIT = 250
MAX_PAGES = 10
# (connect, read). A 15s connect budget stalled the 5s live tick for three
# ticks on every flaky connection — fail the connect fast, keep the read slack.
TIMEOUT = (3.05, 12)


class PolygonError(Exception):
    pass


class PolygonRateLimit(PolygonError):
    """429 — retryable."""


# Transport failures (read/connect timeouts, SSL EOF, reset-by-peer) are the
# common case on this endpoint; they must surface as PolygonError so callers
# can isolate one bad contract instead of losing the whole fetch.
_RETRYABLE = (
    PolygonRateLimit,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise PolygonError("POLYGON_API_KEY not set (env / .env / Streamlit secrets)")
    return key


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    # Fast and shallow: this runs inside a 5s live tick, so the whole retry
    # budget has to stay well under one refresh interval.
    wait=wait_exponential_jitter(initial=0.25, max=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _request(url: str, params: dict) -> requests.Response:
    res = requests.get(url, params=params, timeout=TIMEOUT)
    if res.status_code == 429:
        raise PolygonRateLimit(f"429 rate limited: {url.split('?')[0]}")
    return res


def _get(path_or_url: str, **params: Any) -> dict:
    url = path_or_url if path_or_url.startswith("http") else f"{BASE}{path_or_url}"
    params = {k: v for k, v in params.items() if v is not None}
    params["apiKey"] = _api_key()
    try:
        res = _request(url, params)
    except requests.RequestException as err:
        raise PolygonError(f"request failed for {url.split('?')[0]}: {err!r}") from err
    if res.status_code != 200:
        raise PolygonError(f"HTTP {res.status_code} for {url.split('?')[0]}: {res.text[:200]}")
    try:
        data = res.json()
    except ValueError as err:
        raise PolygonError(f"non-JSON response for {url.split('?')[0]}: {err!r}") from err
    if data.get("status") not in ("OK", "DELAYED", None):
        raise PolygonError(f"API status={data.get('status')} error={data.get('error')}")
    return data


def normalize_underlying(ticker: str) -> str:
    """User input -> Polygon symbology: BRK/B and BRK B -> BRK.B; strip $."""
    return ticker.strip().upper().replace("/", ".").replace(" ", ".").lstrip("$")


def build_occ(ticker: str, expiry_iso: str, contract_type: str, strike: float) -> str:
    """OCC symbol with Polygon's `O:` prefix, e.g. O:AAPL260918C00230000.

    OCC roots are dotless: underlying BRK.B -> root BRKB (the underlying
    symbol in URL paths keeps the dot; only the contract root drops it).
    """
    root = "".join(ch for ch in ticker.upper() if ch.isalnum())
    yymmdd = expiry_iso.replace("-", "")[2:]
    cp = "C" if contract_type.lower().startswith("c") else "P"
    return f"O:{root}{yymmdd}{cp}{int(round(strike * 1000)):08d}"


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


def stock_snapshots(tickers: list[str]) -> dict[str, dict]:
    """Spot + day change for underlyings (stocks snapshot endpoint)."""
    if not tickers:
        return {}
    data = _get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        tickers=",".join(sorted({t.upper() for t in tickers})),
    )
    out: dict[str, dict] = {}
    for t in data.get("tickers", []):
        last = (t.get("lastTrade") or {}).get("p") or (t.get("min") or {}).get("c")
        out[t["ticker"]] = {
            "price": last,
            "chg": t.get("todaysChange"),
            "chg_pct": t.get("todaysChangePerc"),
        }
    return out


def expiries_with_strike(ticker: str, strike: float, direction: str = "gte") -> list[str]:
    """Unexpired expiries that list strikes at/above (gte) or at/below (lte)
    the given level. One reference-endpoint page — used for messaging when a
    requested strike is outside the selected expiry's range."""
    data = _get(
        "/v3/reference/options/contracts",
        underlying_ticker=ticker.upper(),
        expired="false",
        contract_type="call",
        limit=1000,
        order="asc",
        sort="expiration_date",
        **{f"strike_price.{direction}": strike},
    )
    return sorted({c["expiration_date"] for c in data.get("results", []) if c.get("expiration_date")})


# Polygon reports implied_volatility ~0 (and occasionally absurd highs) when
# its solver can't price a contract off a stale or crossed quote — common on
# illiquid strikes. A sub-1% equity-option IV is never real, and the greeks
# come out of the same solve, so they are discarded together rather than
# shown as a plausible-looking 0.1% / near-zero delta.
IV_MIN = 0.01
IV_MAX = 10.0


def _usable_iv(value: Any) -> float | None:
    try:
        iv = float(value)
    except (TypeError, ValueError):
        return None
    return iv if IV_MIN <= iv <= IV_MAX else None


def _row_from_snapshot(r: dict, spot: float | None = None) -> dict:
    """Flatten one snapshot result into a display row.

    When Polygon's own IV is unusable, the greeks are re-solved locally off
    the mid so the row still carries a delta. `greeks_source` records which
    path produced the numbers.

    `spot` is the caller's own underlying price. It takes precedence over
    `underlying_asset.price`, which this plan does not reliably populate —
    the same reason the view sources spot from FMP.
    """
    details = r.get("details", {}) or {}
    quote = r.get("last_quote", {}) or {}
    greeks = r.get("greeks", {}) or {}
    day = r.get("day", {}) or {}
    strike = details.get("strike_price")
    expiry = details.get("expiration_date")
    is_call = str(details.get("contract_type", "")).lower().startswith("c")
    bid, ask = quote.get("bid"), quote.get("ask")
    underlying = (r.get("underlying_asset") or {}).get("price")
    solve_spot = spot if spot is not None else underlying

    iv = _usable_iv(r.get("implied_volatility"))
    source = "polygon" if iv is not None else None
    note = None
    if iv is None:
        # Polygon's solver gave up — recover IV and greeks from the quote.
        greeks = {}
        solved = blackscholes.solve(bid, ask, solve_spot, strike, expiry, is_call)
        if solved:
            iv = solved["ivm"]
            greeks = solved
            source = "computed"
        else:
            # Name the missing input — otherwise a contract with no greeks is
            # indistinguishable from one the solver merely declined. Carried
            # on the row as well as logged, so the UI can show the reason
            # instead of a bare dash.
            missing = [
                name for name, value in (
                    ("bid", bid), ("ask", ask), ("spot", solve_spot),
                    ("strike", strike), ("expiry", expiry),
                ) if value in (None, "")
            ]
            note = (
                f"polygon_iv={r.get('implied_volatility')!r} "
                f"missing={missing or 'nothing (solver declined)'} "
                f"bid={bid!r} ask={ask!r} spot={solve_spot!r}"
            )
            logger.warning("[%s] no usable greeks: %s", details.get("ticker"), note)

    return {
        "occ": details.get("ticker"),
        "expiry": expiry,
        "type": details.get("contract_type"),
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "ivm": iv,
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "vega": greeks.get("vega"),
        "theta": greeks.get("theta"),
        "greeks_source": source,
        "greeks_note": note,
        "open_interest": r.get("open_interest"),
        "volume": day.get("volume"),
        "underlying_price": underlying,
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
        try:
            data = _get(url, **params)
        except PolygonError as e:
            if "HTTP 404" in str(e):
                return []  # unknown underlying / no contracts — not an error
            raise
        rows.extend(_row_from_snapshot(r) for r in data.get("results", []))
        nxt = data.get("next_url")
        if not nxt:
            break
        url, params = nxt, {}
    return rows


def contract_snapshot(
    ticker: str, occ: str, spot: float | None = None
) -> dict | None:
    """Quote + greeks for one contract; None if not found.

    Pass `spot` when the caller has a better underlying price than this plan's
    `underlying_asset.price` — it is what the local greeks fallback solves on.
    """
    try:
        data = _get(f"/v3/snapshot/options/{ticker.upper()}/{occ}")
    except PolygonError as e:
        if "HTTP 404" in str(e):
            logger.warning("[%s] contract not found (404) for underlying %s", occ, ticker)
            return None
        raise
    result = data.get("results")
    if not result:
        logger.warning("[%s] snapshot returned no results", occ)
        return None
    return _row_from_snapshot(result, spot=spot)
