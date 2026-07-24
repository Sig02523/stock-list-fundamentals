"""FMP API wrapper + coverage/unit diagnostics.

This module is the COVERAGE-FIRST deliverable. Before paying for an FMP tier,
run it against a FREE key to learn (a) which fields FMP actually fills and
(b) what UNITS those fields are in:

    python fmp_client.py --coverage          # which fields are populated, per endpoint
    python fmp_client.py --probe             # raw-vs-converted unit diagnostic
    python fmp_client.py --coverage --base https://financialmodelingprep.com/api/v3  # legacy

Default base is FMP's /stable API (v3 is deprecated). Stable takes the ticker as
a ?symbol= query param (v3 used a path segment) and returns JSON arrays. FMP
silently renames fields between tiers/versions — NOTHING here assumes a field
exists; every read goes through safe_get and is reported, not trusted.

No metrics math lives here (that is metrics.py). This file only fetches and
diagnoses.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Iterable

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

load_dotenv()

DEFAULT_BASE = "https://financialmodelingprep.com/stable"
DEFAULT_TIMEOUT = 15
DEFAULT_DELAY = 0.25  # polite gap between calls; free key is 250/day


# --------------------------------------------------------------------------- #
# Defensive helpers (used here and re-exported for metrics.py)
# --------------------------------------------------------------------------- #
def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Chained .get that never raises on missing keys / None / wrong type.

    safe_get(d, "a", "b") == d.get("a", {}).get("b") but tolerant of None and
    non-dict intermediates.
    """
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return default if cur is None else cur


def safe_div(num: Any, den: Any) -> float | None:
    """Division that returns None on 0, None, or non-numeric inputs."""
    try:
        if num is None or den is None:
            return None
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except (TypeError, ValueError):
        return None


def first(rows: Any) -> dict:
    """FMP returns a single-element list for many 'TTM'/profile endpoints.

    Normalize list|dict|None into a single dict (empty dict if absent)."""
    if isinstance(rows, list):
        return rows[0] if rows else {}
    if isinstance(rows, dict):
        return rows
    return {}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class FMPError(Exception):
    """Generic FMP failure."""


class FMPRateLimit(FMPError):
    """429 / quota exhausted — retryable."""


class FMPAccessError(FMPError):
    """403 / endpoint not on this plan — NOT retryable, report as 'no access'."""


_RETRYABLE = (FMPRateLimit, requests.exceptions.ConnectionError,
              requests.exceptions.Timeout)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class FMPClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE,
        timeout: int = DEFAULT_TIMEOUT,
        delay: float = DEFAULT_DELAY,
    ) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise FMPError(
                "FMP_API_KEY not set. Add it to .env (see .env.example) "
                "or export it in your shell."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.call_count = 0

    # -- low-level ---------------------------------------------------------- #
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self, url: str, params: dict) -> Any:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise FMPRateLimit(f"429 rate limited: {url}")
        if resp.status_code in (401, 402, 403):
            # 402 Payment Required = endpoint not on this plan (FMP uses it for
            # tier gates); 401/403 = auth/forbidden. None are retryable.
            raise FMPAccessError(f"{resp.status_code} no access: {url}")
        resp.raise_for_status()
        data = resp.json()
        # FMP encodes some errors as a 200 + {"Error Message": "..."}.
        if isinstance(data, dict) and "Error Message" in data:
            msg = data["Error Message"]
            low = msg.lower()
            if "limit" in low or "bandwidth" in low:
                raise FMPRateLimit(msg)
            if "not available" in low or "subscription" in low or "plan" in low:
                raise FMPAccessError(msg)
            raise FMPError(msg)
        return data

    def get(self, path: str, **params: Any) -> Any:
        """GET {base}/{path} with api key + polite delay + retry/backoff."""
        time.sleep(self.delay)
        self.call_count += 1
        params = {k: v for k, v in params.items() if v is not None}
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        return self._request(url, params)

    # -- endpoint wrappers (STABLE routes; symbol is a QUERY param) --------- #
    # Confirmed against FMP /stable docs 2026-07. Every endpoint returns a JSON
    # ARRAY (even single-symbol quote/profile); first() picks row 0 for the
    # dict-shaped ones. Doc-page slugs differ from real paths (e.g.
    # metrics-ratios-ttm -> ratios-ttm, cashflow-statement -> cash-flow-statement,
    # financial-estimates -> analyst-estimates); the paths below are the real ones.
    def quote(self, symbol: str) -> dict:
        # stable quote has NO sharesOutstanding/avgVolume — derive shares from
        # marketCap/price; compute $-volume ourselves from history.
        return first(self.get("quote", symbol=symbol))

    def profile(self, symbol: str) -> dict:
        return first(self.get("profile", symbol=symbol))

    def ratios_ttm(self, symbol: str) -> dict:
        return first(self.get("ratios-ttm", symbol=symbol))

    def key_metrics_ttm(self, symbol: str) -> dict:
        return first(self.get("key-metrics-ttm", symbol=symbol))

    def income_statement(self, symbol: str, period: str = "annual",
                         limit: int = 2) -> list:
        out = self.get("income-statement", symbol=symbol, period=period,
                       limit=limit)
        return out if isinstance(out, list) else []

    def balance_sheet(self, symbol: str, period: str = "annual",
                      limit: int = 1) -> list:
        out = self.get("balance-sheet-statement", symbol=symbol, period=period,
                       limit=limit)
        return out if isinstance(out, list) else []

    def cash_flow(self, symbol: str, period: str = "annual",
                  limit: int = 2) -> list:
        out = self.get("cash-flow-statement", symbol=symbol, period=period,
                       limit=limit)
        return out if isinstance(out, list) else []

    def analyst_estimates(self, symbol: str, period: str = "annual",
                          limit: int = 4) -> list:
        # stable requires `period`; returns array of forward estimates.
        out = self.get("analyst-estimates", symbol=symbol, period=period,
                       limit=limit)
        return out if isinstance(out, list) else []

    def historical_price(self, symbol: str, adjusted: bool = True,
                         from_: str | None = None, to: str | None = None) -> list:
        """Daily EOD series as a JSON ARRAY (no v3 'historical' wrapper).

        adjusted=True  -> historical-price-eod/dividend-adjusted (split+dividend
                          adjusted; field `adjClose`) — use for individual stocks.
        adjusted=False -> historical-price-eod/full (split-adjusted OHLC; field
                          `close`) — use for price indexes ^GSPC/^NDX, which have
                          no meaningful dividend adjustment.
        """
        path = ("historical-price-eod/dividend-adjusted" if adjusted
                else "historical-price-eod/full")
        out = self.get(path, symbol=symbol, **{"from": from_, "to": to})
        return out if isinstance(out, list) else []

    def dividends(self, symbol: str, limit: int = 100) -> list:
        """Dividend history as ARRAY; per-share adjusted field is `adjDividend`."""
        out = self.get("dividends", symbol=symbol, limit=limit)
        return out if isinstance(out, list) else []

    def earnings(self, symbol: str, limit: int = 12) -> list:
        """Per-symbol earnings (past + upcoming) as ARRAY; use `date` for next."""
        out = self.get("earnings", symbol=symbol, limit=limit)
        return out if isinstance(out, list) else []


# --------------------------------------------------------------------------- #
# Coverage map: (our_label, endpoint_method, field_name)
# These are the FMP v3 field names we EXPECT for each target column. The probe
# reports which are actually filled — do not assume any of them exist.
# --------------------------------------------------------------------------- #
COVERAGE_MAP: list[tuple[str, str, str]] = [
    # quote  (stable quote dropped sharesOutstanding & avgVolume vs v3 —
    # shares are derived from marketCap/price; $-volume computed from history)
    ("last_price",              "quote", "price"),
    ("52w_high",                "quote", "yearHigh"),
    ("52w_low",                 "quote", "yearLow"),
    ("market_cap",              "quote", "marketCap"),
    # profile
    ("sector",                  "profile", "sector"),
    ("industry",                "profile", "industry"),
    ("last_div(profile)",       "profile", "lastDividend"),
    ("avg_volume(profile)",     "profile", "averageVolume"),
    # ratios-ttm  (stable renamed several fields vs v3 — confirmed 2026-07)
    ("pe_trailing",             "ratios_ttm", "priceToEarningsRatioTTM"),
    ("ps",                      "ratios_ttm", "priceToSalesRatioTTM"),
    ("pb",                      "ratios_ttm", "priceToBookRatioTTM"),
    ("gross_margin",            "ratios_ttm", "grossProfitMarginTTM"),
    ("operating_margin",        "ratios_ttm", "operatingProfitMarginTTM"),
    ("current_ratio",           "ratios_ttm", "currentRatioTTM"),
    ("quick_ratio",             "ratios_ttm", "quickRatioTTM"),
    ("debt_equity",             "ratios_ttm", "debtToEquityRatioTTM"),
    ("debt_to_assets",          "ratios_ttm", "debtToAssetsRatioTTM"),
    ("dividend_yield(ratios)",  "ratios_ttm", "dividendYieldTTM"),
    ("div_per_share(ratios)",   "ratios_ttm", "dividendPerShareTTM"),
    ("peg_trailing",            "ratios_ttm", "priceToEarningsGrowthRatioTTM"),
    ("peg_forward",             "ratios_ttm", "forwardPriceToEarningsGrowthRatioTTM"),
    ("interest_coverage",       "ratios_ttm", "interestCoverageRatioTTM"),
    ("ev_multiple(ratios)",     "ratios_ttm", "enterpriseValueMultipleTTM"),
    # key-metrics-ttm  (ROA/ROE live HERE in stable, not in ratios-ttm)
    ("roa",                     "key_metrics_ttm", "returnOnAssetsTTM"),
    ("roe",                     "key_metrics_ttm", "returnOnEquityTTM"),
    ("ev_ebitda(km)",           "key_metrics_ttm", "evToEBITDATTM"),
    ("ev_sales(km)",            "key_metrics_ttm", "evToSalesTTM"),
    ("fcf_yield(km)",           "key_metrics_ttm", "freeCashFlowYieldTTM"),
    ("net_debt_ebitda(km)",     "key_metrics_ttm", "netDebtToEBITDATTM"),
    # income statement (latest annual)
    ("revenue",                 "income_statement", "revenue"),
    ("operating_income(EBIT)",  "income_statement", "operatingIncome"),
    ("ebitda",                  "income_statement", "ebitda"),
    ("interest_expense",        "income_statement", "interestExpense"),
    ("net_income",              "income_statement", "netIncome"),
    # balance sheet (latest annual)
    ("total_debt",              "balance_sheet", "totalDebt"),
    ("cash_and_equiv",          "balance_sheet", "cashAndCashEquivalents"),
    ("net_debt(bs)",            "balance_sheet", "netDebt"),
    ("total_assets",            "balance_sheet", "totalAssets"),
    ("total_equity",            "balance_sheet", "totalStockholdersEquity"),
    # cash flow (latest annual)
    ("free_cash_flow",          "cash_flow", "freeCashFlow"),
    ("dividends_paid",          "cash_flow", "commonDividendsPaid"),
    # analyst estimates (forward) — stable uses revenueAvg/epsAvg (not estimated*)
    ("est_revenue_avg",         "analyst_estimates", "revenueAvg"),
    ("est_eps_avg",             "analyst_estimates", "epsAvg"),
]

# Endpoints whose return is a list-of-periods; coverage checks the latest row.
_LIST_ENDPOINTS = {"income_statement", "balance_sheet", "cash_flow",
                   "analyst_estimates"}


def _load_tickers(path: str = "tickers.csv", limit: int | None = None) -> list[str]:
    import csv
    syms: list[str] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = (row.get("ticker") or "").strip()
            if t:
                syms.append(t)
    return syms[:limit] if limit else syms


def _fetch_bundle(client: FMPClient, symbol: str) -> tuple[dict, dict]:
    """Fetch one record per endpoint for `symbol`.

    Returns (bundle, errors) where bundle[endpoint] is the latest dict (or {})
    and errors[endpoint] is a short string if that endpoint failed.
    """
    bundle: dict[str, Any] = {}
    errors: dict[str, str] = {}
    methods = {
        "quote": lambda: client.quote(symbol),
        "profile": lambda: client.profile(symbol),
        "ratios_ttm": lambda: client.ratios_ttm(symbol),
        "key_metrics_ttm": lambda: client.key_metrics_ttm(symbol),
        "income_statement": lambda: client.income_statement(symbol),
        "balance_sheet": lambda: client.balance_sheet(symbol),
        "cash_flow": lambda: client.cash_flow(symbol),
        "analyst_estimates": lambda: client.analyst_estimates(symbol),
    }
    for name, fn in methods.items():
        try:
            data = fn()
            if name in _LIST_ENDPOINTS:
                bundle[name] = data[0] if data else {}
            else:
                bundle[name] = data or {}
        except FMPAccessError as e:
            errors[name] = f"NO ACCESS ({e})"
            bundle[name] = {}
        except FMPError as e:
            errors[name] = f"ERROR ({e})"
            bundle[name] = {}
    return bundle, errors


# --------------------------------------------------------------------------- #
# --coverage
# --------------------------------------------------------------------------- #
def run_coverage(client: FMPClient, tickers: list[str]) -> None:
    print(f"\n=== FMP COVERAGE PROBE ===  base={client.base_url}")
    print(f"tickers: {', '.join(tickers)}\n")

    filled: dict[tuple[str, str, str], int] = {row: 0 for row in COVERAGE_MAP}
    sample: dict[tuple[str, str, str], Any] = {}
    endpoint_errors: dict[str, set[str]] = {}

    for sym in tickers:
        bundle, errors = _fetch_bundle(client, sym)
        for ep, msg in errors.items():
            endpoint_errors.setdefault(ep, set()).add(f"{sym}: {msg}")
        for row in COVERAGE_MAP:
            label, ep, field = row
            val = safe_get(bundle.get(ep, {}), field)
            if val is not None and val != "":
                filled[row] += 1
                sample.setdefault(row, val)

    n = len(tickers)
    cur_ep = None
    for row in COVERAGE_MAP:
        label, ep, field = row
        if ep != cur_ep:
            cur_ep = ep
            print(f"\n--- {ep} ---")
        cnt = filled[row]
        bar = "OK " if cnt == n else ("PART" if cnt else "MISS")
        ex = sample.get(row)
        ex_s = f"  e.g. {ex!r}" if ex is not None else ""
        print(f"  [{bar}] {cnt}/{n}  {label:<24} <- {field}{ex_s}")

    # zero-fill fields = renamed or not on this plan
    missing = [f"{label} ({ep}.{field})"
               for (label, ep, field), c in filled.items() if c == 0]
    if missing:
        print("\n!! ZERO-FILL (renamed field or not on this plan — investigate):")
        for m in missing:
            print(f"   - {m}")

    if endpoint_errors:
        print("\n!! ENDPOINT ERRORS:")
        for ep, msgs in endpoint_errors.items():
            print(f"   {ep}:")
            for m in sorted(msgs):
                print(f"      {m}")

    # Heavy series endpoints — checked on the FIRST ticker only (1 call each) so
    # you learn whether history/dividends/earnings are on your plan without
    # burning budget across every ticker.
    probe_sym = tickers[0]
    print(f"\n--- series endpoints (reachability, {probe_sym} / ^GSPC only) ---")
    checks = [
        ("historical adj close", lambda: client.historical_price(probe_sym, adjusted=True), "adjClose"),
        ("index history ^GSPC",  lambda: client.historical_price("^GSPC", adjusted=False), "close"),
        ("dividends",            lambda: client.dividends(probe_sym), "adjDividend"),
        ("earnings",             lambda: client.earnings(probe_sym), "date"),
    ]
    for label, fn, field in checks:
        try:
            rows = fn()
            n_rows = len(rows) if isinstance(rows, list) else 0
            has = n_rows > 0 and field in (rows[0] or {})
            tag = "OK  " if has else ("EMPTY" if n_rows == 0 else "PART")
            print(f"  [{tag}] {label:<22} rows={n_rows:<5} field '{field}' present={has}")
        except FMPAccessError as e:
            print(f"  [NOACC] {label:<22} {e}")
        except FMPError as e:
            print(f"  [ERR ] {label:<22} {e}")

    print(f"\nAPI calls used this run: {client.call_count}")
    print("Next: review zero-fills/units, then run --probe.\n")


# --------------------------------------------------------------------------- #
# --probe  (raw vs OUR converted, for the unit-sensitive fields)
# --------------------------------------------------------------------------- #
def _trailing_annual_dividend(client: FMPClient, symbol: str,
                              ratios: dict | None = None) -> float | None:
    """Trailing ~12-month dividend PER SHARE.

    We COMPUTE dividend yield ourselves (annual div/share ÷ last price) rather
    than trusting FMP's yield field. Preferred source is the `dividends`
    endpoint, but that is 402-gated on the FREE plan — so we fall back to
    ratios-ttm.dividendPerShareTTM, which IS the trailing-12M dividend/share
    (still self-computed, NOT the vendor YIELD field). A genuine 0.0
    (paused/no dividend) is a real value, not missing.
    """
    try:
        hist = client.dividends(symbol)
        total, seen = 0.0, False
        # rows are date-descending; 4 most recent cash dividends ~= trailing yr
        # (probe only; metrics.py will window precisely by date).
        for rowi in (hist or [])[:4]:
            div = safe_get(rowi, "adjDividend")
            if div is None:
                div = safe_get(rowi, "dividend")
            if div is not None:
                total += float(div)
                seen = True
        if seen:
            return total
        if isinstance(hist, list):
            return 0.0  # endpoint worked but no dividends => real non-payer
    except FMPAccessError:
        pass  # gated on this plan — fall through to ratios-ttm
    except FMPError:
        return None
    r = ratios if ratios is not None else client.ratios_ttm(symbol)
    dps = safe_get(r, "dividendPerShareTTM")
    return float(dps) if dps is not None else 0.0


def _flag(name: str, raw: Any, converted: Any) -> str:
    """Heuristic 100x / sign sanity flags for the probe."""
    try:
        r = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        r = None
    if name in ("gross_margin", "operating_margin", "roa", "roe") and r is not None:
        if abs(r) > 1.5:
            return "SUSPECT: looks like a percent (>1.5); expected decimal"
    if name == "debt_equity" and r is not None and r > 12:
        return "SUSPECT: >12, possible x100 — but verify, some firms are levered"
    if name == "interest_coverage" and r is not None and r < 0:
        return "NOTE: negative — check interest-expense sign (use EBIT/abs(int))"
    return ""


def run_probe(client: FMPClient, tickers: list[str]) -> None:
    print(f"\n=== FMP UNIT PROBE (raw vs converted) ===  base={client.base_url}")
    hdr = f"{'field':<22}{'RAW':>16}{'CONVERTED':>16}  {'endpoint':<18}flag"
    for sym in tickers:
        print(f"\n##### {sym} #####")
        print(hdr)
        print("-" * len(hdr))
        bundle, _ = _fetch_bundle(client, sym)
        ratios = bundle.get("ratios_ttm", {})
        km = bundle.get("key_metrics_ttm", {})
        inc = bundle.get("income_statement", {})
        quote = bundle.get("quote", {})

        def line(name, raw, conv, ep):
            flg = _flag(name, raw, conv)
            rs = f"{raw}" if raw is not None else "—"
            cs = f"{conv}" if conv is not None else "—"
            print(f"{name:<22}{rs:>16}{cs:>16}  {ep:<18}{flg}")

        # debt/equity — FMP gives a TRUE ratio; do NOT divide by 100
        de = safe_get(ratios, "debtToEquityRatioTTM")
        line("debt_equity", de, de, "ratios-ttm")

        # margins (ratios-ttm) + ROA/ROE (key-metrics-ttm) — all decimals
        for nm, fld in [("gross_margin", "grossProfitMarginTTM"),
                        ("operating_margin", "operatingProfitMarginTTM")]:
            v = safe_get(ratios, fld)
            line(nm, v, v, "ratios-ttm")
        for nm, fld in [("roa", "returnOnAssetsTTM"),
                        ("roe", "returnOnEquityTTM")]:
            v = safe_get(km, fld)
            line(nm, v, v, "key-metrics-ttm")

        # dividend yield — COMPUTE ours (annual div/share ÷ price); show FMP's
        # dividendYieldTTM beside it. dividends endpoint is gated on free, so
        # ann_div/share falls back to ratios.dividendPerShareTTM.
        raw_dy = safe_get(ratios, "dividendYieldTTM")
        ann_div = _trailing_annual_dividend(client, sym, ratios)
        last_px = safe_get(quote, "price")
        our_dy = safe_div(ann_div, last_px)
        line("dividend_yield", raw_dy, our_dy, "computed")
        print(f"{'  ^ ann_div/share=' + str(ann_div):<38}last_price={last_px}")

        # EV/EBITDA & EV/Sales — precomputed in key-metrics-ttm
        line("ev_ebitda", safe_get(km, "evToEBITDATTM"),
             safe_get(km, "evToEBITDATTM"), "key-metrics-ttm")
        line("ev_sales", safe_get(km, "evToSalesTTM"),
             safe_get(km, "evToSalesTTM"), "key-metrics-ttm")

        # interest coverage — raw field vs our EBIT/abs(interest)
        raw_ic = safe_get(ratios, "interestCoverageRatioTTM")
        ebit = safe_get(inc, "operatingIncome")
        int_exp = safe_get(inc, "interestExpense")
        our_ic = safe_div(ebit, abs(int_exp)) if int_exp not in (None, 0) else None
        line("interest_coverage", raw_ic, our_ic, "computed")
        print(f"{'  ^ EBIT=' + str(ebit):<38}interest_expense={int_exp}")

        # PEG — trailing and forward
        line("peg_trailing", safe_get(ratios, "priceToEarningsGrowthRatioTTM"),
             safe_get(ratios, "priceToEarningsGrowthRatioTTM"), "ratios-ttm")
        line("peg_forward", safe_get(ratios, "forwardPriceToEarningsGrowthRatioTTM"),
             safe_get(ratios, "forwardPriceToEarningsGrowthRatioTTM"), "ratios-ttm")

    print(f"\nAPI calls used this run: {client.call_count}")
    print("Document confirmed units in metrics.py 'UNIT NOTES' before step 3.\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FMP client + coverage/unit probes")
    p.add_argument("--coverage", action="store_true",
                   help="report which fields FMP fills, per endpoint")
    p.add_argument("--probe", action="store_true",
                   help="raw-vs-converted unit diagnostic")
    p.add_argument("--base", default=DEFAULT_BASE,
                   help=f"API base URL (default {DEFAULT_BASE}; "
                        f"pass .../api/v3 for legacy)")
    p.add_argument("--tickers", default="tickers.csv",
                   help="path to tickers.csv")
    p.add_argument("--limit", type=int, default=None,
                   help="only probe the first N tickers (save API calls)")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help="seconds between calls")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not (args.coverage or args.probe):
        p.error("pass --coverage and/or --probe")

    try:
        client = FMPClient(base_url=args.base, delay=args.delay)
    except FMPError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    tickers = _load_tickers(args.tickers, args.limit)
    if not tickers:
        print("FATAL: no tickers found", file=sys.stderr)
        return 2

    if args.coverage:
        run_coverage(client, tickers)
    if args.probe:
        run_probe(client, tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
