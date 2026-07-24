"""Pure data layer for stock fundamentals / valuation reports.

Public surface: ``build_report(tickers_df)`` returns one row per input ticker
with a fixed schema (see ``COLUMNS``). No Streamlit imports — safe to call
from a notebook, CLI script, or GitHub Action.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from fmp_client import FMPAccessError, FMPClient, FMPError, first, safe_get


BENCHMARK_SYMBOLS = {"SPX": "^GSPC", "NDX": "^NDX"}

logger = logging.getLogger("stocklist")
# Symbol currently being fetched, so _retry failures can name it. Fetches are
# sequential, so a plain module dict is enough.
_log_ctx: dict[str, str] = {}

# Output column order. Kept here so app.py + Excel writer share one source of truth.
COLUMNS: list[str] = [
    "ticker", "benchmark", "note",
    "sector", "industry",
    "last_price", "ytd_pct", "pct_from_52w_high",
    "fifty_two_week_high", "fifty_two_week_low",
    "market_cap",
    "pe_trailing", "pe_forward", "ps", "pb", "peg",
    "ev_ebitda", "ev_sales", "fcf_yield",
    "rev_growth_ttm", "fwd_eps_growth", "fwd_rev_growth",
    "roa", "roe",
    "gross_margin", "operating_margin",
    "free_cash_flow",
    "current_ratio", "quick_ratio",
    "debt_equity", "debt_assets", "net_debt_ebitda", "interest_coverage",
    "dividend_yield",
    "beta", "vol_2m_annualized",
    "avg_dollar_vol_30d",
    "next_earnings",
    "missing_fields",
    "as_of",
]


# ---------------------------------------------------------------------------
# Defensive helpers
# ---------------------------------------------------------------------------

def _retry(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 0.6) -> Any:
    """Call ``fn`` with exponential backoff. Returns None on persistent failure."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network / parsing / Yahoo flakiness
            last_err = e
            time.sleep(base_delay * (2 ** i))
    # Caller treats None as missing data; the log line is the only trace.
    logger.warning(
        "[%s] fetch failed after %d attempts: %r",
        _log_ctx.get("ticker", "?"), attempts, last_err,
    )
    return None


def _num(x: Any) -> float | None:
    """Coerce to float, mapping None / NaN / non-numerics to None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _div(a: float | None, b: float | None) -> float | None:
    a = _num(a)
    b = _num(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(new: float | None, old: float | None) -> float | None:
    """(new/old - 1) with None on missing/zero denominator."""
    r = _div(new, old)
    return None if r is None else r - 1


def _fiscal_year(row: dict) -> int | None:
    fy = row.get("fiscalYear") or row.get("calendarYear")
    if fy is None:
        d = row.get("date")
        fy = str(d)[:4] if d else None
    try:
        return int(fy)
    except (TypeError, ValueError):
        return None


def _nearest_forward(est_rows: list, latest_fy: int | None) -> dict | None:
    """Nearest forward-year analyst-estimate row (fiscalYear > latest actual)."""
    if not est_rows or latest_fy is None:
        return None
    fwd = [r for r in est_rows if (_fiscal_year(r) or -1) > latest_fy]
    fwd.sort(key=lambda r: _fiscal_year(r) or 0)
    return fwd[0] if fwd else None


def _ntm_estimate(est_rows: list, latest_fy: int | None, field: str) -> float | None:
    """Next-twelve-months consensus for `field`: FY1/FY2 estimates blended by
    the fraction of FY1 still ahead (w*FY1 + (1-w)*FY2). Falls back to FY1
    alone when no FY2 estimate exists."""
    fy1 = _nearest_forward(est_rows, latest_fy)
    if fy1 is None:
        return None
    v1 = _num(safe_get(fy1, field))
    if v1 is None:
        return None
    fy2 = _nearest_forward(est_rows, _fiscal_year(fy1))
    v2 = _num(safe_get(fy2, field)) if fy2 else None
    try:
        days_left = (
            pd.Timestamp(fy1.get("date")).date() - datetime.now(timezone.utc).date()
        ).days
    except (ValueError, TypeError):
        days_left = None
    if v2 is None or days_left is None:
        return v1
    w = min(max(days_left / 365.0, 0.0), 1.0)
    return w * v1 + (1.0 - w) * v2


# ---------------------------------------------------------------------------
# Per-ticker fetch
# ---------------------------------------------------------------------------

@dataclass
class _Row:
    ticker: str
    benchmark: str
    note: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        v = value
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        if v is None and key not in {"note", "sector", "industry", "next_earnings"}:
            self.missing.append(key)
        self.data[key] = v


def _next_earnings_date(tk: yf.Ticker) -> str | None:
    """Next earnings date as 'YYYY-MM-DD (C)' confirmed or '(E)' estimated.

    Primary source is the raw quoteSummary calendarEvents module — the only
    place Yahoo exposes ``isEarningsDateEstimate`` — fetched through
    yfinance's own session so its cookie/crumb handling is reused.
    """
    def _raw_calendar() -> dict:
        from yfinance.data import YfData
        j = YfData().get_raw_json(
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{tk.ticker}",
            params={"modules": "calendarEvents"},
        )
        return j["quoteSummary"]["result"][0]["calendarEvents"]["earnings"]

    earnings = _retry(_raw_calendar)
    if isinstance(earnings, dict):
        epochs = [d.get("raw") for d in earnings.get("earningsDate", []) if d.get("raw")]
        if epochs:
            first = pd.Timestamp(min(epochs), unit="s", tz="UTC")
            flag = "E" if earnings.get("isEarningsDateEstimate", True) else "C"
            return f"{first:%Y-%m-%d} ({flag})"

    # Fallback: the parsed .calendar dict. It drops the estimate flag; Yahoo
    # shows a date *range* until the company confirms, so 2+ dates means (E).
    cal = _retry(lambda: tk.calendar)
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date")
        if isinstance(dates, list) and dates:
            flag = "E" if len(dates) > 1 else "C"
            try:
                return f"{pd.Timestamp(dates[0]):%Y-%m-%d} ({flag})"
            except Exception:
                return None
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
        try:
            v = cal.loc["Earnings Date"].iloc[0]
            return f"{pd.Timestamp(v):%Y-%m-%d} (E)"
        except Exception:
            pass
    return None


def _price_metrics(
    hist: pd.DataFrame,
    bench_hist: pd.DataFrame | None,
    last_info_price: float | None,
    fifty_two_week_high: float | None,
) -> dict[str, float | None]:
    """Compute YTD %, % from 52w high, 2-month annualized vol, beta vs benchmark."""
    out: dict[str, float | None] = {
        "ytd_pct": None,
        "pct_from_52w_high": None,
        "vol_2m_annualized": None,
        "beta": None,
        "avg_dollar_vol_30d": None,
    }
    if hist is None or hist.empty or "Close" not in hist:
        return out

    close = hist["Close"].dropna()
    if close.empty:
        return out

    last_px = _num(last_info_price) or _num(close.iloc[-1])

    # YTD: first close on/after Jan 1 of the most recent close's year.
    last_dt = pd.Timestamp(close.index[-1]).tz_localize(None)
    year_start = pd.Timestamp(year=last_dt.year, month=1, day=1)
    close_naive = close.copy()
    close_naive.index = pd.to_datetime(close_naive.index).tz_localize(None)
    ytd_slice = close_naive[close_naive.index >= year_start]
    if not ytd_slice.empty and last_px is not None:
        first_px = _num(ytd_slice.iloc[0])
        if first_px:
            out["ytd_pct"] = (last_px / first_px) - 1.0

    # % from 52w high (prefer info field, else rolling max over last ~252 days).
    high_52w = _num(fifty_two_week_high)
    if high_52w is None:
        recent = close_naive.tail(252)
        if not recent.empty:
            high_52w = _num(recent.max())
    if high_52w and last_px is not None:
        out["pct_from_52w_high"] = (last_px / high_52w) - 1.0

    # 2-month (~42 trading days) annualized realized vol of daily log returns.
    log_ret = np.log(close_naive / close_naive.shift(1)).dropna()
    if len(log_ret) >= 21:
        recent_ret = log_ret.tail(42)
        if len(recent_ret) >= 10:
            out["vol_2m_annualized"] = float(recent_ret.std(ddof=1) * np.sqrt(252))

    # Avg daily $ volume over last ~30 trading days.
    if "Volume" in hist:
        vol30 = hist[["Close", "Volume"]].dropna().tail(30)
        if not vol30.empty:
            dollar = (vol30["Close"] * vol30["Volume"]).mean()
            out["avg_dollar_vol_30d"] = _num(dollar)

    # Beta via OLS on trailing ~1y of overlapping daily returns vs the assigned benchmark.
    if bench_hist is not None and not bench_hist.empty and "Close" in bench_hist:
        bench_close = bench_hist["Close"].dropna()
        bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None)
        stock_ret = close_naive.pct_change().dropna()
        bench_ret = bench_close.pct_change().dropna()
        joined = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
        joined.columns = ["s", "b"]
        joined = joined.tail(252)
        if len(joined) >= 60:
            var_b = float(joined["b"].var(ddof=1))
            if var_b > 0:
                cov = float(joined["s"].cov(joined["b"]))
                out["beta"] = cov / var_b

    return out


def _fetch_one(
    ticker: str,
    benchmark: str,
    note: str,
    bench_cache: dict[str, pd.DataFrame],
    delay: float,
    fmp: FMPClient,
) -> _Row:
    _log_ctx["ticker"] = ticker
    row = _Row(ticker=ticker, benchmark=benchmark, note=note or "")

    def _ep(name: str, fn: Callable[[], Any], default: Any) -> Any:
        """One FMP endpoint call; failures degrade to `default` + a log line."""
        try:
            return fn()
        except FMPAccessError as e:
            logger.warning("[%s] FMP %s gated on this plan: %r", ticker, name, e)
        except Exception as e:  # FMPError / network after tenacity gave up
            logger.warning("[%s] FMP %s failed: %r", ticker, name, e)
        return default

    quote = _ep("quote", lambda: fmp.quote(ticker), {})
    profile = _ep("profile", lambda: fmp.profile(ticker), {})
    is_fund = bool(safe_get(profile, "isEtf")) or bool(safe_get(profile, "isFund"))
    if is_fund:
        # ETFs/funds have no fundamentals — don't spend the API calls.
        ratios: dict = {}
        km: dict = {}
        inc: list = []
        cash0: dict = {}
        est: list = []
    else:
        ratios = _ep("ratios-ttm", lambda: fmp.ratios_ttm(ticker), {})
        km = _ep("key-metrics-ttm", lambda: fmp.key_metrics_ttm(ticker), {})
        inc = _ep("income-statement", lambda: fmp.income_statement(ticker, limit=2), [])
        cash0 = first(_ep("cash-flow", lambda: fmp.cash_flow(ticker, limit=1), []))
        est = _ep("analyst-estimates", lambda: fmp.analyst_estimates(ticker, limit=10), [])
    inc0 = inc[0] if inc else {}
    inc1 = inc[1] if len(inc) > 1 else {}

    tk = yf.Ticker(ticker)
    # ~2y of daily prices powers YTD, vol, beta — still Yahoo (free, uncapped;
    # FMP gates ^NDX history on the free tier).
    hist = _retry(lambda: tk.history(period="2y", interval="1d", auto_adjust=False))
    if not isinstance(hist, pd.DataFrame):
        hist = pd.DataFrame()

    bench_symbol = BENCHMARK_SYMBOLS.get(benchmark.upper(), BENCHMARK_SYMBOLS["SPX"])
    bench_hist = bench_cache.get(bench_symbol)

    # ---- quote + profile --------------------------------------------------
    last_price = _num(safe_get(quote, "price"))
    high_52w = _num(safe_get(quote, "yearHigh"))
    mcap = _num(safe_get(quote, "marketCap"))
    rev0 = _num(safe_get(inc0, "revenue"))

    row.set("sector", safe_get(profile, "sector"))
    row.set("industry", safe_get(profile, "industry"))
    row.set("last_price", last_price)
    row.set("fifty_two_week_high", high_52w)
    row.set("fifty_two_week_low", _num(safe_get(quote, "yearLow")))
    row.set("market_cap", mcap)

    # ---- ratios-ttm (units probe-confirmed: true ratios, decimal margins) --
    pe_ttm = _num(safe_get(ratios, "priceToEarningsRatioTTM"))
    row.set("pe_trailing", pe_ttm)
    ps = _num(safe_get(ratios, "priceToSalesRatioTTM"))
    if ps is None:
        ps = _div(mcap, rev0)
    row.set("ps", ps)
    row.set("pb", _num(safe_get(ratios, "priceToBookRatioTTM")))
    row.set("current_ratio", _num(safe_get(ratios, "currentRatioTTM")))
    row.set("quick_ratio", _num(safe_get(ratios, "quickRatioTTM")))
    row.set("debt_equity", _num(safe_get(ratios, "debtToEquityRatioTTM")))
    row.set("debt_assets", _num(safe_get(ratios, "debtToAssetsRatioTTM")))
    row.set("gross_margin", _num(safe_get(ratios, "grossProfitMarginTTM")))
    row.set("operating_margin", _num(safe_get(ratios, "operatingProfitMarginTTM")))
    row.set("peg", _num(safe_get(ratios, "priceToEarningsGrowthRatioTTM")))

    # ---- key-metrics-ttm (ROA/ROE + EV multiples live here, not ratios) ----
    row.set("roa", _num(safe_get(km, "returnOnAssetsTTM")))
    row.set("roe", _num(safe_get(km, "returnOnEquityTTM")))
    row.set("ev_ebitda", _num(safe_get(km, "evToEBITDATTM")))
    row.set("ev_sales", _num(safe_get(km, "evToSalesTTM")))
    row.set("fcf_yield", _num(safe_get(km, "freeCashFlowYieldTTM")))
    row.set("net_debt_ebitda", _num(safe_get(km, "netDebtToEBITDATTM")))

    # ---- statements --------------------------------------------------------
    row.set("free_cash_flow", _num(safe_get(cash0, "freeCashFlow")))
    row.set("rev_growth_ttm", _growth(rev0, _num(safe_get(inc1, "revenue"))))
    ebit = _num(safe_get(inc0, "ebit"))
    if ebit is None:
        ebit = _num(safe_get(inc0, "operatingIncome"))
    int_exp = _num(safe_get(inc0, "interestExpense"))
    # interestExpense is reported positive; 0 interest = coverage undefined.
    row.set(
        "interest_coverage",
        _div(ebit, abs(int_exp)) if int_exp not in (None, 0) else None,
    )

    # ---- forward estimates: NTM = time-weighted FY1/FY2 consensus ----------
    latest_fy = _fiscal_year(inc0)
    ntm_eps = _ntm_estimate(est, latest_fy, "epsAvg")
    ntm_rev = _ntm_estimate(est, latest_fy, "revenueAvg")
    # TTM bases derived from the TTM ratios (price/PE, mcap/PS) so numerator
    # and denominator are both twelve-month windows.
    ttm_eps = _div(last_price, pe_ttm)
    if ttm_eps is not None and ttm_eps <= 0:
        ttm_eps = None  # growth vs negative earnings is not meaningful
    row.set("fwd_eps_growth", _growth(ntm_eps, ttm_eps))
    row.set("fwd_rev_growth", _growth(ntm_rev, _div(mcap, ps)))
    row.set("pe_forward", _div(last_price, ntm_eps))

    # ---- dividend yield: computed trailing div/share ÷ price ---------------
    # (reproduces FMP's dividendYieldTTM to ~5dp; absent field on a live
    # ratios payload = genuine non-payer, so 0.0 is real, not missing)
    dps = _num(safe_get(ratios, "dividendPerShareTTM"))
    if dps is None and ratios:
        dps = 0.0
    row.set("dividend_yield", _div(dps, last_price))

    # ---- price-derived (yfinance history) ----------------------------------
    pm = _price_metrics(hist, bench_hist, last_price, high_52w)
    for k, v in pm.items():
        row.set(k, v)

    # ETFs/funds have no earnings; skip the lookup rather than retry a 404.
    if is_fund:
        row.set("next_earnings", None)
    else:
        row.set("next_earnings", _next_earnings_date(tk))

    if delay > 0:
        time.sleep(delay)
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(tickers_df: pd.DataFrame, *, delay: float = 0.25) -> pd.DataFrame:
    """Build a one-row-per-ticker fundamentals/valuation report.

    Parameters
    ----------
    tickers_df : DataFrame with columns ``ticker`` and ``benchmark`` (SPX or NDX);
        optional ``note`` column is passed through.
    delay : per-ticker sleep in seconds to be polite to Yahoo.
    """
    if tickers_df is None or tickers_df.empty:
        return pd.DataFrame(columns=COLUMNS)

    fmp = FMPClient()  # raises FMPError if FMP_API_KEY is not set

    df = tickers_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("tickers_df must have a 'ticker' column")
    if "benchmark" not in df.columns:
        df["benchmark"] = "SPX"
    if "note" not in df.columns:
        df["note"] = ""
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["benchmark"] = df["benchmark"].astype(str).str.strip().str.upper()
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"])

    # Pre-fetch each needed benchmark's 2y history once.
    needed_benches = {BENCHMARK_SYMBOLS.get(b, BENCHMARK_SYMBOLS["SPX"]) for b in df["benchmark"]}
    bench_cache: dict[str, pd.DataFrame] = {}
    logger.info("Fetching %d tickers (FMP fundamentals + Yahoo prices)", len(df))
    for sym in needed_benches:
        _log_ctx["ticker"] = sym
        h = _retry(lambda s=sym: yf.Ticker(s).history(period="2y", interval="1d", auto_adjust=False))
        if not isinstance(h, pd.DataFrame) or h.empty:
            logger.error("[%s] benchmark history unavailable — beta will be null", sym)
        bench_cache[sym] = h if isinstance(h, pd.DataFrame) else pd.DataFrame()

    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        try:
            row = _fetch_one(r["ticker"], r["benchmark"], str(r.get("note", "")), bench_cache, delay, fmp)
        except Exception as e:
            logger.error("[%s] fatal during fetch: %r", r["ticker"], e)
            row = _Row(ticker=r["ticker"], benchmark=r["benchmark"], note=str(r.get("note", "")))
            row.missing.append(f"fatal:{type(e).__name__}")
        rec: dict[str, Any] = {
            "ticker": row.ticker,
            "benchmark": row.benchmark,
            "note": row.note,
            "missing_fields": ",".join(row.missing) if row.missing else "",
            "as_of": as_of,
        }
        rec.update(row.data)
        rows.append(rec)

    logger.info("FMP calls used this pull: %d (free tier caps at 250/day)", fmp.call_count)
    out = pd.DataFrame(rows)
    # Ensure full schema present + stable column order.
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[COLUMNS]


if __name__ == "__main__":
    # Smoke test entry point.
    import sys
    syms = sys.argv[1:] or ["AAPL", "MSFT"]
    sample = pd.DataFrame({"ticker": syms, "benchmark": ["NDX"] * len(syms)})
    report = build_report(sample)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(report.T)
