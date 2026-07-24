"""Pure data layer for stock fundamentals / valuation reports.

Public surface: ``build_report(tickers_df)`` returns one row per input ticker
with a fixed schema (see ``COLUMNS``). No Streamlit imports — safe to call
from a notebook, CLI script, or GitHub Action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


BENCHMARK_SYMBOLS = {"SPX": "^GSPC", "NDX": "^NDX"}

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
    # Swallow — caller treats None as missing data.
    _ = last_err
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
    """Best-effort next earnings date as ISO YYYY-MM-DD."""
    # Newer yfinance versions: .get_earnings_dates() returns a DataFrame indexed by datetime.
    df = _retry(lambda: tk.get_earnings_dates(limit=8))
    if isinstance(df, pd.DataFrame) and not df.empty:
        try:
            idx = pd.to_datetime(df.index, errors="coerce", utc=True)
            now = pd.Timestamp.now(tz="UTC")
            future = idx[idx >= now]
            if len(future) > 0:
                return future.min().strftime("%Y-%m-%d")
        except Exception:
            pass
    # Older fallback: .calendar
    cal = _retry(lambda: tk.calendar)
    if isinstance(cal, dict):
        d = cal.get("Earnings Date")
        if isinstance(d, list) and d:
            d = d[0]
        if isinstance(d, (datetime, pd.Timestamp)):
            return pd.Timestamp(d).strftime("%Y-%m-%d")
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
        try:
            v = cal.loc["Earnings Date"].iloc[0]
            return pd.Timestamp(v).strftime("%Y-%m-%d")
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
) -> _Row:
    row = _Row(ticker=ticker, benchmark=benchmark, note=note or "")

    tk = yf.Ticker(ticker)
    info = _retry(lambda: tk.get_info()) or _retry(lambda: tk.info) or {}
    if not isinstance(info, dict):
        info = {}

    # ~2y of daily prices powers YTD, vol, beta. Single network call per ticker.
    hist = _retry(lambda: tk.history(period="2y", interval="1d", auto_adjust=False))
    if not isinstance(hist, pd.DataFrame):
        hist = pd.DataFrame()

    bench_symbol = BENCHMARK_SYMBOLS.get(benchmark.upper(), BENCHMARK_SYMBOLS["SPX"])
    bench_hist = bench_cache.get(bench_symbol)

    # ---- direct .info lookups ---------------------------------------------
    last_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    high_52w = info.get("fiftyTwoWeekHigh")
    low_52w = info.get("fiftyTwoWeekLow")
    market_cap = info.get("marketCap")
    pe_trailing = info.get("trailingPE")
    pe_forward = info.get("forwardPE")
    ps = info.get("priceToSalesTrailing12Months")
    pb = info.get("priceToBook")
    rev_growth_ttm = info.get("revenueGrowth")
    roa = info.get("returnOnAssets")
    roe = info.get("returnOnEquity")
    fcf = info.get("freeCashflow")
    current_ratio = info.get("currentRatio")
    quick_ratio = info.get("quickRatio")
    debt_equity = info.get("debtToEquity")
    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    sector = info.get("sector")
    industry = info.get("industry")
    div_yield = info.get("dividendYield")
    # Forward growth: Yahoo's "earningsGrowth" is YoY recent quarter (best-effort proxy).
    fwd_eps_growth = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
    fwd_rev_growth = info.get("revenueGrowth")  # Yahoo doesn't expose forward separately

    # yfinance >=0.2.40 returns dividendYield in percent points (0.35 == 0.35%);
    # normalize to a fraction so the UI can format it like every other ratio.
    div_yield_n = _num(div_yield)
    if div_yield_n is not None:
        div_yield_n = div_yield_n / 100.0
    elif _num(market_cap) is not None:
        # Yahoo omits dividendYield entirely for non-payers; only treat it as
        # missing when the whole .info fetch came back empty.
        div_yield_n = 0.0

    # debtToEquity from Yahoo is in percent (e.g. 150.0 means 1.50). Normalize.
    de_n = _num(debt_equity)
    if de_n is not None and abs(de_n) > 5:
        de_n = de_n / 100.0

    row.set("sector", sector)
    row.set("industry", industry)
    row.set("last_price", _num(last_price))
    row.set("fifty_two_week_high", _num(high_52w))
    row.set("fifty_two_week_low", _num(low_52w))
    row.set("market_cap", _num(market_cap))
    row.set("pe_trailing", _num(pe_trailing))
    row.set("pe_forward", _num(pe_forward))
    row.set("ps", _num(ps))
    row.set("pb", _num(pb))
    row.set("rev_growth_ttm", _num(rev_growth_ttm))
    row.set("fwd_eps_growth", _num(fwd_eps_growth))
    row.set("fwd_rev_growth", _num(fwd_rev_growth))
    row.set("roa", _num(roa))
    row.set("roe", _num(roe))
    row.set("free_cash_flow", _num(fcf))
    row.set("current_ratio", _num(current_ratio))
    row.set("quick_ratio", _num(quick_ratio))
    row.set("debt_equity", de_n)
    row.set("gross_margin", _num(gross_margin))
    row.set("operating_margin", _num(operating_margin))
    row.set("dividend_yield", div_yield_n)

    # ---- price-derived ----------------------------------------------------
    pm = _price_metrics(hist, bench_hist, _num(last_price), _num(high_52w))
    for k, v in pm.items():
        row.set(k, v)

    # ---- derived ratios ---------------------------------------------------
    ev = _num(info.get("enterpriseValue"))
    ebitda = _num(info.get("ebitda"))
    total_revenue = _num(info.get("totalRevenue"))
    total_debt = _num(info.get("totalDebt"))
    total_assets = _num(info.get("totalAssets"))
    if total_assets is None:
        # .info only carries totalAssets for funds/ETFs; equities need the
        # balance sheet.
        bs = _retry(lambda: tk.balance_sheet)
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            latest_bs = bs.iloc[:, 0]
            for key in ("Total Assets", "TotalAssets"):
                if key in bs.index:
                    total_assets = _num(latest_bs.get(key))
                    if total_assets is not None:
                        break
    total_cash = _num(info.get("totalCash"))
    mcap = _num(market_cap)
    # EBIT and interest expense aren't on .info reliably; pull from financials best-effort.
    ebit = None
    interest_exp = None
    fin = _retry(lambda: tk.financials)
    if isinstance(fin, pd.DataFrame) and not fin.empty:
        latest = fin.iloc[:, 0]
        for key in ("EBIT", "Ebit", "Operating Income", "OperatingIncome"):
            if key in fin.index:
                ebit = _num(latest.get(key))
                if ebit is not None:
                    break
        for key in ("Interest Expense", "InterestExpense"):
            if key in fin.index:
                v = _num(latest.get(key))
                if v is not None:
                    interest_exp = abs(v)  # often reported as negative
                    break

    row.set("ev_ebitda", _div(ev, ebitda))
    row.set("ev_sales", _div(ev, total_revenue))
    row.set("fcf_yield", _div(_num(fcf), mcap))
    row.set("debt_assets", _div(total_debt, total_assets))
    net_debt = None
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash
    row.set("net_debt_ebitda", _div(net_debt, ebitda))
    row.set("interest_coverage", _div(ebit, interest_exp))
    # PEG = trailing P/E / (forward EPS growth as percent points). Defensive.
    peg = None
    fwd_eps_n = _num(fwd_eps_growth)
    pe_n = _num(pe_trailing)
    if pe_n is not None and fwd_eps_n is not None and fwd_eps_n != 0:
        peg = pe_n / (fwd_eps_n * 100.0)
    row.set("peg", peg)

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
    for sym in needed_benches:
        h = _retry(lambda s=sym: yf.Ticker(s).history(period="2y", interval="1d", auto_adjust=False))
        bench_cache[sym] = h if isinstance(h, pd.DataFrame) else pd.DataFrame()

    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        try:
            row = _fetch_one(r["ticker"], r["benchmark"], str(r.get("note", "")), bench_cache, delay)
        except Exception as e:
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
