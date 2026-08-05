"""Streamlit frontend for the daily stock fundamentals report."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Streamlit Cloud provides secrets via st.secrets; clients read them from the
# environment (locally .env via python-dotenv). Bridge the two.
try:
    for _k in ("FMP_API_KEY", "POLYGON_API_KEY", "LOG_GIST_ID", "LOG_GIST_TOKEN"):
        if _k in st.secrets:
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:
    pass

from metrics import COLUMNS, build_report
from options_view import render as render_options
from snapshot import (
    DEFAULT_PARQUET,
    DEFAULT_TICKERS_CSV,
    EXCEL_FORMATS,
    build_excel_bytes,
)


REPO_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Stock List Fundamentals",
    page_icon=":bar_chart:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session log: metrics.py and this file log to "stocklist"; entries land in a
# session buffer surfaced by the Logs expander. The handler is swapped every
# rerun, so with concurrent viewers module-level logs go to the most recent
# session — fine for a single-user dashboard.
# ---------------------------------------------------------------------------

_MAX_LOG_LINES = 400
_DUP_RE = re.compile(r" \(x(\d+)\)$")


def _scrub(msg: str) -> str:
    """Never let an API key reach the log buffer (e.g. via a request URL)."""
    return re.sub(r"(api[_-]?key=)[^&\s'\"]+", r"\1***", msg, flags=re.I)


def _log_core(line: str) -> str:
    """Line minus timestamp and any (xN) repeat suffix, for dedup compare."""
    core = line.split(" ", 1)[1] if " " in line else line
    return _DUP_RE.sub("", core)


def _ship_logs(lines: list[str]) -> None:
    """Mirror the tail of the log buffer to a private gist (throttled, async)
    so errors can be read remotely without console access."""
    gid, tok = os.environ.get("LOG_GIST_ID"), os.environ.get("LOG_GIST_TOKEN")
    if not (gid and tok):
        return
    now = time.time()
    if now - getattr(_ship_logs, "_last", 0.0) < 20:
        return
    _ship_logs._last = now
    content = "\n".join(lines[-200:]) or "(empty)"

    def _post() -> None:
        try:
            requests.patch(
                f"https://api.github.com/gists/{gid}",
                json={"files": {"jdctickers-errors.log": {"content": content}}},
                headers={
                    "Authorization": f"Bearer {tok}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
        except Exception:
            pass  # log shipping must never break the app

    threading.Thread(target=_post, daemon=True).start()


class _SessionLogHandler(logging.Handler):
    def __init__(self, buf: list[str]) -> None:
        super().__init__(level=logging.INFO)
        self.buf = buf

    def emit(self, record: logging.LogRecord) -> None:
        msg = _scrub(self.format(record))
        # Collapse consecutive repeats (live 5s refresh can retry the same
        # failure) into one line with an (xN) counter.
        if self.buf and _log_core(self.buf[-1]) == _log_core(msg):
            n = int(m.group(1)) + 1 if (m := _DUP_RE.search(self.buf[-1])) else 2
            self.buf[-1] = _DUP_RE.sub("", self.buf[-1]) + f" (x{n})"
        else:
            self.buf.append(msg)
        del self.buf[:-_MAX_LOG_LINES]
        if record.levelno >= logging.ERROR:
            _ship_logs(self.buf)


if "_logs" not in st.session_state:
    st.session_state["_logs"] = []
LOGS: list[str] = st.session_state["_logs"]

logger = logging.getLogger("stocklist")
logger.setLevel(logging.INFO)
for _h in list(logger.handlers):
    if isinstance(_h, _SessionLogHandler):
        logger.removeHandler(_h)
_handler = _SessionLogHandler(LOGS)
_fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
_fmt.converter = time.gmtime
_handler.setFormatter(_fmt)
logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Optional password gate (only enforced if APP_PASSWORD is set in secrets)
# ---------------------------------------------------------------------------

def _password_gate() -> bool:
    required = None
    try:
        required = st.secrets.get("APP_PASSWORD")  # type: ignore[union-attr]
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):  # type: ignore[attr-defined]
        required = None
    except Exception:
        required = None
    if not required:
        return True
    if st.session_state.get("_auth_ok"):
        return True
    with st.form("auth"):
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok and pw == required:
        st.session_state["_auth_ok"] = True
        st.rerun()
    if ok:
        st.error("Incorrect password.")
    return False


if not _password_gate():
    st.stop()


# ---------------------------------------------------------------------------
# Data loading: cached snapshot + live refresh
# ---------------------------------------------------------------------------

def _load_snapshot() -> pd.DataFrame | None:
    path = REPO_ROOT / "snapshot.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error("Snapshot load failed: %r", e)
        st.warning(f"Couldn't load cached snapshot: {e}")
        return None


def _load_default_tickers() -> pd.DataFrame:
    if DEFAULT_TICKERS_CSV.exists():
        df = pd.read_csv(DEFAULT_TICKERS_CSV)
    else:
        df = pd.DataFrame({"ticker": ["AAPL"], "benchmark": ["NDX"], "note": [""]})
    for col in ("ticker", "benchmark", "note"):
        if col not in df.columns:
            df[col] = "" if col == "note" else ("SPX" if col == "benchmark" else "")
    return df[["ticker", "benchmark", "note"]]


@st.cache_data(ttl=900, show_spinner=False)
def _cached_build_report(tickers_tuple: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
    df = pd.DataFrame(list(tickers_tuple), columns=["ticker", "benchmark", "note"])
    return build_report(df)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title(":bar_chart: Stock List Fundamentals")
st.caption("Daily valuation snapshot for a configurable ticker list. Free Yahoo data via yfinance.")

tab_fund, tab_opts = st.tabs(["Fundamentals", "Options"])

with tab_fund:
    snapshot = _load_snapshot()
    if "report" not in st.session_state:
        st.session_state["report"] = snapshot
    if "tickers_df" not in st.session_state:
        st.session_state["tickers_df"] = _load_default_tickers()

    with st.form("quick_add", clear_on_submit=True, border=False):
        c1, c2, c3 = st.columns([2, 1, 5])
        new_ticker = c1.text_input(
            "Add ticker", placeholder="e.g. ORCL", label_visibility="collapsed"
        )
        add_clicked = c2.form_submit_button("Add ticker", use_container_width=True)

    if add_clicked and new_ticker.strip():
        t = new_ticker.strip().upper()
        tdf = st.session_state["tickers_df"]
        if t in tdf["ticker"].astype(str).str.upper().values:
            st.toast(f"{t} is already in the list.")
        else:
            # Fetch ONLY the new ticker (~7 FMP calls) and append to the cached
            # table — a full-list re-pull would burn the free tier's daily budget.
            one = None
            with st.spinner(f"Fetching {t}..."):
                try:
                    one = build_report(
                        pd.DataFrame([{"ticker": t, "benchmark": "SPX", "note": ""}])
                    )
                except Exception as e:
                    logger.error("[%s] add-ticker pull failed: %r", t, e)
                    st.error(f"Couldn't add {t}: {e}")
            # Accept the row if ANYTHING came back (Yahoo price metrics still fill
            # when the FMP budget is exhausted); fundamentals backfill on the next
            # successful full refresh. Only a totally-empty row = bad symbol.
            has_any = False
            if one is not None and not one.empty:
                data_only = one.drop(
                    columns=["ticker", "benchmark", "note", "missing_fields", "as_of"],
                    errors="ignore",
                )
                has_any = bool(data_only.notna().any().any())
            if has_any:
                st.session_state["tickers_df"] = pd.concat(
                    [tdf, pd.DataFrame([{"ticker": t, "benchmark": "SPX", "note": ""}])],
                    ignore_index=True,
                )
                rep = st.session_state.get("report")
                st.session_state["report"] = (
                    pd.concat([rep, one], ignore_index=True) if rep is not None else one
                )
                # Reset the editor widget so it picks up the appended row.
                st.session_state.pop("ticker_editor", None)
                st.rerun()
            elif one is not None:
                logger.error("[%s] add-ticker fetch returned no data at all", t)
                st.error(f"No data at all for {t} — probably a bad symbol (see Logs).")

    with st.expander("Edit ticker list"):
        st.caption("Edit rows, add/delete, then click **Refresh data**.")
        edited = st.data_editor(
            st.session_state["tickers_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "benchmark": st.column_config.SelectboxColumn(
                    "Benchmark", options=["SPX", "NDX"], required=True
                ),
                "note": st.column_config.TextColumn("Note"),
            },
            key="ticker_editor",
        )
        st.session_state["tickers_df"] = edited

        refresh = st.button("Refresh data", type="primary")

        if snapshot is not None and "as_of" in snapshot.columns and not snapshot.empty:
            as_of = snapshot["as_of"].dropna().iloc[0] if snapshot["as_of"].notna().any() else "unknown"
            st.caption(f"Cached snapshot as of {as_of}")
        else:
            st.warning("No cached snapshot — click **Refresh data**.")

    if refresh:
        clean = (
            st.session_state["tickers_df"]
            .dropna(subset=["ticker"])
            .assign(
                ticker=lambda d: d["ticker"].astype(str).str.strip().str.upper(),
                benchmark=lambda d: d["benchmark"].fillna("SPX").astype(str).str.upper(),
                note=lambda d: d["note"].fillna("").astype(str),
            )
        )
        clean = clean[clean["ticker"] != ""].drop_duplicates(subset=["ticker"])
        if clean.empty:
            st.error("Add at least one ticker.")
        else:
            with st.spinner(f"Fetching {len(clean)} tickers from Yahoo..."):
                tuples = tuple((r.ticker, r.benchmark, r.note) for r in clean.itertuples(index=False))
                t0 = time.time()
                try:
                    st.session_state["report"] = _cached_build_report(tuples)
                except Exception as e:
                    logger.error("Data pull failed: %r", e)
                    st.error(f"Data pull failed: {e} — details in Logs below.")
                else:
                    st.session_state["last_pull_utc"] = datetime.now(timezone.utc)
                    logger.info("Pulled %d tickers in %.1fs", len(clean), time.time() - t0)

    report: pd.DataFrame | None = st.session_state.get("report")

    if report is None or report.empty:
        st.info("No data yet. Edit the ticker list and click **Refresh data**.")
        st.stop()

    # Ensure full schema even if loaded from an older snapshot.
    for col in COLUMNS:
        if col not in report.columns:
            report[col] = None
    report = report[COLUMNS]

    # ---- Styled, sortable table ----------------------------------------------

    VIEW_GROUPS: dict[str, list[str] | None] = {
        "All": None,
        "Valuation": [
            "market_cap", "pe_trailing", "pe_forward", "ps", "pb", "peg",
            "ev_ebitda", "ev_sales", "fcf_yield", "dividend_yield",
        ],
        "Growth": ["rev_growth_ttm", "fwd_eps_growth", "fwd_rev_growth", "next_earnings"],
        "Profitability": ["roa", "roe", "gross_margin", "operating_margin", "free_cash_flow"],
        "Balance sheet": [
            "current_ratio", "quick_ratio", "debt_equity", "debt_assets",
            "net_debt_ebitda", "interest_coverage",
        ],
        "Price": [
            "last_price", "ytd_pct", "pct_from_52w_high", "fifty_two_week_high",
            "fifty_two_week_low", "vol_2m_annualized", "beta", "avg_dollar_vol_30d",
        ],
    }
    last_pull = st.session_state.get("last_pull_utc")
    if last_pull is not None:
        st.caption(f"Last data pull: {last_pull:%Y-%m-%d %H:%M:%S} UTC (live this session)")
    elif report["as_of"].notna().any():
        st.caption(f"Last data pull: {report['as_of'].dropna().iloc[0]} (cached snapshot)")

    view = st.radio("View", list(VIEW_GROUPS), horizontal=True, label_visibility="collapsed")

    display = report.copy()
    display = display.drop(columns=["sector", "industry"], errors="ignore")
    group = VIEW_GROUPS[view]
    if group is not None:
        display = display[["ticker"] + [c for c in group if c in display.columns]]
    labels = {c: EXCEL_FORMATS.get(c, (c, "@"))[0] for c in display.columns}
    display = display.rename(columns=labels)


    def _abbrev_dollars(val: object) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(v) >= div:
                x = v / div
                return f"{x:.0f}{suffix}" if abs(x) >= 100 else f"{x:.1f}{suffix}"
        return f"{v:,.0f}"


    def _color_signed(val: object) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v > 0:
            return "color: #1a7f37; font-weight: 600;"
        if v < 0:
            return "color: #cf222e; font-weight: 600;"
        return ""


    percent_cols = [
        labels[c]
        for c in (
            "ytd_pct", "pct_from_52w_high", "fcf_yield", "rev_growth_ttm",
            "fwd_eps_growth", "fwd_rev_growth", "roa", "roe",
            "gross_margin", "operating_margin", "dividend_yield", "vol_2m_annualized",
        )
        if c in labels
    ]
    ratio_cols = [
        labels[c]
        for c in (
            "pe_trailing", "pe_forward", "ps", "pb", "peg",
            "ev_ebitda", "ev_sales", "current_ratio", "quick_ratio",
            "debt_equity", "debt_assets", "net_debt_ebitda", "interest_coverage",
            "beta",
        )
        if c in labels
    ]
    money_cols = [
        labels[c]
        for c in ("last_price", "fifty_two_week_high", "fifty_two_week_low")
        if c in labels
    ]
    int_cols = [labels[c] for c in ("free_cash_flow", "avg_dollar_vol_30d") if c in labels]

    fmt: dict[str, object] = {}
    if "market_cap" in labels:
        fmt[labels["market_cap"]] = _abbrev_dollars
    for c in percent_cols:
        fmt[c] = "{:+.2%}"
    for c in ratio_cols:
        fmt[c] = "{:.2f}"
    for c in money_cols:
        fmt[c] = "{:,.2f}"
    for c in int_cols:
        fmt[c] = "{:,.0f}"

    styled = display.style.format(fmt, na_rep="—")
    signed_cols = [labels[c] for c in ("ytd_pct", "pct_from_52w_high") if c in labels]
    if signed_cols:
        styled = styled.map(_color_signed, subset=signed_cols)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={labels["ticker"]: st.column_config.Column(pinned=True)},
    )

    # ---- Excel download -------------------------------------------------------

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    xlsx_bytes = build_excel_bytes(report)
    st.download_button(
        "Download as Excel",
        data=xlsx_bytes,
        file_name=f"stock_fundamentals_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ---- Diagnostics ----------------------------------------------------------

    with st.expander("Diagnostics: missing fields per ticker"):
        diag = report[["ticker", "missing_fields"]].copy()
        diag = diag[diag["missing_fields"].fillna("").str.len() > 0]
        if diag.empty:
            st.success("All fields populated for every ticker.")
        else:
            diag_text = "\n".join(
                f"{r.ticker}: {r.missing_fields}" for r in diag.itertuples(index=False)
            )
            st.code(diag_text, language=None)
            st.caption(
                "Yahoo's free feed has gaps — common nulls: `next_earnings`, forward "
                "growth, bank tickers (no FCF/EBITDA-based metrics), and certain "
                "non-US tickers."
            )

with tab_opts:
    render_options()

# ---- Logs -----------------------------------------------------------------

with st.expander(f"Logs ({len(LOGS)} entries)"):
    if LOGS:
        st.code("\n".join(LOGS), language=None)
        st.caption("Copy icon (top-right of the block) copies everything.")
        if st.button("Clear logs"):
            st.session_state["_logs"] = []
            st.rerun()
    else:
        st.caption(
            "Nothing logged this session. Fetch retries, data-loading failures, "
            "and pull results will appear here."
        )
