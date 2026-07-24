"""Streamlit frontend for the daily stock fundamentals report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from metrics import COLUMNS, build_report
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

snapshot = _load_snapshot()
if "report" not in st.session_state:
    st.session_state["report"] = snapshot
if "tickers_df" not in st.session_state:
    st.session_state["tickers_df"] = _load_default_tickers()

with st.sidebar:
    st.header("Tickers")
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

    refresh = st.button("Refresh data", type="primary", use_container_width=True)
    if st.button("Reset to default list", use_container_width=True):
        st.session_state["tickers_df"] = _load_default_tickers()
        st.rerun()

    if snapshot is not None and "as_of" in snapshot.columns and not snapshot.empty:
        as_of = snapshot["as_of"].dropna().iloc[0] if snapshot["as_of"].notna().any() else "unknown"
        st.info(f"Cached snapshot as of {as_of}")
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
            st.session_state["report"] = _cached_build_report(tuples)

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

display = report.copy()
display = display.drop(columns=["sector", "industry"], errors="ignore")
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
]
ratio_cols = [
    labels[c]
    for c in (
        "pe_trailing", "pe_forward", "ps", "pb", "peg",
        "ev_ebitda", "ev_sales", "current_ratio", "quick_ratio",
        "debt_equity", "debt_assets", "net_debt_ebitda", "interest_coverage",
        "beta",
    )
]
money_cols = [labels[c] for c in ("last_price", "fifty_two_week_high", "fifty_two_week_low")]
int_cols = [labels[c] for c in ("free_cash_flow", "avg_dollar_vol_30d")]

fmt: dict[str, object] = {labels["market_cap"]: _abbrev_dollars}
for c in percent_cols:
    fmt[c] = "{:+.2%}"
for c in ratio_cols:
    fmt[c] = "{:.2f}"
for c in money_cols:
    fmt[c] = "{:,.2f}"
for c in int_cols:
    fmt[c] = "{:,.0f}"

styled = (
    display.style
    .format(fmt, na_rep="—")
    .map(_color_signed, subset=[labels["ytd_pct"], labels["pct_from_52w_high"]])
)

st.dataframe(styled, use_container_width=True, hide_index=True)

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
