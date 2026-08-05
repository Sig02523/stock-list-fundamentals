"""Options tab: live-ticking positions/greeks + chain lookup (Polygon)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from polygon_options import (
    PolygonError,
    build_occ,
    chain_snapshot,
    contract_snapshot,
    list_expirations,
)

logger = logging.getLogger("stocklist")
POSITIONS_CSV = Path(__file__).resolve().parent / "positions.csv"

QUOTE_COLS = ["bid", "ask", "ivm", "delta", "gamma", "vega", "theta"]
LIVE_INTERVAL = "5s"

_NUM_FMT = {
    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
    "bid": st.column_config.NumberColumn("Bid", format="%.2f"),
    "ask": st.column_config.NumberColumn("Offer", format="%.2f"),
    "ivm": st.column_config.NumberColumn("IVM", format="percent"),
    "delta": st.column_config.NumberColumn("Delta", format="%.4f"),
    "gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
    "vega": st.column_config.NumberColumn("Vega", format="%.4f"),
    "theta": st.column_config.NumberColumn("Theta", format="%.4f"),
    "open_interest": st.column_config.NumberColumn("OI", format="%d"),
    "volume": st.column_config.NumberColumn("Volume", format="%d"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def _expirations(ticker: str) -> list[str]:
    return list_expirations(ticker)


@st.cache_data(ttl=4, show_spinner=False)
def _chain(ticker: str, expiry: str, cp: str) -> pd.DataFrame:
    return pd.DataFrame(chain_snapshot(ticker, expiry, cp))


def _fetch_positions(records: tuple) -> pd.DataFrame:
    """Live quotes for every position, fetched in parallel (no cache)."""

    def one(rec):
        ticker, expiry, cp, strike, qty = rec
        occ = build_occ(ticker, expiry, cp, strike)
        snap = None
        try:
            snap = contract_snapshot(ticker, occ)
        except PolygonError as err:
            logger.error("[%s] position fetch failed: %r", occ, err)
        row = {
            "ticker": ticker, "expiry": expiry, "type": cp,
            "strike": strike, "qty": qty,
        }
        row.update({c: (snap or {}).get(c) for c in QUOTE_COLS})
        row["underlying_price"] = (snap or {}).get("underlying_price")
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, records))
    return pd.DataFrame(rows)


def _load_positions() -> pd.DataFrame:
    if POSITIONS_CSV.exists():
        return pd.read_csv(POSITIONS_CSV)
    return pd.DataFrame(columns=["ticker", "expiry", "type", "strike", "qty"])


def _stamp() -> str:
    return f"{datetime.now(timezone.utc):%H:%M:%S} UTC"


def render() -> None:
    # ---- Controls ----------------------------------------------------------
    c_live, c_btn, _ = st.columns([2, 2, 4])
    live_on = c_live.toggle(f"Live quotes (every {LIVE_INTERVAL})", value=True)
    if c_btn.button("🔄 Refresh now", type="primary", use_container_width=True):
        _chain.clear()
        st.rerun()
    run_every = LIVE_INTERVAL if live_on else None

    # ---- Positions ---------------------------------------------------------
    st.subheader("Positions")
    if "positions_df" not in st.session_state:
        st.session_state["positions_df"] = _load_positions()

    with st.expander("Edit positions"):
        st.caption(
            "Session-only edits — to make positions permanent, commit them to "
            "`positions.csv` in the repo."
        )
        edited = st.data_editor(
            st.session_state["positions_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "expiry": st.column_config.TextColumn(
                    "Expiry", help="YYYY-MM-DD", required=True
                ),
                "type": st.column_config.SelectboxColumn(
                    "Type", options=["call", "put"], required=True
                ),
                "strike": st.column_config.NumberColumn("Strike", required=True),
                "qty": st.column_config.NumberColumn("Qty"),
            },
            key="positions_editor",
        )
        st.session_state["positions_df"] = edited

    pos = st.session_state["positions_df"].dropna(subset=["ticker", "expiry", "strike"])
    if pos.empty:
        st.info("No positions yet — add rows under **Edit positions**.")
    else:
        records = tuple(
            (str(r.ticker).strip().upper(), str(r.expiry).strip(),
             str(r.type).strip().lower(), float(r.strike),
             float(r.qty) if pd.notna(r.qty) else None)
            for r in pos.itertuples(index=False)
        )

        def _positions_body() -> None:
            try:
                live = _fetch_positions(records)
            except Exception as err:
                logger.error("Positions fetch failed: %r", err)
                st.error(f"Couldn't fetch position quotes: {err}")
                return
            st.caption(f"Quotes as of **{_stamp()}**" + (" — live" if live_on else ""))
            st.dataframe(
                live.drop(columns=["underlying_price"]),
                use_container_width=True, hide_index=True,
                column_config=_NUM_FMT,
            )

            # Delta-dollars rollup per underlying: qty x 100 x delta x spot.
            dd = live.dropna(subset=["qty", "delta", "underlying_price"]).copy()
            if not dd.empty:
                dd["delta_dollars"] = dd["qty"] * 100 * dd["delta"] * dd["underlying_price"]
                roll = (
                    dd.groupby("ticker")
                    .agg(spot=("underlying_price", "last"), delta_dollars=("delta_dollars", "sum"))
                    .reset_index()
                )
                total = pd.DataFrame(
                    [{"ticker": "TOTAL", "spot": None, "delta_dollars": roll["delta_dollars"].sum()}]
                )
                st.caption("Delta $ by underlying")
                st.dataframe(
                    pd.concat([roll, total], ignore_index=True),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "spot": st.column_config.NumberColumn("Spot", format="%.2f"),
                        "delta_dollars": st.column_config.NumberColumn(
                            "Delta $", format="accounting"
                        ),
                    },
                )

        st.fragment(_positions_body, run_every=run_every)()

    st.divider()

    # ---- Lookup ------------------------------------------------------------
    st.subheader("Option lookup")
    c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
    ticker = c1.text_input("Ticker", placeholder="e.g. NVDA", key="opt_ticker")
    cp = c2.segmented_control(
        "Type", options=["Call", "Put"], default="Call", key="opt_cp"
    )
    strike_near = c4.number_input(
        "Strike (optional)", value=None, min_value=0.0, step=1.0,
        key="opt_strike", help="Show only the 5 strikes above and below this",
    )
    ticker = (ticker or "").strip().upper()
    if not ticker:
        st.caption("Type a ticker to load its expiries.")
        return

    try:
        expiries = _expirations(ticker)
    except PolygonError as err:
        logger.error("[%s] expiry fetch failed: %r", ticker, err)
        st.error(f"Couldn't load expiries for {ticker}: {err}")
        return
    if not expiries:
        st.warning(f"No listed options found for {ticker}.")
        return

    expiry = c3.selectbox("Expiry", expiries, key="opt_expiry")

    def _chain_body() -> None:
        try:
            chain = _chain(ticker, expiry, (cp or "Call").lower())
        except PolygonError as err:
            logger.error("[%s] chain fetch failed: %r", ticker, err)
            st.error(f"Couldn't fetch the {ticker} chain: {err}")
            return
        if chain.empty:
            st.warning(f"No {(cp or 'call').lower()}s listed for {ticker} {expiry}.")
            return
        show = chain[["strike"] + QUOTE_COLS + ["open_interest", "volume"]]
        show = show.sort_values("strike")
        scope = f"{len(show)} strikes"
        if strike_near:
            below = show[show["strike"] < strike_near].tail(5)
            at = show[show["strike"] == strike_near]
            above = show[show["strike"] > strike_near].head(5)
            show = pd.concat([below, at, above])
            scope = f"{len(show)} strikes around {strike_near:g}"
        st.caption(
            f"{ticker} {expiry} {(cp or 'call').lower()}s — {scope}, "
            f"as of **{_stamp()}**" + (" — live" if live_on else "")
        )
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config={
                **_NUM_FMT,
                "strike": st.column_config.NumberColumn("Strike", format="%.2f", pinned=True),
            },
            height=min(38 + 35 * len(show), 600),
        )

    st.fragment(_chain_body, run_every=run_every)()
