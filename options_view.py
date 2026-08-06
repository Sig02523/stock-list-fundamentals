"""Options tab: live-ticking positions/greeks + chain lookup (Polygon)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from polygon_options import (
    PolygonError,
    build_occ,
    chain_snapshot,
    contract_snapshot,
    expiries_with_strike,
    list_expirations,
    normalize_underlying,
    stock_snapshots,
)

from fmp_client import FMPClient, FMPError

logger = logging.getLogger("stocklist")
POSITIONS_CSV = Path(__file__).resolve().parent / "positions.csv"


@st.cache_resource
def _fmp() -> FMPClient:
    return FMPClient(delay=0)


def _underlying_quotes(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Real-time spot + day change per underlying from FMP (the Polygon
    options-only plan serves 15-min-delayed stock trades — do not use it for
    spot). FMP symbology uses dashes for class shares: BRK.B -> BRK-B."""

    def one(t: str) -> tuple[str, dict]:
        try:
            q = _fmp().quote(t.replace(".", "-"))
            return t, {
                "price": q.get("price"),
                "chg": q.get("change"),
                "chg_pct": q.get("changePercentage"),
            }
        except FMPError as err:
            logger.warning("[%s] FMP quote failed: %r", t, err)
            return t, {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, tickers))

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


@st.cache_data(ttl=3600, show_spinner="Loading expiries...")
def _expirations(ticker: str) -> list[str]:
    return list_expirations(ticker)


@st.cache_data(ttl=4, show_spinner=False)
def _chain(ticker: str, expiry: str, cp: str) -> pd.DataFrame:
    return pd.DataFrame(chain_snapshot(ticker, expiry, cp))


@st.cache_data(ttl=3600, show_spinner=False)
def _expiries_with_strike(ticker: str, strike: float, direction: str) -> list[str]:
    try:
        return expiries_with_strike(ticker, strike, direction)
    except PolygonError:
        return []


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
    return f"{datetime.now(ZoneInfo('America/New_York')):%H:%M:%S} ET"


def _color_signed(val: object) -> str:
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    if v > 0:
        return "color: #1a7f37; font-weight: 600;"
    if v < 0:
        return "color: #cf222e; font-weight: 600;"
    return ""


_POS_LABELS = {
    "ticker": "Ticker", "expiry": "Expiry", "type": "Type", "strike": "Strike",
    "qty": "Qty", "spot": "Spot", "chg": "Chg $", "chg_pct": "Chg %",
    "bid": "Bid", "ask": "Offer", "ivm": "IVM", "delta": "Delta",
    "gamma": "Gamma", "vega": "Vega", "theta": "Theta",
}
_POS_FMT = {
    "Strike": "{:g}", "Qty": "{:g}", "Spot": "{:,.2f}",
    "Chg $": "{:+.2f}", "Chg %": "{:+.2f}%",
    "Bid": "{:.2f}", "Offer": "{:.2f}", "IVM": "{:.1%}",
    "Delta": "{:.4f}", "Gamma": "{:.4f}", "Vega": "{:.4f}", "Theta": "{:.4f}",
}


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
            (normalize_underlying(str(r.ticker)), str(r.expiry).strip(),
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
            tickers_u = tuple(sorted(live["ticker"].unique()))
            und = _underlying_quotes(tickers_u)
            if not any(v.get("price") for v in und.values()):
                # FMP down — fall back to Polygon's (delayed) stocks snapshot.
                try:
                    und = stock_snapshots(list(tickers_u))
                except PolygonError as err:
                    logger.warning("underlying snapshot fetch failed: %r", err)

            # Underlying spot + day change on every position row.
            live["spot"] = live["ticker"].map(
                lambda t: und.get(t, {}).get("price")
            ).fillna(live["underlying_price"])
            live["chg"] = live["ticker"].map(lambda t: und.get(t, {}).get("chg"))
            live["chg_pct"] = live["ticker"].map(lambda t: und.get(t, {}).get("chg_pct"))

            pos_cols = (
                ["ticker", "expiry", "type", "strike", "qty", "spot", "chg", "chg_pct"]
                + QUOTE_COLS
            )
            st.caption(f"Quotes as of **{_stamp()}**" + (" — live" if live_on else ""))
            disp = live[pos_cols].rename(columns=_POS_LABELS)
            st.dataframe(
                disp.style.format(_POS_FMT, na_rep="—")
                .map(_color_signed, subset=["Chg $", "Chg %"]),
                use_container_width=True, hide_index=True,
            )

            dd = live.dropna(subset=["qty", "delta", "spot"]).copy()
            dd["delta_dollars"] = dd["qty"] * 100 * dd["delta"] * dd["spot"]
            dd_by_ticker = dd.groupby("ticker")["delta_dollars"].sum()

            rows = []
            for t in sorted(live["ticker"].unique()):
                u = und.get(t, {})
                spot = u.get("price")
                if spot is None:
                    opt_spot = live.loc[live["ticker"] == t, "underlying_price"].dropna()
                    spot = opt_spot.iloc[0] if not opt_spot.empty else None
                rows.append({
                    "ticker": t, "spot": spot,
                    "chg": u.get("chg"), "chg_pct": u.get("chg_pct"),
                    "delta_dollars": dd_by_ticker.get(t),
                })
            rows.append({
                "ticker": "TOTAL", "spot": None, "chg": None, "chg_pct": None,
                "delta_dollars": dd_by_ticker.sum(),
            })
            st.caption("By underlying")
            roll_disp = pd.DataFrame(rows).rename(
                columns={**_POS_LABELS, "delta_dollars": "Delta $"}
            )
            st.dataframe(
                roll_disp.style.format(
                    {"Spot": "{:,.2f}", "Chg $": "{:+.2f}", "Chg %": "{:+.2f}%",
                     "Delta $": "{:,.0f}"},
                    na_rep="—",
                )
                .map(_color_signed, subset=["Chg $", "Chg %"]),
                use_container_width=True, hide_index=True,
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
    ticker = normalize_underlying(ticker or "")
    if not ticker:
        st.caption("Type a ticker (press Enter) to load its expiries.")
        return

    try:
        expiries = _expirations(ticker)
        if not expiries and ticker.isalpha() and len(ticker) > 2:
            # Dotless class-share input (BRKB) -> retry as BRK.B
            alt = f"{ticker[:-1]}.{ticker[-1]}"
            if _expirations(alt):
                ticker = alt
                expiries = _expirations(alt)
                st.caption(f"Interpreting as **{ticker}**.")
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
        spot = chain["underlying_price"].dropna()
        spot_txt = f", spot {spot.iloc[0]:,.2f}" if not spot.empty else ""
        show = chain[["strike"] + QUOTE_COLS + ["open_interest", "volume"]]
        show = show.sort_values("strike")
        scope = f"{len(show)} strikes"
        if strike_near:
            lo, hi = show["strike"].min(), show["strike"].max()
            if strike_near > hi or strike_near < lo:
                side = "above the highest" if strike_near > hi else "below the lowest"
                bound = hi if strike_near > hi else lo
                msg = (
                    f"{strike_near:g} is {side} listed strike ({bound:g}) for "
                    f"**this expiry** ({expiry})."
                )
                alt = _expiries_with_strike(
                    ticker, strike_near, "gte" if strike_near > hi else "lte"
                )
                alt = [d for d in alt if d != expiry]
                if alt:
                    msg += (
                        f" Strikes {'≥' if strike_near > hi else '≤'} "
                        f"{strike_near:g} are listed from **{alt[0]}** onward — "
                        "pick a later expiry in the dropdown."
                    )
                else:
                    msg += f" No {ticker} expiry lists a strike that far out."
                st.warning(msg + " Showing the nearest edge of this chain.")
            below = show[show["strike"] < strike_near].tail(5)
            at = show[show["strike"] == strike_near]
            above = show[show["strike"] > strike_near].head(5)
            show = pd.concat([below, at, above])
            scope = f"{len(show)} strikes around {strike_near:g}"
        st.caption(
            f"{ticker} {expiry} {(cp or 'call').lower()}s — {scope}{spot_txt}, "
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
