"""Black-Scholes IV + greeks — the fallback for contracts Polygon can't solve.

Polygon returns `implied_volatility` ~0 (and greeks to match) for contracts
whose quote its solver won't converge on. That is not a usable number, but the
contract still has a real two-sided market, so the greeks can be recovered
locally from the mid.

Conventions, chosen to match the scale vendors report and the table's %.4f
columns: **vega is per 1 vol point (0.01)** and **theta is per calendar day**.
Delta and gamma are scale-free. Rates are flat assumptions — this exists to
put a usable delta on screen, not to reprice the book.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365.0 * 24 * 3600

RISK_FREE_RATE = float(os.environ.get("BS_RISK_FREE_RATE", "0.04"))
DIVIDEND_YIELD = float(os.environ.get("BS_DIVIDEND_YIELD", "0.0"))

# Solver bounds. The lower bound sits under any real equity-option IV so a
# genuinely cheap contract still converges instead of pinning to the floor.
IV_LO = 1e-4
IV_HI = 10.0
_MAX_ITER = 100
_TOL = 1e-8


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def year_fraction(expiry_iso: str, now: datetime | None = None) -> float | None:
    """Years until the 16:00 ET close on `expiry_iso`; None once expired."""
    try:
        y, m, d = (int(p) for p in str(expiry_iso).split("-"))
        expiry = datetime(y, m, d, 16, 0, tzinfo=NY)
    except (AttributeError, TypeError, ValueError):
        return None
    seconds = (expiry - (now or datetime.now(NY))).total_seconds()
    return seconds / SECONDS_PER_YEAR if seconds > 0 else None


def price(
    S: float, K: float, T: float, sigma: float, is_call: bool,
    r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD,
) -> float:
    """Black-Scholes-Merton price with continuous dividend yield."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = (S - K) if is_call else (K - S)
        return max(intrinsic, 0.0)
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    d2 = d1 - vol_t
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if is_call:
        return disc_s * _norm_cdf(d1) - disc_k * _norm_cdf(d2)
    return disc_k * _norm_cdf(-d2) - disc_s * _norm_cdf(-d1)


def greeks(
    S: float, K: float, T: float, sigma: float, is_call: bool,
    r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD,
) -> dict[str, float] | None:
    """Delta, gamma, vega (per vol point), theta (per calendar day)."""
    if not (T > 0 and sigma > 0 and S > 0 and K > 0):
        return None
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    d2 = d1 - vol_t
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    delta = disc_q * (_norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0)
    gamma = disc_q * pdf_d1 / (S * vol_t)
    vega = S * disc_q * pdf_d1 * math.sqrt(T)
    decay = -(S * disc_q * pdf_d1 * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        theta = decay - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
    else:
        theta = decay + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,
        "theta": theta / 365.0,
    }


def implied_vol(
    option_price: float, S: float, K: float, T: float, is_call: bool,
    r: float = RISK_FREE_RATE, q: float = DIVIDEND_YIELD,
) -> float | None:
    """Solve IV from a price by bisection; None if no arbitrage-free solution.

    Bisection rather than Newton: vega collapses on the deep-OTM contracts
    this fallback is for, which is exactly where Newton diverges.
    """
    try:
        option_price, S, K, T = float(option_price), float(S), float(K), float(T)
    except (TypeError, ValueError):
        return None
    if not (option_price > 0 and S > 0 and K > 0 and T > 0):
        return None

    # Outside the no-arbitrage band there is no IV to find — most often a
    # stale quote sitting under intrinsic.
    lo_price = price(S, K, T, IV_LO, is_call, r, q)
    hi_price = price(S, K, T, IV_HI, is_call, r, q)
    if option_price <= lo_price or option_price >= hi_price:
        return None

    # Bisect to interval width, not to price difference: on a deep-OTM
    # contract the price surface is flat to within 1e-8 across a wide band of
    # vols, so a price-based exit converges on an arbitrary point in it.
    lo, hi = IV_LO, IV_HI
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if (hi - lo) < _TOL:
            return mid
        if price(S, K, T, mid, is_call, r, q) > option_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve(
    bid: float | None, ask: float | None, S: float | None, K: float | None,
    expiry_iso: str, is_call: bool, now: datetime | None = None,
) -> dict[str, float] | None:
    """IV + greeks from a two-sided quote; None if the inputs can't support it."""
    try:
        mid = (float(bid) + float(ask)) / 2.0
    except (TypeError, ValueError):
        return None
    T = year_fraction(expiry_iso, now)
    if T is None or S is None or K is None or mid <= 0:
        return None
    iv = implied_vol(mid, S, K, T, is_call)
    if iv is None:
        return None
    g = greeks(S, K, T, iv, is_call)
    return {"ivm": iv, **g} if g else None
