"""
Bid/ask spread estimated from daily high, low and close, as a check on the model.

The bhavcopy has no bid or ask, so the cost model assumes a half-spread from tick
size and volume (pair_match.annotate). These two published estimators instead read
the spread out of each contract's own prices:

  Corwin & Schultz (2012). A day's high is usually a trade at the ask and its low a
  trade at the bid, so the high-low range mixes volatility with the spread. Over two
  days volatility doubles but the spread does not, which separates them. Overnight
  gaps are removed first. Daily estimates are returned RAW (often negative) and
  must be averaged before a negative result is set to zero: flooring each day
  first, the other option in the paper, inflates the average several-fold when
  daily volatility is large next to the spread, as it is for gold (tested).

  Abdi & Ranaldo (2017). The close sits on one side of the spread; the midpoint of
  the day's high and low approximates the efficient price. The covariance of the
  close's distance from today's and tomorrow's midpoints is spread^2 / 4.

Both are noisy for one day and meant to be averaged over weeks per contract. Both
estimate the FULL quoted spread; the functions return the HALF-spread in bp, the
unit the cost model uses. They need genuine trading: a settlement MCX computed on a
day nothing traded has no bid/ask in it, so such rows are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def _usable(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows.copy()
    ok = (d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0) & (d["high"] >= d["low"])
    for flag in ("settle_untradeable", "ohlc_bad", "no_trade"):
        if flag in d.columns:
            ok &= ~d[flag].fillna(False).astype(bool)
    return d[ok].sort_values(["contract_id", "date"])


def corwin_schultz(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Two-day Corwin-Schultz half-spread (bp) per contract and session.

    Each estimate uses session t and the contract's next usable session; a pair
    with a gap of more than one session in between is skipped. Returns
    contract_id, date (the first day of the pair), cs_half_spread_bp, RAW: average
    it with `average` before reading it as a spread.
    """
    d = _usable(rows)
    out = []
    for cid, g in d.groupby("contract_id", sort=False):
        if len(g) < 2:
            continue
        h, lo, c = (np.log(g[k].to_numpy(float)) for k in ("high", "low", "close"))
        h1, l1 = h[1:].copy(), lo[1:].copy()
        # Overnight adjustment: shift the second day's range by any gap from the
        # first day's close, so a jump between sessions is not read as spread.
        up = l1 - c[:-1]
        down = c[:-1] - h1
        shift = np.where(up > 0, -up, np.where(down > 0, down, 0.0))
        h1, l1 = h1 + shift, l1 + shift
        beta = (h[:-1] - lo[:-1]) ** 2 + (h1 - l1) ** 2
        gamma = (np.maximum(h[:-1], h1) - np.minimum(lo[:-1], l1)) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        if "sessions_to_expiry" in g.columns:           # consecutive sessions only
            step = -np.diff(g["sessions_to_expiry"].to_numpy(float))
            s = np.where(np.isfinite(step) & (step <= 1.5), s, np.nan)
        out.append(pd.DataFrame({"contract_id": cid, "date": g["date"].to_numpy()[:-1],
                                 "cs_half_spread_bp": s / 2 * 1e4}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["contract_id", "date", "cs_half_spread_bp"]))


def abdi_ranaldo(rows: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Abdi-Ranaldo half-spread (bp) per contract and session, from a trailing
    `window` of sessions. Returns contract_id, date, ar_half_spread_bp.
    """
    d = _usable(rows)
    out = []
    for cid, g in d.groupby("contract_id", sort=False):
        if len(g) < 3:
            continue
        h, lo, c = (np.log(g[k].to_numpy(float)) for k in ("high", "low", "close"))
        eta = (h + lo) / 2
        prod = np.full(len(g), np.nan)
        prod[:-1] = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
        s2 = 4 * pd.Series(prod).rolling(window, min_periods=max(5, window // 2)).mean()
        s = np.sqrt(np.clip(s2.to_numpy(), 0.0, None))
        out.append(pd.DataFrame({"contract_id": cid, "date": g["date"].to_numpy(),
                                 "ar_half_spread_bp": s / 2 * 1e4}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["contract_id", "date", "ar_half_spread_bp"]))


def average(values) -> float:
    """Mean of raw daily estimates, floored at zero AFTER averaging."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(max(v.mean(), 0.0)) if len(v) else float("nan")
