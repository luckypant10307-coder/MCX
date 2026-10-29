"""
Tradability: whether a published settlement could plausibly have been a fill, and
how big a position the tape could have absorbed.

Two distinctions this module refuses to blur:

  volume is not depth. A day with 500 GOLDPETAL lots traded is 500 GRAMS of gold
  changing hands across the whole session. It says nothing about how much was
  available at any instant, or at what price. Bhavcopy contains no depth data at
  all, so every depth number here is an ASSUMPTION, labelled as one, and swept in
  the cost sensitivity analysis rather than asserted.

  a settlement price is not an executable price. It is an official exchange value
  computed from a closing window, and on a no-trade day it is derived outright.
  Filling a backtest at a derived settlement means backtesting the exchange's own
  valuation model, which is smooth and mean-reverting by construction -- it will
  manufacture a beautiful, entirely fictional equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LiquidityConfig:
    min_volume_lots: dict = field(default_factory=lambda: {
        "GOLDM": 200.0, "GOLDTEN": 40.0, "GOLDGUINEA": 25.0, "GOLDPETAL": 20.0,
    })
    min_oi_lots: dict = field(default_factory=lambda: {
        "GOLDM": 500.0, "GOLDTEN": 100.0, "GOLDGUINEA": 60.0, "GOLDPETAL": 40.0,
    })
    # Fraction of a session's volume we assume we could take without moving price.
    participation_cap: float = 0.05
    min_sessions_to_expiry: int = 5
    # Half-spread model: half_spread_bp = max(half_spread_ticks * tick_bp,
    #                                        illiquidity_coef / sqrt(volume))
    half_spread_ticks: float = 1.0
    illiquidity_coef: float = 60.0
    # Fit-error contribution to the measurement noise floor.
    fit_rmse_weight: float = 1.0


def annotate(rows: pd.DataFrame, daily: pd.DataFrame,
             cfg: LiquidityConfig | None = None) -> pd.DataFrame:
    """
    Attach tradability, an assumed half-spread, and a measurement noise floor.

    Adds:
      tradable          all hard gates passed: genuine trade, volume, OI, lifecycle
      half_spread_bp    assumed one-way crossing cost, in bp of normalised price
      max_lots          participation-capped size the session could absorb
      noise_bp          measurement noise floor: tick quantisation + curve fit
                        error. A dislocation smaller than this is not information,
                        and the signal gate compares against it.
    """
    cfg = cfg or LiquidityConfig()
    d = rows.copy()

    minvol = d["symbol"].map(lambda s: cfg.min_volume_lots.get(s, 10.0))
    minoi = d["symbol"].map(lambda s: cfg.min_oi_lots.get(s, 20.0))

    d["gate_genuine_trade"] = ~d["settle_untradeable"]
    d["gate_volume"] = d["volume"] >= minvol
    d["gate_oi"] = d["open_interest"] >= minoi
    d["gate_lifecycle"] = (
        (d["sessions_to_expiry"] >= cfg.min_sessions_to_expiry) & (~d["in_tender"])
    )
    d["tradable"] = (d["gate_genuine_trade"] & d["gate_volume"]
                     & d["gate_oi"] & d["gate_lifecycle"])

    vol = d["volume"].clip(lower=1.0).astype(float)
    spread_from_ticks = cfg.half_spread_ticks * d["tick_bp_of_norm"].astype(float)
    spread_from_thin = cfg.illiquidity_coef / np.sqrt(vol)
    d["half_spread_bp"] = np.fmax(spread_from_ticks, spread_from_thin)

    d["max_lots"] = np.floor(vol * cfg.participation_cap).clip(lower=0)

    rmse = daily.set_index("date")["fit_rmse"].reindex(d["date"]).to_numpy()
    rmse_bp = np.nan_to_num(rmse * 10000.0, nan=0.0)
    # Quadrature: tick quantisation and fit error are independent error sources.
    d["noise_bp"] = np.sqrt(
        (d["tick_bp_of_norm"].astype(float) / np.sqrt(12.0)) ** 2
        + (cfg.fit_rmse_weight * rmse_bp) ** 2
    )
    return d


def symbol_noise_floor(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol) noise floor, for the signal gate to compare against."""
    r = rows[rows["tradable"]] if "tradable" in rows.columns else rows
    if r.empty:
        return pd.DataFrame(columns=["date", "symbol", "noise_bp"])
    return (r.groupby(["date", "symbol"])["noise_bp"].min()
            .reset_index().sort_values(["date", "symbol"]))


def tradable_universe(rows: pd.DataFrame) -> pd.DataFrame:
    """
    The single best tradable contract per (date, symbol): most volume wins.

    Chosen by volume rather than by nearness to expiry, because the leg we would
    actually execute is the one with a tape, and the front month is not always it.
    """
    t = rows[rows["tradable"]]
    if t.empty:
        return t
    idx = t.groupby(["date", "symbol"])["volume"].idxmax()
    return rows.loc[idx].sort_values(["date", "symbol"]).reset_index(drop=True)


def coverage_report(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol tradability audit, and which gate does the binding.

    Worth printing in any writeup: it quantifies how much of the nominal dataset
    is actually usable, which for the small contracts is the whole story.
    """
    gates = ["gate_genuine_trade", "gate_volume", "gate_oi", "gate_lifecycle",
             "tradable"]
    out = []
    for sym, g in rows.groupby("symbol"):
        rec = {"symbol": sym, "rows": len(g),
               "sessions": int(g["date"].nunique())}
        for gate in gates:
            if gate in g.columns:
                rec[gate + "_rate"] = round(float(g[gate].mean()), 4)
        td = g[g["tradable"]] if "tradable" in g.columns else g.iloc[:0]
        rec["tradable_sessions"] = int(td["date"].nunique())
        rec["median_half_spread_bp"] = (round(float(td["half_spread_bp"].median()), 2)
                                        if len(td) else np.nan)
        rec["median_max_lots"] = (float(td["max_lots"].median())
                                  if len(td) else np.nan)
        out.append(rec)
    return pd.DataFrame(out).sort_values("symbol").reset_index(drop=True)
