"""
Transaction costs for a two-leg spread trade.

The arithmetic that kills most versions of this strategy: a spread position is
TWO legs, each opened and closed, so a round trip crosses FOUR spreads. Costing
one crossing, or netting the legs, understates friction by roughly 4x.

Statutory charges are small. On MCX non-agri futures they total single-digit basis
points. The cost that actually decides whether this strategy exists is the
bid-ask spread and market impact on the thin legs -- and bhavcopy contains no
depth data whatsoever, so that number is an ASSUMPTION.

The correct response to an unobservable dominant parameter is not to pick a
flattering value. It is to sweep it and report where the strategy dies. That is
what `sensitivity_grid` is for, and a breakeven cost is a more honest headline
than any single Sharpe ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    """
    Statutory rates are fractions of turnover unless stated. Defaults are
    plausible retail-through-broker values for MCX non-agri futures; they are
    parameters to be set by whoever runs this, not claims of fact.
    """
    brokerage_rate: float = 0.00020        # 2 bp of turnover
    brokerage_min_inr: float = 20.0        # per order
    exchange_txn_rate: float = 0.000026    # ~0.0026%
    sebi_turnover_rate: float = 0.000001   # Rs 10 per crore
    ctt_rate_sell: float = 0.0001          # 0.01%, SELL side only
    stamp_duty_rate_buy: float = 0.00002   # 0.002%, BUY side only
    gst_rate: float = 0.18                 # on brokerage + exchange charges

    # Execution assumptions -- the dominant and unobservable part.
    half_spread_bp_override: float | None = None  # None: use per-row estimate
    impact_coef_bp: float = 8.0            # impact = coef * sqrt(participation)
    slippage_bp: float = 0.0               # extra pad, for stress testing


def statutory_cost_inr(turnover_inr: float, side: str, cfg: CostConfig) -> float:
    """
    Statutory and brokerage charges for ONE leg, ONE side, in rupees.

    CTT applies to the sell side only and stamp duty to the buy side only, so a
    round trip pays each exactly once per leg regardless of direction.
    """
    turnover_inr = max(float(turnover_inr), 0.0)
    brokerage = max(turnover_inr * cfg.brokerage_rate, cfg.brokerage_min_inr)
    exch = turnover_inr * cfg.exchange_txn_rate
    sebi = turnover_inr * cfg.sebi_turnover_rate
    gst = (brokerage + exch) * cfg.gst_rate
    ctt = turnover_inr * cfg.ctt_rate_sell if side.lower() == "sell" else 0.0
    stamp = turnover_inr * cfg.stamp_duty_rate_buy if side.lower() == "buy" else 0.0
    return brokerage + exch + sebi + gst + ctt + stamp


def execution_cost_bp(half_spread_bp: float, lots: float, max_lots: float,
                      cfg: CostConfig) -> float:
    """
    Execution cost for ONE crossing, in bp of notional.

    Half-spread plus a square-root impact term in participation rate. Square-root
    impact is the standard empirical form; the coefficient is an assumption.
    """
    hs = (cfg.half_spread_bp_override if cfg.half_spread_bp_override is not None
          else half_spread_bp)
    hs = float(np.nan_to_num(hs, nan=0.0))
    part = 0.0 if max_lots <= 0 else min(float(lots) / float(max_lots), 1.0)
    impact = cfg.impact_coef_bp * np.sqrt(part)
    return hs + impact + cfg.slippage_bp


def leg_cost_inr(turnover_inr: float, side: str, half_spread_bp: float,
                 lots: float, max_lots: float, cfg: CostConfig) -> dict:
    """Full cost of one crossing of one leg, itemised."""
    stat = statutory_cost_inr(turnover_inr, side, cfg)
    exec_bp = execution_cost_bp(half_spread_bp, lots, max_lots, cfg)
    exec_inr = turnover_inr * exec_bp / 10000.0
    return {"statutory_inr": stat, "execution_inr": exec_inr,
            "execution_bp": exec_bp, "total_inr": stat + exec_inr}


def round_trip_bp(turnover_per_leg_inr: float, half_spread_a_bp: float,
                  half_spread_b_bp: float, cfg: CostConfig,
                  lots_a: float = 1.0, max_lots_a: float = 1e9,
                  lots_b: float = 1.0, max_lots_b: float = 1e9) -> dict:
    """
    Total round-trip cost of a two-leg spread, in bp of ONE leg's notional.

    Four crossings: open A, open B, close A, close B. This function exists mostly
    so the 4x is impossible to forget.
    """
    legs = [
        leg_cost_inr(turnover_per_leg_inr, "buy", half_spread_a_bp, lots_a, max_lots_a, cfg),
        leg_cost_inr(turnover_per_leg_inr, "sell", half_spread_b_bp, lots_b, max_lots_b, cfg),
        leg_cost_inr(turnover_per_leg_inr, "sell", half_spread_a_bp, lots_a, max_lots_a, cfg),
        leg_cost_inr(turnover_per_leg_inr, "buy", half_spread_b_bp, lots_b, max_lots_b, cfg),
    ]
    total_inr = sum(l["total_inr"] for l in legs)
    stat_inr = sum(l["statutory_inr"] for l in legs)
    exec_inr = sum(l["execution_inr"] for l in legs)
    denom = max(turnover_per_leg_inr, 1e-9)
    return {
        "total_inr": total_inr,
        "total_bp": total_inr / denom * 10000.0,
        "statutory_bp": stat_inr / denom * 10000.0,
        "execution_bp": exec_inr / denom * 10000.0,
        "n_crossings": 4,
    }


def cost_summary_table(notional_inr: float, cfg: CostConfig | None = None,
                       half_spreads=(2.0, 5.0, 10.0, 20.0, 40.0)) -> pd.DataFrame:
    """
    Round-trip cost against the assumed half-spread -- the parameter we cannot
    observe. Read alongside the typical dislocation size to see immediately
    whether any of this is worth trading.
    """
    cfg = cfg or CostConfig()
    rows = []
    for hs in half_spreads:
        r = round_trip_bp(notional_inr, hs, hs, cfg)
        rows.append({
            "half_spread_bp_per_leg": hs,
            "statutory_bp": round(r["statutory_bp"], 2),
            "execution_bp": round(r["execution_bp"], 2),
            "round_trip_bp": round(r["total_bp"], 2),
            "breakeven_dislocation_bp": round(r["total_bp"], 2),
        })
    return pd.DataFrame(rows)


def breakeven_half_spread_bp(gross_edge_bp: float, notional_inr: float,
                             cfg: CostConfig | None = None,
                             lo: float = 0.0, hi: float = 500.0) -> float:
    """
    The per-leg half-spread at which a given gross edge exactly breaks even.

    The single most useful summary statistic in this project: it converts "is
    there an edge?" into "how tight would the book have to be?", which a trader
    can check against reality in seconds.
    """
    cfg = cfg or CostConfig()

    def net(hs):
        return gross_edge_bp - round_trip_bp(notional_inr, hs, hs, cfg)["total_bp"]

    if net(lo) <= 0:
        return 0.0
    if net(hi) > 0:
        return float("inf")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
