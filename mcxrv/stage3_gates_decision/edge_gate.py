"""
Cost hurdle: does a signal's expected profit beat the cost of trading it?

One function, used by the backtest at entry and by the live board, so an alert and
a simulated trade are decided by the same rule.

Expected gross, in bp of leg A's notional (the unit the backtest reports):

    exp_gross_bp = (|z| - exit_z) * scale_bp * (1 - 0.5 ** (H / halflife)) * haircut

  (|z| - exit_z) * scale_bp    the distance from today's gap to the exit band, in bp
  1 - 0.5 ** (H / halflife)    the share of it a mean-reverting gap closes in H
                               sessions if it decays at its own measured half-life
  H                            sessions the trade can stay open: the holding limit,
                               or the runway to the forced exit before expiry
  haircut                      an allowance for gaps that close less than their
                               history says; fixed up front, never fitted

Cost is the entry cost plus an estimated exit cost (same spread estimates, wider on
legs already untradable), in bp of leg A's notional.

  ok  <=>  exp_gross_bp >= hurdle_mult * cost_bp   and   exp_net_bp >= min_net_bp

The three settings (haircut 0.7, hurdle 1.5x, 5 bp) were fixed before any result was
seen and are not tuned to the data: changing them after looking would turn the
backtest into a fitting exercise. See docs/EXPERIMENTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeGateConfig:
    exit_z: float = 0.5
    max_holding_sessions: int = 40
    min_sessions_to_expiry_exit: int = 4
    hurdle_mult: float = 1.5
    min_net_bp: float = 5.0
    haircut: float = 0.7


def edge_check(z, scale_bp, halflife, sessions_to_expiry_a, sessions_to_expiry_b,
               entry_cost_inr, est_exit_cost_inr, notional_a_inr,
               cfg: EdgeGateConfig | None = None) -> dict:
    """Expected gross, cost and net in bp, whether the hurdle is cleared, and why."""
    cfg = cfg or EdgeGateConfig()
    out = {"exp_gross_bp": np.nan, "cost_bp": np.nan, "exp_net_bp": np.nan,
           "ok": False, "reason": ""}
    try:
        z, scale, hl = float(z), float(scale_bp), float(halflife)
        notional = float(notional_a_inr)
        ste = min(float(sessions_to_expiry_a), float(sessions_to_expiry_b))
    except (TypeError, ValueError):
        out["reason"] = "cost hurdle: inputs missing"
        return out
    if not (np.isfinite(notional) and notional > 0):
        out["reason"] = "cost hurdle: no notional to price"
        return out
    cost_bp = (float(entry_cost_inr) + float(est_exit_cost_inr)) / notional * 1e4
    out["cost_bp"] = cost_bp
    if not all(np.isfinite(v) for v in (z, scale, hl, ste)) or hl <= 0 or scale <= 0:
        out["reason"] = "cost hurdle: no half-life or scale to project from"
        return out

    horizon = min(cfg.max_holding_sessions, ste - cfg.min_sessions_to_expiry_exit)
    if horizon <= 0:
        out["exp_gross_bp"], out["exp_net_bp"] = 0.0, -cost_bp
        out["reason"] = "cost hurdle: no time left before the forced exit"
        return out

    distance = max(abs(z) - cfg.exit_z, 0.0) * scale
    closes = 1.0 - 0.5 ** (horizon / hl)
    gross = distance * closes * cfg.haircut
    net = gross - cost_bp
    ok = bool(gross >= cfg.hurdle_mult * cost_bp and net >= cfg.min_net_bp)
    out.update(exp_gross_bp=gross, exp_net_bp=net, ok=ok)
    out["reason"] = ("expected {:.1f} bp vs cost {:.1f} bp: net {:+.1f} bp{}".format(
        gross, cost_bp, net,
        "" if ok else " (needs {:.1f}x cost and {:+.0f} bp net)".format(cfg.hurdle_mult, cfg.min_net_bp)))
    return out
