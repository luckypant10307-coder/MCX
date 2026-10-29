"""
Per-pair margin and capital ledger, and margin-call simulation.

WHY A LEDGER RATHER THAN A SINGLE MARGIN NUMBER
-----------------------------------------------
Initial margin at entry says what it cost to open. It does not say whether the
position could be HELD. Those are different questions, and the second one is what
decides whether a backtest's "hold to convergence" assumption is fiction:

  * Margin is recomputed every session. It rises when volatility rises, when a leg
    approaches delivery, and when the exchange revises parameters -- which is
    precisely when the position is already losing.
  * Indian commodity clearing marks to market in CASH daily. A loss is not netted
    against a buffer; it is debited and must be paid in by the next session. So the
    binding constraint is free cash, not notional or even margin.
  * A position that cannot fund its pay-in is liquidated by the broker, at whatever
    price is available, regardless of how attractive the signal still looks.

So every trade carries a daily record, and the engine can force an exit on a margin
call. A strategy that only works if you are never forced out is not a strategy.

A NOTE ON "MAINTENANCE MARGIN"
------------------------------
The US-style maintenance level -- a floor below initial, below which you top back up
to initial -- is not how MCX works. Indian practice is that the FULL margin
requirement must be maintained at all times, with daily MTM settled in cash and a
penalty for shortfall. `maintenance_pct` therefore defaults to 1.0, meaning 100% of
the requirement. It is configurable for anyone modelling a venue that works the other
way, and the difference is stated rather than quietly assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stage3_gates_decision import margin_capital as margin_mod


@dataclass
class LedgerConfig:
    account_capital: float = 1_000_000.0
    # Fraction of the computed requirement that must be maintained. 1.0 = the full
    # requirement, which is Indian practice; lower values model a US-style
    # maintenance level.
    maintenance_pct: float = 1.0
    # Sessions allowed to meet a call before the position is liquidated. Indian
    # pay-in is T+1, so one session.
    grace_sessions: int = 1
    # Free cash a prudent operator holds beyond the requirement, as a multiple of the
    # requirement. Used to report whether the account was ever uncomfortably close.
    target_buffer_mult: float = 0.25
    force_liquidate_on_call: bool = True


# ---------------------------------------------------------------------------
# per-position daily ledger
# ---------------------------------------------------------------------------

def build_ledger(position_daily: pd.DataFrame,
                 cfg: LedgerConfig | None = None) -> pd.DataFrame:
    """
    Enrich the backtest's per-position daily records with capital fields.

    `position_daily` comes from `backtest.run`, one row per (position, session), and
    already carries the margin breakdown priced on that session. This adds the
    account-level view: utilisation, free cash, buffer, and whether a call was live.
    """
    cfg = cfg or LedgerConfig()
    if position_daily is None or position_daily.empty:
        return pd.DataFrame()

    d = position_daily.sort_values(["trade_id", "date"]).copy()

    # capital blocked = the margin requirement plus the cash a prudent operator
    # keeps free against it
    d["required_margin"] = d["initial_margin"] * cfg.maintenance_pct
    d["capital_blocked"] = d["initial_margin"] * (1.0 + cfg.target_buffer_mult)

    # running MTM on the position itself
    g = d.groupby("trade_id", sort=False)
    d["mtm_pnl_cum"] = d["mtm_pnl"]
    d["mtm_pnl_day"] = g["mtm_pnl"].diff().fillna(d["mtm_pnl"])
    d["worst_mtm_to_date"] = g["mtm_pnl"].cummin()

    # the buffer that would have been needed to survive this position's drawdown
    d["cash_buffer_needed"] = (-d["worst_mtm_to_date"]).clip(lower=0.0)

    d["margin_utilisation"] = np.where(
        cfg.account_capital > 0,
        d["required_margin"] / cfg.account_capital, np.nan)

    return d.reset_index(drop=True)


def position_summary(ledger: pd.DataFrame, trades: pd.DataFrame,
                     cfg: LedgerConfig | None = None) -> pd.DataFrame:
    """
    One row per contract pair position, carrying every capital field.

    This is the record a risk function would ask for: what it cost to open, what it
    cost to hold, how bad it got, and whether it was ever forced.
    """
    cfg = cfg or LedgerConfig()
    if ledger is None or ledger.empty:
        return pd.DataFrame()

    rows = []
    for tid, g in ledger.groupby("trade_id", sort=True):
        g = g.sort_values("date")
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            "trade_id": tid,
            "pair": first["pair"],
            "entry_date": first["date"],
            "exit_date": last["date"],
            "sessions_held": len(g),

            # exposure
            "notional_exposure": float(first["notional"]),
            "notional_peak": float(g["notional"].max()),
            "net_delta_peak": float(g["net_delta"].abs().max())
            if "net_delta" in g.columns else np.nan,

            # margin components, at entry and at their worst
            "initial_margin": float(first["initial_margin"]),
            "initial_margin_peak": float(g["initial_margin"].max()),
            "span_margin": float(first["span"]),
            "span_margin_peak": float(g["span"].max()),
            "maintenance_margin": float(first["required_margin"]),
            "elm": float(first["elm"]),
            "additional_margin": float(first["additional"]),
            "delivery_margin": float(g["delivery"].max()),
            "concentration_margin": float(g["concentration"].max()),
            "inter_commodity_credit": float(first["inter_credit"]),
            "floor_binds": bool(first["floor_binds"]),

            # capital
            "total_capital_blocked": float(g["capital_blocked"].max()),
            "margin_pct_notional": float(first["initial_margin"]
                                         / max(first["notional"], 1e-9)),
            "peak_margin_utilisation": float(g["margin_utilisation"].max()),

            # P&L path
            "mtm_final": float(last["mtm_pnl"]),
            "mtm_worst": float(g["mtm_pnl"].min()),
            "mtm_best": float(g["mtm_pnl"].max()),
            "mtm_daily_vol": float(g["mtm_pnl_day"].std())
            if len(g) > 1 else 0.0,
            "min_cash_buffer_required": float(g["cash_buffer_needed"].max()),

            # stress events
            "margin_call_sessions": int(g["margin_call"].sum())
            if "margin_call" in g.columns else 0,
            "was_forced_out": bool(g["forced_exit"].any())
            if "forced_exit" in g.columns else False,
        })

    out = pd.DataFrame(rows)
    if trades is not None and not trades.empty and "trade_id" in trades.columns:
        keep = [c for c in ("trade_id", "net_pnl", "gross_pnl", "total_cost",
                            "exit_reason", "net_bp") if c in trades.columns]
        out = out.merge(trades[keep], on="trade_id", how="left")
    if "net_pnl" in out.columns:
        out["return_on_capital_blocked"] = (
            out["net_pnl"] / out["total_capital_blocked"].replace(0, np.nan))
    return out


# ---------------------------------------------------------------------------
# account-level margin calls
# ---------------------------------------------------------------------------

def account_ledger(bt_daily: pd.DataFrame,
                   cfg: LedgerConfig | None = None) -> pd.DataFrame:
    """
    Daily account view: equity, requirement, free cash, utilisation, call state.

    Equity is capital plus realised plus unrealised, because Indian clearing settles
    MTM in cash daily -- an unrealised loss has already left the account by the next
    session, so treating it as notional would overstate what is available.
    """
    cfg = cfg or LedgerConfig()
    if bt_daily is None or bt_daily.empty:
        return pd.DataFrame()

    d = bt_daily.sort_values("date").copy()
    if "margin" not in d.columns:
        return pd.DataFrame()

    d["equity"] = cfg.account_capital + d["equity"].fillna(0.0)
    d["required_margin"] = d["margin"].fillna(0.0) * cfg.maintenance_pct
    d["free_cash"] = d["equity"] - d["required_margin"]
    d["margin_utilisation"] = np.where(
        d["equity"] > 0, d["required_margin"] / d["equity"], np.nan)
    d["buffer_mult"] = np.where(
        d["required_margin"] > 0, d["free_cash"] / d["required_margin"], np.nan)
    d["margin_call"] = d["free_cash"] < 0
    d["below_target_buffer"] = (
        (d["required_margin"] > 0) & (d["buffer_mult"] < cfg.target_buffer_mult))
    return d.reset_index(drop=True)


def call_events(acct: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of margin-call sessions, with the shortfall at their worst."""
    if acct is None or acct.empty or "margin_call" not in acct.columns:
        return pd.DataFrame()
    d = acct.sort_values("date").reset_index(drop=True)
    grp = (d["margin_call"] != d["margin_call"].shift()).cumsum()
    out = []
    for _k, g in d[d["margin_call"]].groupby(grp):
        out.append({
            "start": g["date"].iloc[0],
            "end": g["date"].iloc[-1],
            "sessions": len(g),
            "worst_shortfall": float(-g["free_cash"].min()),
            "n_open_at_worst": int(g.loc[g["free_cash"].idxmin(), "n_open"]),
        })
    return pd.DataFrame(out)


def capital_report(acct: pd.DataFrame, summary: pd.DataFrame,
                   cfg: LedgerConfig | None = None) -> dict:
    """Headline capital statistics, for the written report."""
    cfg = cfg or LedgerConfig()
    out = {"account_capital": cfg.account_capital,
           "maintenance_pct": cfg.maintenance_pct}
    if acct is not None and not acct.empty:
        out.update({
            "peak_required_margin": float(acct["required_margin"].max()),
            "peak_margin_utilisation": float(acct["margin_utilisation"].max()),
            "min_free_cash": float(acct["free_cash"].min()),
            "margin_call_sessions": int(acct["margin_call"].sum()),
            "sessions_below_target_buffer": int(acct["below_target_buffer"].sum()),
            "sessions_with_position": int((acct["n_open"] > 0).sum()),
        })
    if summary is not None and not summary.empty:
        out.update({
            "positions": int(len(summary)),
            "max_cash_buffer_required": float(
                summary["min_cash_buffer_required"].max()),
            "mean_margin_pct_notional": float(summary["margin_pct_notional"].mean()),
            "forced_exits": int(summary["was_forced_out"].sum()),
        })
        if "return_on_capital_blocked" in summary.columns:
            out["mean_return_on_capital_blocked"] = float(
                summary["return_on_capital_blocked"].mean())
    return out


def minimum_viable_capital(summary: pd.DataFrame, acct: pd.DataFrame,
                           safety: float = 1.5) -> dict:
    """
    The smallest account that could have run this without a forced exit.

    Peak requirement plus the worst cash drawdown, times a safety multiple. Reported
    because "the strategy lost 3%" means something different on an account that was
    never stressed than on one that was one bad session from liquidation.
    """
    if summary is None or summary.empty or acct is None or acct.empty:
        return {}
    peak_req = float(acct["required_margin"].max())
    worst_draw = float(summary["min_cash_buffer_required"].max())
    need = (peak_req + worst_draw) * safety
    return {
        "peak_requirement": peak_req,
        "worst_position_drawdown": worst_draw,
        "safety_multiple": safety,
        "minimum_viable_capital": need,
    }
