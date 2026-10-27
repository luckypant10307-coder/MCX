"""
The account kill switch: once the account is far enough below its peak, nothing new
is opened -- live (risk board) or in the walk-forward backtest.

What matters is not the arithmetic of a drawdown but that the switch REFUSES, names
itself as the reason, overrides every pair-level rule, and does not silently change
results when it is switched off.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from mcxrv import snapshot
from mcxrv.stage4_simulate_deliver import replay_driver as backtest
from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage3_gates_decision import risk
from mcxrv.stage2_normalize_signal import z_score as sig_mod
from mcxrv.pipeline import account_state
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

CAP = 1_000_000.0
# The same world test_risk_board uses: long enough to price every pair on the last
# session and to produce a few dozen trades, so both refusals have something to refuse.
START, END = date(2022, 1, 1), date(2025, 12, 31)


# ------------------------------------------------------------------ account state

def _daily(equity):
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(equity)).date,
                         "equity": equity})


def test_drawdown_is_measured_from_the_peak_value():
    a = account_state(_daily([0.0, 50_000.0, -80_000.0]), CAP, 0.10)
    assert a["peak_value"] == pytest.approx(1_050_000.0)
    assert a["value_now"] == pytest.approx(920_000.0)
    assert a["drawdown_inr"] == pytest.approx(130_000.0)
    assert a["drawdown_pct"] == pytest.approx(0.13)
    assert a["halted"] is True
    assert a["max_drawdown_inr"] == pytest.approx(130_000.0)


def test_starting_capital_counts_as_a_peak():
    # A book that only ever lost is in drawdown from day one.
    a = account_state(_daily([-20_000.0, -60_000.0]), CAP, 0.10)
    assert a["peak_value"] == pytest.approx(CAP)
    assert a["drawdown_pct"] == pytest.approx(0.06)
    assert a["halted"] is False


def test_recovery_above_the_limit_reopens():
    a = account_state(_daily([0.0, -150_000.0, -40_000.0]), CAP, 0.10)
    assert a["halted"] is False
    assert a["max_drawdown_inr"] == pytest.approx(150_000.0)


# ------------------------------------------------------------------ decisions

def test_a_halt_turns_a_passing_signal_into_no_trade_with_the_reason():
    pair = {"z": 2.6, "signal_ok": True}
    board_row = {"action": "do not add exposure",
                 "reason": "account drawdown 12.2% of capital, limit 10%: new exposure halted until reviewed"}
    d = snapshot.decide(pair, board_row, z_entry=2.0)
    assert d["status"] == "NO_TRADE"
    assert d["reason"].startswith("account drawdown")


# ------------------------------------------------------------------ board and backtest

@pytest.fixture(scope="module")
def world():
    raw = SyntheticSource(SyntheticConfig(start=START, end=END, seed=7)).fetch_range(START, END)
    df, _r, _t = ingest.build_dataset(raw)
    df = normalize.add_normalized(df)
    rows, daily, _prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    pairs = sig_mod.pair_spreads(carry.symbol_residual(rows), sig_mod.SignalConfig())
    sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
    return rows, pairs, sp


def _board(world, dd):
    rows, pairs, sp = world
    return risk.board(rows, pairs, risk.RiskBoardConfig(account_capital=CAP), span_params=sp,
                      account_drawdown_inr=dd)


def _priced(bd):
    return bd[bd["action"] != "unavailable"]


def test_board_halts_every_pair_beyond_the_limit(world):
    bd = _priced(_board(world, 0.12 * CAP))
    assert not bd.empty
    assert (bd["action"] == "do not add exposure").all()
    assert bd["reason"].str.startswith("account drawdown 12.0%").all()


def test_board_ignores_a_drawdown_inside_the_limit(world):
    bd = _priced(_board(world, 0.05 * CAP))
    assert not bd["reason"].str.startswith("account drawdown").any()


def _run(world, **kw):
    rows, pairs, sp = world
    cfg = backtest.BacktestConfig(span_params=sp, account_capital=CAP, **kw)
    return backtest.run(rows, sig_mod.lag_for_execution(pairs, sessions=1), cfg, costs.CostConfig())


def test_backtest_switch_off_by_default_changes_nothing(world):
    base, _d, diag, _p = _run(world)
    off, _d2, diag2, _p2 = _run(world, max_drawdown_halt_pct=None)
    assert len(base) == len(off)
    assert "skip_drawdown_halt" not in set(diag["reason"]) | set(diag2["reason"])


def test_backtest_hair_trigger_switch_stops_new_entries(world):
    base, _d, _g, _p = _run(world)
    tight, _d2, diag, _p2 = _run(world, max_drawdown_halt_pct=0.0001)
    assert len(base) >= 2, "fixture must trade more than once for a halt to cut anything"
    # Any loss trips a 0.01% switch, after which nothing new is opened.
    assert len(tight) < len(base)
    assert "skip_drawdown_halt" in set(diag["reason"])
