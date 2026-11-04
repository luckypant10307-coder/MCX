"""
Cost hurdle and the cost model under it.

What must hold: a round trip is four crossings; CTT is charged on sells only and
stamp duty on buys only; the "cap" brokerage schedule never exceeds Rs 20 an order
while "floor" never falls below it; the hurdle passes at exactly 1.5x cost and fails
a hair above; live alerts and the backtest use the same rule; and the gate decides
an entry from what was known at the time, so changing later prices cannot change it.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv.stage3_gates_decision import edge_gate
from mcxrv.stage3_gates_decision import liquidity_cost as costs


# ------------------------------------------------------------------ cost model

def test_round_trip_is_four_crossings():
    cfg = costs.CostConfig(brokerage_mode="cap")
    rt = costs.round_trip_bp(500_000, 2.0, 3.0, cfg)
    one = [costs.leg_cost_inr(500_000, s, h, 1, 1e9, cfg)["total_inr"]
           for s, h in (("buy", 2.0), ("sell", 3.0), ("sell", 2.0), ("buy", 3.0))]
    assert rt["n_crossings"] == 4 and rt["total_inr"] == pytest.approx(sum(one))
    # spread alone: each leg crossed twice
    assert rt["execution_bp"] == pytest.approx(2 * 2.0 + 2 * 3.0, abs=0.1)


def test_ctt_on_sells_only_and_stamp_duty_on_buys_only():
    cfg = costs.CostConfig()
    t = 1_000_000
    buy, sell = costs.statutory_cost_inr(t, "buy", cfg), costs.statutory_cost_inr(t, "sell", cfg)
    assert sell - buy == pytest.approx(t * cfg.ctt_rate_sell - t * cfg.stamp_duty_rate_buy)
    no_tax = replace(cfg, ctt_rate_sell=0.0, stamp_duty_rate_buy=0.0)
    assert costs.statutory_cost_inr(t, "buy", no_tax) == pytest.approx(costs.statutory_cost_inr(t, "sell", no_tax))


def test_brokerage_cap_and_floor():
    cap, floor = costs.CostConfig(brokerage_mode="cap"), costs.CostConfig(brokerage_mode="floor")
    assert costs.brokerage_inr(500_000, cap) == 20.0                 # 0.03% would be Rs 150
    assert costs.brokerage_inr(5_862, cap) == pytest.approx(1.7586)  # Kite's GOLDPETAL example
    assert costs.brokerage_inr(500_000, floor) == 100.0              # 2 bp
    assert costs.brokerage_inr(5_000, floor) == 20.0                 # the minimum
    with pytest.raises(ValueError):
        costs.brokerage_inr(1, costs.CostConfig(brokerage_mode="flat"))


# ------------------------------------------------------------------ the hurdle

def _check(cost_inr, **kw):
    # |z| 2.5, scale 40, half-life 36, 40 sessions to expiry -> H = 36 = one half-life:
    # expected gross = (2.5 - 0.5) x 40 x 0.5 x 0.7 = 28.0 bp exactly. Notional Rs 10,000
    # makes cost_bp equal the rupee cost, so the boundary is exact in floating point.
    return edge_gate.edge_check(2.5, 40.0, 36.0, 40, 50, cost_inr, 0.0, 1e4,
                                edge_gate.EdgeGateConfig(**kw))


def test_gate_passes_at_exactly_the_hurdle_and_fails_just_above():
    at = 28.0 / 1.5
    r = _check(at)
    assert r["exp_gross_bp"] == 28.0 and r["cost_bp"] == at and r["ok"] is True
    assert _check(math.nextafter(at, 99))["ok"] is False


def test_gate_needs_the_minimum_net_as_well():
    # cost 18 bp: 28 >= 1.5 x 18 = 27, but net 10 < 12 required
    assert _check(18.0, min_net_bp=12.0)["ok"] is False
    assert _check(18.0, min_net_bp=10.0)["ok"] is True


def test_no_runway_or_no_half_life_never_passes():
    r = edge_gate.edge_check(3.0, 40.0, 5.0, 4, 30, 1.0, 1.0, 1e4)
    assert r["ok"] is False and "no time left" in r["reason"]
    r = edge_gate.edge_check(3.0, 40.0, float("nan"), 30, 30, 1.0, 1.0, 1e4)
    assert r["ok"] is False and "half-life" in r["reason"]


def test_live_alert_and_backtest_share_the_rule():
    from mcxrv import snapshot
    row = {"z": 2.6, "signal_ok": True}
    board = {"action": "may add"} if "may add" in snapshot.ADDABLE else {"action": next(iter(snapshot.ADDABLE))}
    fail = {"ok": False, "reason": "expected 9.0 bp vs cost 17.0 bp"}
    assert snapshot.decide(row, board, 2.0, edge=fail, cost_gate=True)["status"] == "WATCH"
    assert snapshot.decide(row, board, 2.0, edge=fail, cost_gate=False)["status"] == "ALERT"
    assert snapshot.decide(row, board, 2.0, edge={"ok": True}, cost_gate=True)["status"] == "ALERT"


# ------------------------------------------------------------------ backtest

@pytest.fixture(scope="module")
def synthetic():
    from mcxrv import pipeline
    from mcxrv.stage1_intake import validation as ingest
    from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource
    from mcxrv.stage2_normalize_signal import half_life as carry, normalize
    from mcxrv.stage2_normalize_signal import pair_match as liquidity
    from mcxrv.stage2_normalize_signal import z_score as sig
    from mcxrv.stage1_intake import specs
    raw = SyntheticSource(SyntheticConfig(start=date(2023, 1, 2), end=date(2026, 6, 30), seed=7)) \
        .fetch_range(date(2023, 1, 2), date(2026, 6, 30))
    q, _rep, _ = ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)
    q = normalize.add_normalized(q)
    rows, daily, _ = carry.fit(q, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    pairs = sig.pair_spreads(carry.symbol_residual(rows), sig.SignalConfig())
    return rows, sig.lag_for_execution(pairs, 1)


def test_trades_itemise_costs_and_record_the_expectation(synthetic):
    from mcxrv.stage4_simulate_deliver import replay_driver as bt
    rows, pairs = synthetic
    trades, _d, _g, _p = bt.run(rows, pairs, bt.BacktestConfig(emit_position_daily=False))
    assert len(trades) > 0
    total_bp = trades["total_cost"] / trades["notional_a"] * 1e4
    parts = trades["brokerage_bp"] + trades["statutory_bp"] + trades["execution_bp"]
    assert np.allclose(parts, total_bp)
    assert trades["exp_gross_bp"].notna().any()


def test_gate_on_only_removes_trades_and_counts_them(synthetic):
    from mcxrv.stage4_simulate_deliver import replay_driver as bt
    rows, pairs = synthetic
    off, _, g_off, _ = bt.run(rows, pairs, bt.BacktestConfig(emit_position_daily=False))
    on, _, g_on, _ = bt.run(rows, pairs, bt.BacktestConfig(emit_position_daily=False, cost_gate=True))
    assert len(on) <= len(off)
    if len(on):                                                   # everything entered cleared it
        assert (on["exp_net_bp"] >= 5.0 - 1e-9).all()
        assert (on["exp_gross_bp"] >= 1.5 * on["exp_cost_bp"] - 1e-9).all()
    skips = dict(zip(g_on["reason"], g_on["count"])) if len(g_on) else {}
    assert skips.get("skip_below_cost_hurdle", 0) >= len(off) - len(on)


def test_gate_decision_does_not_look_ahead(synthetic):
    """Scrambling every price after a cut-off cannot change any entry made before it."""
    from mcxrv.stage4_simulate_deliver import replay_driver as bt
    rows, pairs = synthetic
    cut = pd.Timestamp(2025, 7, 1).date()
    # A loose hurdle, so the gate admits some trades and refuses others: with nothing
    # entered the comparison below would prove nothing.
    cfg = bt.BacktestConfig(emit_position_daily=False, cost_gate=True, hurdle_mult=0.3, min_net_bp=-40.0)
    base, *_ = bt.run(rows, pairs, cfg)
    assert (pd.to_datetime(base["entry_date"]) < pd.Timestamp(cut)).sum() > 0
    later = rows["date"] >= cut
    shaken = rows.copy()
    rng = np.random.default_rng(1)
    shaken.loc[later, "close"] = shaken.loc[later, "close"] * (1 + rng.normal(0, 0.01, later.sum()))
    shaken.loc[later, "half_spread_bp"] = shaken.loc[later, "half_spread_bp"] * 5
    other, *_ = bt.run(shaken, pairs, cfg)
    key = ["pair", "entry_date", "lots_a", "lots_b", "exp_gross_bp", "exp_cost_bp"]
    before = lambda t: t[pd.to_datetime(t["entry_date"]) < pd.Timestamp(cut)][key].reset_index(drop=True)
    pd.testing.assert_frame_equal(before(base), before(other))


# ------------------------------------------------------------------ the headline

def test_verdict_calls_a_noisy_small_positive_break_even():
    from mcxrv import snapshot
    noisy = np.array([40.0, -38.0, 25.0, -20.0, 10.0, -8.0, 30.0, -27.0])     # mean +1.5
    v = snapshot._verdict({"n_trades": 8, "mean_net_bp": float(noisy.mean()), "mean_gross_bp": 20.0},
                          net_bp=noisy)
    assert v["edge_survives_costs"] is False and v["headline"].startswith("About break-even")
    steady = np.array([6.0, 7.0, 5.5, 6.5, 7.5, 5.0, 6.0, 7.0])
    v = snapshot._verdict({"n_trades": 8, "mean_net_bp": float(steady.mean()), "mean_gross_bp": 20.0},
                          net_bp=steady)
    assert v["edge_survives_costs"] is True and v["headline"].startswith("Net edge positive")
    v = snapshot._verdict({"n_trades": 8, "mean_net_bp": -3.0, "mean_gross_bp": 15.0}, net_bp=noisy)
    assert v["headline"].startswith("No edge survives costs")


def test_default_brokerage_is_the_discount_schedule():
    assert costs.CostConfig().brokerage_mode == "cap"
    assert costs.brokerage_inr(500_000, costs.CostConfig()) == 20.0
