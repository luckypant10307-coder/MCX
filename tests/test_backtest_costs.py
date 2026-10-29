"""Sizing, calendar discipline and cost arithmetic."""
from datetime import date

import numpy as np
import pytest

from mcxrv.stage4_simulate_deliver import replay_driver as backtest
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage1_intake import specs


# --------------------------------------------------------------------- sizing

def test_sizer_refuses_when_granularity_cannot_hedge():
    """
    A GOLDPETAL leg capped at 4 lots (4 g fine) cannot hedge one GOLDGUINEA lot
    (7.992 g). Forcing it would leave a ~50% unhedged gold position, which is a
    directional bet mislabelled as a spread. The sizer must decline.
    """
    cfg = backtest.BacktestConfig(max_residual_frac=0.05)
    out = backtest._size_legs("GOLDPETAL", "GOLDGUINEA",
                              close_a=1215.0, close_b=9700.0,
                              max_lots_a=4, max_lots_b=100, cfg=cfg)
    assert out is None


def test_sizer_hedges_within_tolerance_when_it_can():
    cfg = backtest.BacktestConfig(max_residual_frac=0.05,
                                  target_notional_per_leg=500_000.0)
    out = backtest._size_legs("GOLDM", "GOLDGUINEA",
                              close_a=12000.0, close_b=9700.0,
                              max_lots_a=50, max_lots_b=2000, cfg=cfg)
    assert out is not None
    lots_a, lots_b, resid_g, frac = out
    assert lots_a >= 1 and lots_b >= 1
    assert frac <= 0.05
    fine_a = lots_a * specs.get("GOLDM").fine_g_per_lot
    fine_b = lots_b * specs.get("GOLDGUINEA").fine_g_per_lot
    assert abs(fine_a - fine_b) == pytest.approx(abs(resid_g))


def test_sizer_respects_participation_caps():
    cfg = backtest.BacktestConfig()
    out = backtest._size_legs("GOLDM", "GOLDPETAL",
                              close_a=12000.0, close_b=1215.0,
                              max_lots_a=3, max_lots_b=400, cfg=cfg)
    if out is not None:
        lots_a, lots_b, _r, _f = out
        assert lots_a <= 3
        assert lots_b <= 400


def test_sizer_returns_none_when_a_leg_has_no_capacity():
    cfg = backtest.BacktestConfig()
    assert backtest._size_legs("GOLDM", "GOLDPETAL", 12000.0, 1215.0,
                               max_lots_a=0, max_lots_b=100, cfg=cfg) is None


def test_exact_neutrality_is_practically_unattainable():
    """
    The fine-gram ratio GOLDM:GOLDGUINEA is 99.5/7.992 = 24875/1998 in lowest terms,
    so the SMALLEST exactly neutral integer combination is 1998 GOLDM lots against
    24875 GOLDGUINEA -- roughly Rs 24 crore per leg, and 24875 lots of a contract
    that trades a few hundred a day.

    The practical consequence: a residual gold exposure ALWAYS exists on this pair.
    It can be made small but never zero, so it has to be measured and reported
    rather than assumed away. That is why `residual_frac` is carried on every trade
    and why attribution isolates a `level_leak` term.
    """
    f_m = specs.get("GOLDM").fine_g_per_lot
    f_g = specs.get("GOLDGUINEA").fine_g_per_lot

    # the widely-quoted "20 : 249" is close but NOT exact
    assert abs(20 * f_m - 249 * f_g) == pytest.approx(0.008, abs=1e-6)

    # the true smallest exact solution
    assert 1998 * f_m == pytest.approx(24875 * f_g, abs=1e-6)
    assert 1998 * specs.get("GOLDM").lot_notional(12000.0) > 200_000_000

    # no small combination is exact
    best = min(abs(n * f_m - round(n * f_m / f_g) * f_g) for n in range(1, 60))
    assert best > 1e-9


def test_multiplier_converts_quote_to_position_value():
    assert backtest._multiplier("GOLDM") == pytest.approx(10.0)
    assert backtest._multiplier("GOLDTEN") == pytest.approx(1.0)
    assert backtest._multiplier("GOLDGUINEA") == pytest.approx(1.0)
    assert backtest._multiplier("GOLDPETAL") == pytest.approx(1.0)


# ---------------------------------------------------------------------- costs

def test_round_trip_charges_four_crossings():
    cfg = costs.CostConfig(brokerage_min_inr=0.0, impact_coef_bp=0.0)
    r = costs.round_trip_bp(500_000.0, 5.0, 5.0, cfg)
    assert r["n_crossings"] == 4
    # 4 crossings x 5 bp half-spread = 20 bp of execution cost
    assert r["execution_bp"] == pytest.approx(20.0, abs=1e-6)


def test_one_leg_costing_understates_by_about_four_times():
    """The arithmetic that sinks most versions of this strategy."""
    cfg = costs.CostConfig(brokerage_min_inr=0.0, impact_coef_bp=0.0)
    one = costs.leg_cost_inr(500_000.0, "buy", 5.0, 1, 1e9, cfg)
    full = costs.round_trip_bp(500_000.0, 5.0, 5.0, cfg)
    assert full["total_inr"] / one["total_inr"] > 3.0


def test_ctt_on_sell_and_stamp_on_buy_only():
    cfg = costs.CostConfig(brokerage_rate=0.0, brokerage_min_inr=0.0,
                           exchange_txn_rate=0.0, sebi_turnover_rate=0.0,
                           gst_rate=0.0)
    sell = costs.statutory_cost_inr(1_000_000.0, "sell", cfg)
    buy = costs.statutory_cost_inr(1_000_000.0, "buy", cfg)
    assert sell == pytest.approx(1_000_000.0 * cfg.ctt_rate_sell)
    assert buy == pytest.approx(1_000_000.0 * cfg.stamp_duty_rate_buy)


def test_cost_rises_with_assumed_half_spread():
    tbl = costs.cost_summary_table(500_000.0)
    assert tbl["round_trip_bp"].is_monotonic_increasing


def test_breakeven_half_spread_is_the_headline_statistic():
    cfg = costs.CostConfig()
    tight = costs.breakeven_half_spread_bp(200.0, 500_000.0, cfg)
    loose = costs.breakeven_half_spread_bp(20.0, 500_000.0, cfg)
    assert tight > loose
    # a tiny edge cannot support any spread at all
    assert costs.breakeven_half_spread_bp(1.0, 500_000.0, cfg) == 0.0


def test_impact_grows_with_participation():
    cfg = costs.CostConfig(impact_coef_bp=10.0)
    low = costs.execution_cost_bp(2.0, lots=1, max_lots=100, cfg=cfg)
    high = costs.execution_cost_bp(2.0, lots=100, max_lots=100, cfg=cfg)
    assert high > low
    assert high == pytest.approx(2.0 + 10.0)


# ----------------------------------------------------------- roll-down algebra

def test_roll_down_is_negative_for_contango():
    from mcxrv.stage2_normalize_signal import half_life as carry
    rd = carry.roll_down_bp(slope=0.06, tau_years=0.25, holding_sessions=20)
    assert rd < 0


def test_roll_cancels_for_simultaneous_entry_and_exit():
    """
    Worth stating explicitly: when both legs are entered and exited on the same
    sessions they age identically, so b*(dtau_a - dtau_b) = 0 and mechanical
    roll-down cancels out of the spread. What remains is the CURVE term,
    db*(tau_a - tau_b), which is a term-structure bet rather than roll-down. The
    roll component is non-zero only when the legs are held over different windows.
    """
    b0 = 0.06
    dtau_a = dtau_b = -20 / 252.0
    assert b0 * (dtau_a - dtau_b) == pytest.approx(0.0)
    db, tau_a, tau_b = 0.01, 0.20, 0.10
    assert db * (tau_a - tau_b) != 0.0
