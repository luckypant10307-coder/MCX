"""
Per-pair capital ledger and margin-call simulation.

The question this answers is not "what did it cost to open" but "could it have been
HELD". Indian clearing settles mark-to-market in cash daily, so the binding
constraint is free cash rather than notional or even margin. A backtest that never
models a forced exit is assuming an infinite balance sheet, and its "hold to
convergence" assumption is doing unearned work.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv.stage4_simulate_deliver import replay_driver as backtest
from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage4_simulate_deliver import position_ledger as ledger
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage2_normalize_signal import z_score as sig_mod
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

START, END = date(2022, 1, 1), date(2025, 12, 31)


def _pipeline(account_capital=1_000_000.0, **bt_kwargs):
    cfg = SyntheticConfig(start=START, end=END, seed=7)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(START, END)
    df, _r, _t = ingest.build_dataset(raw)
    df = normalize.add_normalized(df)
    rows, daily, _prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    sr = carry.symbol_residual(rows)
    pairs = sig_mod.pair_spreads(sr, sig_mod.SignalConfig())
    pe = sig_mod.lag_for_execution(pairs, 1)
    sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
    bc = backtest.BacktestConfig(
        span_params=sp, margin_schedule=margin.volatility_schedule(rows),
        account_capital=account_capital, **bt_kwargs)
    tr, bd, dg, pdly = backtest.run(rows, pe, bc, costs.CostConfig())
    return tr, bd, dg, pdly


@pytest.fixture(scope="module")
def run_normal():
    return _pipeline(account_capital=1_000_000.0)


@pytest.fixture(scope="module")
def led_normal(run_normal):
    tr, bd, _dg, pdly = run_normal
    cfg = ledger.LedgerConfig(account_capital=1_000_000.0)
    led = ledger.build_ledger(pdly, cfg)
    return led, ledger.position_summary(led, tr, cfg), ledger.account_ledger(bd, cfg)


# --------------------------------------------------------------- the record

def test_every_requested_field_is_present(led_normal):
    _led, summary, _acct = led_normal
    assert not summary.empty
    required = [
        "notional_exposure", "initial_margin", "span_margin",
        "maintenance_margin", "elm", "additional_margin", "delivery_margin",
        "concentration_margin", "total_capital_blocked", "mtm_final", "mtm_worst",
        "min_cash_buffer_required", "peak_margin_utilisation",
        "margin_call_sessions", "was_forced_out",
    ]
    missing = [c for c in required if c not in summary.columns]
    assert not missing, "missing ledger fields: {}".format(missing)
    assert summary[required].notna().all().all()


def test_one_row_per_position(led_normal, run_normal):
    _led, summary, _acct = led_normal
    trades = run_normal[0]
    assert len(summary) == len(trades)
    assert summary["trade_id"].is_unique


def test_capital_blocked_exceeds_bare_margin(led_normal):
    """Blocked capital includes the buffer a prudent operator holds beside it."""
    _led, summary, _acct = led_normal
    assert (summary["total_capital_blocked"] >= summary["initial_margin"]).all()


def test_margin_components_sum_sensibly(led_normal):
    _led, summary, _acct = led_normal
    # SPAN is one component of initial margin, never larger than it
    assert (summary["span_margin"] <= summary["initial_margin"] + 1e-6).all()
    assert (summary["elm"] > 0).all()
    assert (summary["notional_exposure"] > 0).all()


def test_maintenance_defaults_to_the_full_requirement():
    """
    MCX does not use a US-style maintenance level below initial: the full
    requirement must be maintained, with daily MTM settled in cash.
    """
    cfg = ledger.LedgerConfig()
    assert cfg.maintenance_pct == 1.0


def test_cash_buffer_is_the_worst_drawdown(led_normal):
    led, summary, _acct = led_normal
    for tid, g in led.groupby("trade_id"):
        want = max(0.0, -float(g["mtm_pnl"].min()))
        got = float(summary.loc[summary["trade_id"] == tid,
                                "min_cash_buffer_required"].iloc[0])
        assert got == pytest.approx(want, abs=1e-6)


def test_marginal_margin_is_attributed_not_standalone(led_normal):
    """
    Margin is a portfolio property. Attribution uses book-with minus book-without,
    so a second position that offsets the first is not charged as though it stood
    alone.
    """
    led, _summary, _acct = led_normal
    assert "marginal_margin" in led.columns
    multi = led.groupby("date")["trade_id"].nunique()
    shared = multi[multi > 1]
    if shared.empty:
        pytest.skip("no session with two positions open")
    d = shared.index[0]
    day = led[led["date"] == d]
    assert (day["marginal_margin"] <= day["initial_margin"] + 1e-6).any()


# ------------------------------------------------------------- account view

def test_account_ledger_fields(led_normal):
    _led, _summary, acct = led_normal
    assert not acct.empty
    for c in ("equity", "required_margin", "free_cash", "margin_utilisation",
              "buffer_mult", "margin_call", "below_target_buffer"):
        assert c in acct.columns
    # free cash is equity minus requirement, by construction
    assert np.allclose(acct["free_cash"],
                       acct["equity"] - acct["required_margin"])


def test_a_well_capitalised_account_is_never_called(led_normal):
    _led, summary, acct = led_normal
    assert int(acct["margin_call"].sum()) == 0
    assert not summary["was_forced_out"].any()
    assert float(acct["margin_utilisation"].max()) < 1.0


def test_capital_report_and_minimum_viable(led_normal):
    _led, summary, acct = led_normal
    rep = ledger.capital_report(acct, summary)
    assert rep["margin_call_sessions"] == 0
    assert rep["peak_required_margin"] > 0
    mv = ledger.minimum_viable_capital(summary, acct)
    assert mv["minimum_viable_capital"] > mv["peak_requirement"]
    # an account at the minimum would still be tight, which is the point of stating it
    assert mv["minimum_viable_capital"] < 1_000_000.0


# ------------------------------------------------------- the forced exit path

@pytest.fixture(scope="module")
def run_starved():
    """An account too small to carry the book: calls and liquidation must occur."""
    return _pipeline(account_capital=25_000.0)


def test_undercapitalised_account_gets_called_and_forced_out(run_starved):
    """
    THE test for this module. With capital far below the requirement the engine must
    raise margin calls and liquidate, rather than silently holding positions it could
    never have funded.
    """
    trades, bt_daily, diag, pdly = run_starved
    reasons = dict(zip(diag["reason"], diag["count"])) if not diag.empty else {}
    assert reasons.get("margin_call_sessions", 0) > 0, \
        "a starved account must register margin calls"

    cfg = ledger.LedgerConfig(account_capital=25_000.0)
    acct = ledger.account_ledger(bt_daily, cfg)
    assert int(acct["margin_call"].sum()) > 0
    assert float(acct["free_cash"].min()) < 0

    if not trades.empty and "exit_reason" in trades.columns:
        # either positions were liquidated, or none could be opened at all
        assert (reasons.get("forced_exit_margin_call", 0) > 0
                or "margin_call" in set(trades["exit_reason"])
                or len(trades) == 0)


def test_margin_call_exit_outranks_other_reasons(run_starved):
    trades, _bd, _dg, _pd = run_starved
    if trades.empty or "margin_called" not in trades.columns:
        pytest.skip("no trades in the starved run")
    called = trades[trades["margin_called"]]
    if called.empty:
        pytest.skip("no position was liquidated")
    assert (called["exit_reason"] == "margin_call").all()


def test_forced_exits_are_charged_a_widened_spread(run_starved):
    """
    A liquidation crosses a stressed book. Filling it at the normal half-spread would
    make a forced exit look as cheap as a chosen one.
    """
    trades, _bd, _dg, _pd = run_starved
    if trades.empty or "margin_called" not in trades.columns:
        pytest.skip("no trades in the starved run")
    called = trades[trades["margin_called"]]
    if called.empty:
        pytest.skip("no position was liquidated")
    assert called["illiquid_exit"].all()


def test_call_events_are_grouped_into_runs(run_starved):
    _tr, bt_daily, _dg, _pd = run_starved
    cfg = ledger.LedgerConfig(account_capital=25_000.0)
    acct = ledger.account_ledger(bt_daily, cfg)
    ev = ledger.call_events(acct)
    if ev.empty:
        pytest.skip("no call events")
    assert (ev["sessions"] >= 1).all()
    assert (ev["worst_shortfall"] > 0).all()
    assert (ev["end"] >= ev["start"]).all()


def test_capital_changes_the_outcome(run_normal, run_starved):
    """
    Capital is not a cosmetic denominator: it decides which positions survive, and
    the same signal on the same data produces a materially worse result when the
    account cannot carry the book.

    Note the trade COUNT goes UP when starved, not down. A liquidation frees the
    pair slot and the signal is usually still live, so the strategy re-enters and is
    liquidated again: an undercapitalised account churns. Asserting fewer trades
    would be asserting the opposite of what actually happens.
    """
    rich_trades = run_normal[0]
    poor_trades = run_starved[0]
    rich_diag = dict(zip(run_normal[2]["reason"], run_normal[2]["count"]))
    poor_diag = dict(zip(run_starved[2]["reason"], run_starved[2]["count"]))

    assert rich_diag.get("margin_call_sessions", 0) == 0
    assert poor_diag.get("margin_call_sessions", 0) > 0
    assert poor_diag.get("forced_exit_margin_call", 0) > 0

    # the economically meaningful difference: forced exits cost money
    if not rich_trades.empty and not poor_trades.empty:
        assert float(poor_trades["net_bp"].mean()) < float(rich_trades["net_bp"].mean())
        assert "margin_call" in set(poor_trades["exit_reason"])
        assert "margin_call" not in set(rich_trades["exit_reason"])


# ----------------------------------------------------------------- edges

def test_empty_inputs_return_empty():
    assert ledger.build_ledger(pd.DataFrame()).empty
    assert ledger.build_ledger(None).empty
    assert ledger.position_summary(pd.DataFrame(), pd.DataFrame()).empty
    assert ledger.account_ledger(pd.DataFrame()).empty
    assert ledger.call_events(pd.DataFrame()).empty
    assert ledger.minimum_viable_capital(pd.DataFrame(), pd.DataFrame()) == {}


def test_trade_ids_are_stable_and_unique(run_normal):
    """
    REGRESSION. `trade_id` was `id(p)` -- a memory ADDRESS, which CPython reuses the
    moment a closed position is freed. Two different trades in one run could share an
    id, which merged their daily records into a single ledger group (sessions_held
    became the SUM of both) and then fanned the join back out into duplicate summary
    rows. It depended on garbage-collection timing, so it surfaced in a full pipeline
    run and not in a smaller one -- the worst kind of defect to leave untested.
    """
    trades, _bd, _dg, pdly = run_normal
    assert trades["trade_id"].is_unique
    assert set(pdly["trade_id"]) <= set(trades["trade_id"])
    assert pdly["trade_id"].nunique() == len(trades)
    # ids come from a counter, so they are small and dense -- not addresses
    assert trades["trade_id"].min() >= 1
    assert trades["trade_id"].max() <= len(trades)


def test_sessions_held_matches_the_trade_not_a_merged_group(led_normal, run_normal):
    """The symptom the id collision produced: a ledger row twice as long as its trade."""
    _led, summary, _acct = led_normal
    trades = run_normal[0]
    merged = summary.merge(trades[["trade_id", "sessions_held"]], on="trade_id",
                           suffixes=("_ledger", "_trade"))
    assert len(merged) == len(summary), "no fan-out on the join"
    # the ledger counts the entry session too, so it is never SHORTER and never double
    assert (merged["sessions_held_ledger"]
            <= merged["sessions_held_trade"] + 1).all()
