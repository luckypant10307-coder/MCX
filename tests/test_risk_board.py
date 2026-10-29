"""
The live risk board: can this pair be added to today, and if not, why not?

The property that matters is not that the board produces numbers. It is that the
board REFUSES. A risk monitor that never says no is decoration, so most of what is
asserted here is that specific deteriorations produce specific refusals, and that the
refusal names the binding constraint rather than a generic one.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv import (backtest, carry, costs, ingest, liquidity, margin, normalize,
                   risk)
from mcxrv import signal as sig_mod
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

START, END = date(2022, 1, 1), date(2025, 12, 31)


@pytest.fixture(scope="module")
def world():
    src = SyntheticSource(SyntheticConfig(start=START, end=END, seed=7))
    raw = src.fetch_range(START, END)
    df, _r, _t = ingest.build_dataset(raw)
    df = normalize.add_normalized(df)
    rows, daily, _prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    sr = carry.symbol_residual(rows)
    pairs = sig_mod.pair_spreads(sr, sig_mod.SignalConfig())
    sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
    sched = margin.volatility_schedule(rows)
    return rows, pairs, sp, sched


def _board(world, capital=1_000_000.0, book=None, **kw):
    rows, pairs, sp, sched = world
    cfg = risk.RiskBoardConfig(account_capital=capital, **kw)
    return risk.board(rows, pairs, cfg, span_params=sp, margin_schedule=sched,
                      open_book=book)


# ------------------------------------------------------------------ the record

def test_board_has_every_field_the_monitor_shows(world):
    bd = _board(world)
    assert not bd.empty
    for c in ("pair", "notional_exposure", "marginal_margin", "initial_margin",
              "span_margin", "margin_utilisation", "utilisation_if_added",
              "buffer_state", "buffer_mult", "span_change_pct", "action", "reason"):
        assert c in bd.columns, c


def test_one_row_per_pair(world):
    rows, pairs, _sp, _s = world
    bd = _board(world)
    asof = max(pairs["date"])
    assert len(bd) == int((pairs["date"] == asof).sum())
    assert bd["pair"].is_unique


def test_every_row_carries_an_action_and_a_reason(world):
    bd = _board(world)
    assert bd["action"].notna().all()
    assert bd["reason"].notna().all()
    assert (bd["reason"].astype(str).str.len() > 0).all()


# --------------------------------------------------------------- the refusals

def test_a_starved_account_is_refused_every_pair(world):
    """The board must block when the requirement cannot be funded at all."""
    bd = _board(world, capital=20_000.0)
    priced = bd[bd["action"] != "unavailable"]
    assert not priced.empty
    assert (priced["action"] == "do not add exposure").all()
    assert (priced["buffer_state"] == "breached").all()
    assert set(priced["reason"]) == {"requirement exceeds capital"}


def test_a_thin_buffer_blocks_before_the_requirement_is_breached(world):
    """
    Between 'affordable' and 'unaffordable' there is a band where the position fits
    but leaves no room for the next adverse session. That band must also refuse.
    """
    bd = _board(world)
    priced = bd[bd["action"] != "unavailable"]
    assert not priced.empty
    im = float(priced["required_if_added"].max())
    # capital just 10% above the requirement: fits, but the buffer is thin
    tight = _board(world, capital=im * 1.10)
    tp = tight[tight["pair"] == priced["pair"].iloc[
        priced["required_if_added"].argmax()]]
    r = tp.iloc[0]
    assert r["free_cash_if_added"] > 0, "should still fit"
    assert r["buffer_state"] == "low"
    assert r["action"] == "do not add exposure"
    assert r["reason"] == "maintenance buffer low"


def test_a_rising_requirement_blocks_a_pair_that_is_otherwise_fine(world):
    """
    Margin regime deterioration is its own refusal. A pair can be affordable and
    statistically valid and still be the wrong thing to add while the requirement is
    climbing, because today's margin is not tomorrow's.
    """
    rows, pairs, sp, sched = world
    asof = max(pairs["date"])
    # Force a rising regime: halve every scan range in the prior session's schedule,
    # leaving today's alone, so the same position costs materially more now.
    dates = sorted(pd.Series(rows["date"].unique()))
    prior = dates[max(0, dates.index(asof) - 21)]
    bumped = dict(sched)
    for (d, sym) in list(bumped):
        if d == prior:
            bumped[(d, sym)] = bumped[(d, sym)] * 0.25
    cfg = risk.RiskBoardConfig(account_capital=1_000_000.0)
    bd = risk.board(rows, pairs, cfg, span_params=sp, margin_schedule=bumped)
    priced = bd[bd["action"] != "unavailable"]
    if priced["span_change_pct"].max() < cfg.span_rise_block_pct:
        pytest.skip("floor binds so hard the scan range cannot move the requirement")
    hit = priced[priced["span_change_pct"] >= cfg.span_rise_block_pct]
    assert (hit["action"] == "do not add exposure").all()
    assert set(hit["reason"]) == {
        "SPAN requirement rising; margin regime deteriorating"}


def test_refusal_outranks_a_valid_signal(world):
    """A cleared gate never overrides an unfundable position."""
    bd = _board(world, capital=20_000.0)
    priced = bd[bd["action"] != "unavailable"]
    fired = priced[priced["signal_ok"]]
    if fired.empty:
        pytest.skip("no signal fired on the latest session")
    assert (fired["action"] == "do not add exposure").all()


def test_ungranular_pairs_are_declined_not_silently_sized(world):
    """
    The pairs the backtest cannot hedge to tolerance must appear as UNAVAILABLE, not
    be dropped. A board that quietly omits them is hiding the constraint.
    """
    bd = _board(world)
    un = bd[bd["action"] == "unavailable"]
    if un.empty:
        pytest.skip("every pair is sizable in this world")
    assert (un["reason"].str.contains("granularity")
            | un["reason"].str.contains("runway")).all()
    assert un["notional_exposure"].isna().all()


# -------------------------------------------------------------- marginal cost

def test_margin_is_marginal_against_the_open_book(world):
    """
    Adding to a book is not the same as opening from flat. The board must price the
    difference the book makes, whichever way it cuts.
    """
    rows, pairs, sp, sched = world
    asof = max(pairs["date"])
    live = rows[(rows["date"] == asof) & rows["tradable"]].sort_values(
        "volume", ascending=False)
    if live.empty:
        pytest.skip("no tradable contract on the latest session")
    a = live[live["symbol"] == "GOLDM"]
    b = live[live["symbol"] == "GOLDTEN"]
    if a.empty or b.empty:
        pytest.skip("front contracts unavailable")
    a, b = a.iloc[0], b.iloc[0]
    book = [margin.MarginPosition("GOLDM", a["contract_id"], 5, float(a["close"]),
                                  int(a["sessions_to_expiry"])),
            margin.MarginPosition("GOLDTEN", b["contract_id"], -50, float(b["close"]),
                                  int(b["sessions_to_expiry"]))]
    flat = _board(world)
    held = _board(world, book=book)
    assert float(held["book_required"].iloc[0]) > 0
    assert float(flat["book_required"].iloc[0]) == 0
    # utilisation must rise once the book is already consuming capital
    assert (float(held["utilisation_if_added"].max())
            > float(flat["utilisation_if_added"].max()))


def test_while_the_floor_binds_there_is_no_netting_benefit(world):
    """
    A finding, asserted so it cannot silently stop being true: when the initial-margin
    FLOOR binds, the requirement is a flat percentage of gross notional. Gross
    notional is additive, so the marginal cost of an offsetting pair equals its
    standalone cost and the inter-commodity credit is worth exactly nothing.
    """
    bd = _board(world)
    priced = bd[bd["action"] != "unavailable"]
    floored = priced[priced["floor_binds"]]
    if floored.empty:
        pytest.skip("floor does not bind in this world")
    assert np.allclose(floored["marginal_margin"], floored["initial_margin"],
                       atol=1.0)
    assert (floored["inter_commodity_credit"] > 0).any(), \
        "the credit is computed; it is just not realised"


def test_requirement_change_holds_position_size_fixed(world):
    """
    The change figure must be a risk statement. If it were recomputed with today's
    sizing it would move when prices moved, and a monitor reading '+14%' could not
    tell a margin revision from a larger trade.
    """
    rows, pairs, sp, sched = world
    cfg = risk.RiskBoardConfig(account_capital=1_000_000.0)
    big = backtest.BacktestConfig(target_notional_per_leg=2_000_000.0)
    small = backtest.BacktestConfig(target_notional_per_leg=200_000.0)
    b1 = risk.board(rows, pairs, cfg, span_params=sp, margin_schedule=sched,
                    bt_cfg=big)
    b2 = risk.board(rows, pairs, cfg, span_params=sp, margin_schedule=sched,
                    bt_cfg=small)
    m = b1.merge(b2, on="pair", suffixes=("_big", "_small"))
    m = m[m["notional_exposure_big"].notna() & m["notional_exposure_small"].notna()]
    m = m[m["notional_exposure_big"] != m["notional_exposure_small"]]
    if m.empty:
        pytest.skip("sizing did not differ")
    # different size, same percentage change: it is scale-free
    ok = m[m["span_change_pct_big"].notna() & m["span_change_pct_small"].notna()]
    if ok.empty:
        pytest.skip("no prior session to compare against")
    assert np.allclose(ok["span_change_pct_big"], ok["span_change_pct_small"],
                       atol=0.02)


# ------------------------------------------------------------------- rendering

def test_alerts_render_the_lines_a_monitor_shows(world):
    bd = _board(world, capital=20_000.0)
    lines = risk.alerts(bd)
    assert lines
    body = "\n".join(lines)
    assert "spread trade" in body
    assert "Current margin utilization" in body
    assert "Maintenance buffer" in body
    assert "Signal:" in body
    assert "do not add exposure" in body


def test_alerts_can_be_restricted_to_what_is_blocked(world):
    bd = _board(world)
    everything = risk.alerts(bd)
    blocked = risk.alerts(bd, only_blocked=True)
    assert len(blocked) <= len(everything)
    for line in blocked:
        assert "do not add exposure" in line or "Unavailable" in line


def test_span_change_line_is_omitted_when_unmeasurable(world):
    """No prior session means no change figure, and the line must not be invented."""
    rows, pairs, sp, sched = world
    first = min(pairs["date"])
    cfg = risk.RiskBoardConfig(account_capital=1_000_000.0)
    bd = risk.board(rows, pairs, cfg, span_params=sp, margin_schedule=sched,
                    asof=first)
    if bd.empty:
        pytest.skip("nothing priceable on the first session")
    unmeasurable = set(bd.loc[bd["span_change_pct"].isna(), "pair"])
    if not unmeasurable:
        pytest.skip("every pair had a prior session to compare against")
    for line in risk.alerts(bd):
        pair = line.split(" spread trade")[0]
        if pair in unmeasurable:
            assert "SPAN requirement" not in line, \
                "invented a change figure with nothing to compare against"


def test_summary_counts_add_up(world):
    bd = _board(world)
    s = risk.summary(bd)
    assert s["pairs"] == len(bd)
    assert (s["blocked"] + s["reduced"] + s["addable"] + s["unavailable"]
            + int((bd["action"] == "no signal").sum())) == s["pairs"]


# ----------------------------------------------------------------------- edges

def test_empty_inputs_return_empty():
    assert risk.board(pd.DataFrame(), pd.DataFrame()).empty
    assert risk.board(None, None).empty
    assert risk.alerts(pd.DataFrame()) == []
    assert risk.summary(pd.DataFrame()) == {}
    assert risk.open_book_from_trades(pd.DataFrame(), pd.DataFrame(), None) == []


def test_open_book_reconstruction_only_includes_live_positions(world):
    rows, pairs, sp, sched = world
    pe = sig_mod.lag_for_execution(pairs, 1)
    bc = backtest.BacktestConfig(span_params=sp, margin_schedule=sched)
    trades, _bd, _dg, _pd = backtest.run(rows, pe, bc, costs.CostConfig())
    if trades.empty:
        pytest.skip("no trades")
    t = trades.iloc[0]
    mid = t["entry_date"]
    book = risk.open_book_from_trades(rows, trades, mid)
    # every leg rebuilt at `mid` belongs to a trade that spans it
    live = trades[(trades["entry_date"] <= mid) & (trades["exit_date"] > mid)]
    assert len(book) <= 2 * len(live)
    after = risk.open_book_from_trades(rows, trades, max(rows["date"]))
    assert after == [] or all(p.lots != 0 for p in after)
