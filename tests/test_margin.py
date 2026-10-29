"""
SPAN margin: the algorithm, the parameters, and the capital denominator.

What is tested is the ALGORITHM -- the 16-scenario scan, the netting, the spread
charge, the inter-commodity credit, SPAN + ELM. The PARAMETERS cannot be tested
against truth because they are not derivable from bhavcopy: MCX publishes them
separately. So the tests pin structural properties that must hold for any parameter
set, plus the monotonicities a sensitivity sweep relies on.

The point of the whole module is one number: return on capital actually blocked,
rather than on an arbitrary denominator. Sharpe is scale-invariant and says nothing
about capital efficiency in either direction.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv import margin, specs


def _long_goldm(lots=1, price=35485.0, s2e=20):
    return margin.MarginPosition("GOLDM", "GOLDM|2026-10-05", lots, price, s2e)


def _short_petal(lots=99, price=3596.5, s2e=20):
    return margin.MarginPosition("GOLDPETAL", "GOLDPETAL|2026-09-30", -lots, price, s2e)


# ---------------------------------------------------------------- scenarios

def test_sixteen_scenarios_with_weighted_extremes():
    assert len(margin.SPAN_SCENARIOS) == 16
    weights = {w for _f, _v, w in margin.SPAN_SCENARIOS}
    assert weights == {1.0, 0.35}
    extremes = [f for f, _v, w in margin.SPAN_SCENARIOS if w < 1.0]
    assert sorted(extremes) == [-2.0, 2.0]
    # thirds of the scan range are swept in both directions
    fracs = sorted({round(f, 4) for f, _v, _w in margin.SPAN_SCENARIOS})
    assert fracs == [-2.0, -1.0, -0.6667, -0.3333, 0.0, 0.3333, 0.6667, 1.0, 2.0]


def test_extreme_scenarios_never_bind_for_linear_positions():
    """
    A future is linear, so the +/-2x extreme at 35% weight gives 0.70x the loss of
    the full 1x scan. It can never be the maximum. This is a property of the weights,
    and it is why the extremes matter only once options enter the portfolio.
    """
    p = margin.SpanParameters()
    scan = margin.scanning_risk([_long_goldm()], p)
    expected = _long_goldm().notional * p.scan_pct("GOLDM")
    assert scan == pytest.approx(expected, rel=1e-9)
    assert 2.0 * 0.35 < 1.0


def test_scanning_risk_is_symmetric_in_direction():
    p = margin.SpanParameters()
    long_only = margin.scanning_risk([_long_goldm(lots=2)], p)
    short_only = margin.scanning_risk([_long_goldm(lots=-2)], p)
    assert long_only == pytest.approx(short_only)


def test_scanning_risk_scales_with_size_and_scan_range():
    p = margin.SpanParameters()
    one = margin.scanning_risk([_long_goldm(lots=1)], p)
    three = margin.scanning_risk([_long_goldm(lots=3)], p)
    assert three == pytest.approx(3 * one)

    wide = replace(p, price_scan_range_pct={k: 2 * v for k, v
                                            in p.price_scan_range_pct.items()})
    assert margin.scanning_risk([_long_goldm()], wide) == pytest.approx(2 * one)


def test_offsetting_positions_net_inside_a_commodity():
    """The scan applies one price move to the group, so equal and opposite nets out."""
    p = margin.SpanParameters()
    both = margin.scanning_risk(
        [_long_goldm(lots=2), _long_goldm(lots=-2)], p)
    assert both == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ spread charges

def test_intra_commodity_charge_on_matched_legs():
    p = margin.SpanParameters(intra_spread_charge_pct=0.01)
    pos = [_long_goldm(lots=2), _long_goldm(lots=-2)]
    charge = margin.intra_commodity_charge(pos, p)
    assert charge == pytest.approx(_long_goldm(lots=2).notional * 0.01)
    # an outright forms no spread and attracts no charge
    assert margin.intra_commodity_charge([_long_goldm()], p) == pytest.approx(0.0)


def test_spot_month_charge_only_inside_the_window():
    """The flat spot-month charge is now the FLOOR of the delivery-period VaR."""
    p = margin.SpanParameters(delivery_margin_floor_pct=0.02, spot_month_sessions=5)
    far = margin.spot_month_charge([_long_goldm(s2e=30)], p)
    near = margin.spot_month_charge([_long_goldm(s2e=3)], p)
    assert far == 0.0
    assert near == pytest.approx(_long_goldm().notional * 0.02)


def test_inter_commodity_credit_requires_two_recognised_members():
    p = margin.SpanParameters()
    one = {"GOLDM": [_long_goldm()]}
    scans = {"GOLDM": 100.0}
    assert margin.inter_commodity_credit(one, scans, p) == 0.0


def test_credit_needs_opposing_sides():
    """Two longs are not a spread; the credit must be zero."""
    p = margin.SpanParameters()
    pos = [_long_goldm(), margin.MarginPosition("GOLDPETAL", "y", +99, 3596.5, 20)]
    m = margin.span_margin(pos, p)
    assert m["inter_credit"] == pytest.approx(0.0)
    # and the two same-direction legs attract the full combined scan
    assert m["scanning_risk"] > 0


# ----------------------------------------------------------- the whole thing

def test_spread_margin_is_far_below_two_outrights():
    """
    SPAN grants a large credit for a long/short in recognised related commodities,
    so the SPAN requirement on a spread is a fraction of two separate positions.

    The floor is disabled here deliberately: this test is about the SPAN algorithm.
    In practice the 6% floor absorbs the whole credit, which
    `test_initial_margin_floor_absorbs_the_spread_credit` covers separately -- and
    that, not this, is what decides the capital actually blocked.
    """
    p = margin.SpanParameters(inter_spread_credit=0.70, im_floor_basis="none")
    a, b = _long_goldm(), _short_petal()
    separate = (margin.span_margin([a], p)["initial_margin"]
                + margin.span_margin([b], p)["initial_margin"])
    together = margin.span_margin([a, b], p)["initial_margin"]
    assert together < separate * 0.6
    assert together > 0


def test_breakdown_is_internally_consistent():
    p = margin.SpanParameters()
    m = margin.span_margin([_long_goldm(), _short_petal()], p)
    assert m["span"] == pytest.approx(
        max(m["scanning_risk"] + m["intra_spread"] + m["delivery"]
            + m["concentration"] - m["inter_credit"], 0.0))
    assert m["computed_margin"] == pytest.approx(
        m["span"] + m["elm"] + m["additional"])
    # initial margin is the HIGHER of the computed requirement and the floor
    assert m["initial_margin"] == pytest.approx(
        max(m["computed_margin"], m["im_floor"]))
    assert m["margin_pct_notional"] == pytest.approx(
        m["initial_margin"] / m["notional"])


def test_span_is_floored_at_zero():
    """An enormous credit must not produce negative margin."""
    p = margin.SpanParameters(inter_spread_credit=5.0)
    m = margin.span_margin([_long_goldm(), _short_petal()], p)
    assert m["span"] >= 0.0
    assert m["initial_margin"] >= m["elm"]


def test_elm_is_charged_on_full_notional_without_spread_relief():
    """ELM is a flat charge on notional: it does not receive the spread credit."""
    p = margin.SpanParameters(elm_pct=0.0125)
    a, b = _long_goldm(), _short_petal()
    m = margin.span_margin([a, b], p)
    assert m["elm"] == pytest.approx((a.notional + b.notional) * 0.0125)


def test_empty_portfolio_is_zero_everywhere():
    m = margin.span_margin([])
    assert m["initial_margin"] == 0.0
    assert m["notional"] == 0.0
    assert margin.initial_margin([]) == 0.0
    # zero-lot positions are ignored rather than counted
    assert margin.initial_margin([_long_goldm(lots=0)]) == 0.0


def test_margin_is_a_portfolio_property_not_a_sum_of_trades():
    """
    Summing per-trade margins overstates the requirement, because SPAN nets and
    credits across everything held. This is why the engine assembles the whole book.
    """
    # Floor off: a gross-notional floor is blind to offsetting, so it would mask the
    # netting this test is about. That blindness is itself real and is covered by
    # `test_floor_is_blind_to_offsetting_positions`.
    p = margin.SpanParameters(im_floor_basis="none")
    t1 = [_long_goldm(lots=1), _short_petal(lots=99)]
    t2 = [_long_goldm(lots=1), _short_petal(lots=99)]
    per_trade_sum = (margin.span_margin(t1, p)["initial_margin"]
                     + margin.span_margin(t2, p)["initial_margin"])
    portfolio = margin.span_margin(t1 + t2, p)["initial_margin"]
    assert portfolio == pytest.approx(per_trade_sum, rel=1e-6)   # identical books add
    # but an offsetting second trade must NOT double the margin
    t3 = [_long_goldm(lots=-1), _short_petal(lots=-99)]
    flat = margin.span_margin(t1 + t3, p)["initial_margin"]
    assert flat < portfolio


# ------------------------------------------------------------- parameters

def test_scan_range_derived_from_observed_volatility():
    rng = np.random.default_rng(0)
    n = 400
    rows = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n).date.tolist() * 1,
        "symbol": ["GOLDM"] * n,
        "contract_id": ["GOLDM|2026-10-05"] * n,
        "volume": [1000] * n,
        "norm_close": 35000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
    })
    out = margin.derive_scan_range(rows, k=3.5, floor_pct=0.01)
    assert "GOLDM" in out
    # ~1% daily vol at k=3.5 should land near 3.5%
    assert 0.02 < out["GOLDM"] < 0.06


def test_derive_scan_range_respects_the_floor():
    rows = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=60).date,
        "symbol": ["GOLDM"] * 60, "contract_id": ["GOLDM|x"] * 60,
        "volume": [100] * 60, "norm_close": [35000.0] * 60,   # zero volatility
    })
    out = margin.derive_scan_range(rows, floor_pct=0.03)
    assert out["GOLDM"] == pytest.approx(0.03)


def test_default_parameters_are_labelled_as_assumptions():
    """
    A margin number presented as fact when it was assumed is worse than none. The
    source string is what a report prints, so it must say so.
    """
    p = margin.SpanParameters()
    assert "assum" in p.source.lower()
    assert "not MCX published" in p.source


def test_parameters_load_from_file(tmp_path):
    import json
    f = tmp_path / "mcx_risk.json"
    f.write_text(json.dumps({"inter_spread_credit": 0.55, "elm_pct": 0.02,
                             "price_scan_range_pct": {"GOLDM": 0.08}}),
                 encoding="utf-8")
    p = margin.SpanParameters.from_mcx_file(str(f))
    assert p.inter_spread_credit == 0.55
    assert p.elm_pct == 0.02
    assert p.scan_pct("GOLDM") == 0.08
    assert "mcx_risk.json" in p.source
    assert "assum" not in p.source.lower()


# ------------------------------------------------------------- sensitivity

def test_credit_sweep_is_monotone_decreasing():
    pos = [_long_goldm(), _short_petal()]
    t = margin.sweep_credit(pos)
    assert t["initial_margin"].is_monotonic_decreasing
    assert t["pct_of_notional"].is_monotonic_decreasing


def test_scan_sweep_is_monotone_increasing():
    pos = [_long_goldm(), _short_petal()]
    t = margin.sweep_scan_range(pos)
    assert t["initial_margin"].is_monotonic_increasing
    assert t["scanning_risk"].is_monotonic_increasing


# ------------------------------------------------- the capital denominator

def test_return_on_margin_beats_an_arbitrary_denominator():
    """
    The reason the module exists. A spread is margin-efficient, so the same loss on
    a small blocked base is a much worse result than the same loss measured against
    an arbitrary large number.
    """
    eff = margin.capital_efficiency(net_pnl=-13_120.0, peak_margin=49_285.0,
                                    sessions=1410)
    assert eff["return_on_margin"] < 0
    arbitrary = -13_120.0 / 2_000_000.0
    assert eff["return_on_margin"] < arbitrary          # more negative
    assert abs(eff["return_on_margin"]) > abs(arbitrary) * 10


def test_capital_efficiency_handles_no_margin():
    eff = margin.capital_efficiency(0.0, 0.0, 100)
    assert not np.isfinite(eff["return_on_margin"])
    assert "note" in eff


# ---------------------------------------------------------------------------
# MCX specification rules: floor, ELM minimum, delivery VaR, concentration
# ---------------------------------------------------------------------------

def test_initial_margin_floor_absorbs_the_spread_credit():
    """
    The rule that decides this strategy's capital. MCX takes the HIGHER of SPAN and a
    percentage of contract value. On a matched spread SPAN falls far below the floor,
    so the inter-commodity credit is worth nothing: the floor sets what is blocked.
    """
    p = margin.SpanParameters(im_floor_pct=0.06, im_floor_basis="gross")
    m = margin.span_margin([_long_goldm(), _short_petal()], p)
    assert m["floor_binds"] is True
    assert m["computed_margin"] < m["im_floor"]
    assert m["initial_margin"] == pytest.approx(m["notional"] * 0.06)
    assert m["margin_pct_notional"] == pytest.approx(0.06)
    # the credit was computed and then made irrelevant
    assert m["inter_credit"] > 0


def test_credit_is_irrelevant_while_the_floor_binds():
    pos = [_long_goldm(), _short_petal()]
    t = margin.sweep_credit(pos, margin.SpanParameters(im_floor_pct=0.06))
    floored = t[t["pct_of_notional"] <= 0.0601]
    assert len(floored) >= 3
    assert floored["initial_margin"].nunique() == 1


def test_floor_can_be_disabled_and_basis_chosen():
    pos = [_long_goldm(), _short_petal()]
    none = margin.span_margin(pos, margin.SpanParameters(im_floor_basis="none"))
    gross = margin.span_margin(pos, margin.SpanParameters(im_floor_basis="gross"))
    net = margin.span_margin(pos, margin.SpanParameters(im_floor_basis="net_delta"))
    assert none["im_floor"] == 0.0
    assert not none["floor_binds"]
    # a fine-gram matched spread has near-zero net delta, so that basis barely floors
    assert net["im_floor"] < gross["im_floor"]
    assert gross["initial_margin"] >= net["initial_margin"]


def test_elm_never_falls_below_the_minimum():
    p = margin.SpanParameters(elm_pct=0.002, elm_min_pct=0.01, im_floor_basis="none")
    m = margin.span_margin([_long_goldm()], p)
    assert m["elm"] == pytest.approx(m["notional"] * 0.01)


def test_delivery_margin_is_a_var_that_rises_with_volatility():
    """
    A 5-day 99% VaR, not a flat percentage. It is heaviest exactly when the market is
    most disturbed, which is when a delivery-period position is hardest to exit.
    """
    p = margin.SpanParameters(delivery_margin_floor_pct=0.03)
    calm = margin.delivery_var_pct(0.005, p)
    normal = margin.delivery_var_pct(0.011, p)
    stressed = margin.delivery_var_pct(0.022, p)
    assert calm == pytest.approx(0.03)            # floor binds when quiet
    assert normal > calm
    assert stressed > normal
    # z * sigma * sqrt(5)
    assert stressed == pytest.approx(2.326 * 0.022 * np.sqrt(5), rel=1e-6)


def test_delivery_charge_only_inside_the_window():
    p = margin.SpanParameters()
    far = margin.MarginPosition("GOLDM", "x", 1, 35485.0, 30, sigma_daily=0.015)
    near = margin.MarginPosition("GOLDM", "x", 1, 35485.0, 2, sigma_daily=0.015)
    assert margin.delivery_charge([far], p) == 0.0
    assert margin.delivery_charge([near], p) > 0.0


def test_concentration_surcharge_binds_on_thin_contracts():
    """
    The thin contracts this strategy needs are exactly where concentration bites: a
    few hundred GOLDPETAL lots is a large share of that book, while the same rupee
    amount of GOLDM is negligible.
    """
    p = margin.SpanParameters(concentration_threshold=0.05,
                              concentration_surcharge_pct=0.02)
    thin = margin.MarginPosition("GOLDPETAL", "y", -99, 3596.5, 20,
                                 open_interest=400)      # ~25% of OI
    deep = margin.MarginPosition("GOLDM", "x", 1, 35485.0, 20,
                                 open_interest=40000)    # negligible
    assert margin.concentration_charge([thin], p) > 0
    assert margin.concentration_charge([deep], p) == 0
    # absent open interest the surcharge is silently skipped, not guessed
    unknown = margin.MarginPosition("GOLDPETAL", "y", -99, 3596.5, 20)
    assert margin.concentration_charge([unknown], p) == 0


# ---------------------------------------------------------------- dynamics

def test_volatility_schedule_is_trailing_and_stepped():
    rng = np.random.default_rng(1)
    n = 300
    rows = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n).date,
        "symbol": ["GOLDM"] * n, "contract_id": ["GOLDM|x"] * n,
        "volume": [1000] * n,
        "norm_close": 35000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
    })
    sched = margin.volatility_schedule(rows, revision_sessions=21)
    assert len(sched) == n
    vals = [sched[(d, "GOLDM")] for d in rows["date"]]
    # stepped, not continuous: far fewer distinct values than sessions
    assert 1 < len(set(vals)) <= n // 10


def test_params_for_session_substitutes_that_day_scan_range():
    base = margin.SpanParameters()
    d = date(2026, 9, 18)
    sched = {(d, "GOLDM"): 0.09}
    p = margin.params_for_session(base, d, sched)
    assert p.scan_pct("GOLDM") == 0.09
    assert p.scan_pct("GOLDPETAL") == base.scan_pct("GOLDPETAL")
    # an unknown date leaves the base untouched
    assert margin.params_for_session(base, date(2020, 1, 1), sched) is base
    assert margin.params_for_session(base, d, None) is base


def test_stress_covers_the_named_deterioration_paths():
    pos = [margin.MarginPosition("GOLDM", "x", 1, 35485.0, 20, sigma_daily=0.011,
                                 open_interest=40000),
           margin.MarginPosition("GOLDPETAL", "y", -99, 3596.5, 20, sigma_daily=0.012,
                                 open_interest=1500)]
    t = margin.stress(pos)
    scen = set(t["scenario"])
    for needle in ("volatility doubles", "spread credit withdrawn",
                   "delivery window", "ad-hoc"):
        assert any(needle in s for s in scen), needle
    assert (t["vs_baseline"] >= 1.0).all(), "no deterioration may reduce margin"
    # holding into delivery is the most expensive state
    dv = float(t.loc[t["scenario"].str.contains("delivery"), "vs_baseline"].iloc[0])
    assert dv > 1.0


def test_floor_makes_margin_insensitive_to_volatility_shocks():
    """
    An honest consequence of the floor: while it binds, posting more than SPAN
    requires insulates the position against scan-range increases. The dynamic risks
    mostly do not bite until something pushes computed margin above the floor.
    """
    pos = [_long_goldm(), _short_petal()]
    p = margin.SpanParameters(im_floor_pct=0.06)
    base = margin.span_margin(pos, p)["initial_margin"]
    wide = replace(p, price_scan_range_pct={k: 1.5 * v for k, v
                                            in p.price_scan_range_pct.items()})
    assert margin.span_margin(pos, wide)["initial_margin"] == pytest.approx(base)


def test_floor_is_blind_to_offsetting_positions():
    """
    A consequence of flooring on GROSS notional: closing a position by opening its
    exact opposite leaves the floor unchanged, because gross notional doubles while
    net delta goes to zero. Real exchanges net offsetting lots in the same contract
    before charging, so this is a known limitation of the gross basis -- which is why
    `im_floor_basis` is configurable and the choice is stated rather than assumed.
    """
    gross = margin.SpanParameters(im_floor_basis="gross")
    net = margin.SpanParameters(im_floor_basis="net_delta")
    opposed = [_long_goldm(lots=1), _long_goldm(lots=-1)]
    assert margin.span_margin(opposed, gross)["im_floor"] > 0
    assert margin.span_margin(opposed, net)["im_floor"] == pytest.approx(0.0, abs=1e-6)
