"""
The falsification test, and the most valuable file in the repo.

Generate a world with residual_bp = 0. There is NO relative value in it by
construction: prices are level plus carry plus a persistent structural premium plus
noise, and nothing mean-reverts across contracts. A correct pipeline must find
nothing.

This test caught a genuine bug during development. With same-bar entry, the event
study reported t-statistics above 30 on this data -- because signed convergence
measured from the same bar contains the term +|spread[t] - centre|, which is
positive whenever spread[t] is noisy. Conditioning on a large deviation guarantees
the next observation sits closer to the mean, so ANY noise manufactures apparent
mean reversion. Entering one session later removes the shared noise term.

The discriminating signature is worth keeping in mind: noise-driven pseudo-
convergence is FLAT in horizon and fully realised at h=1, while a real
mean-reverting signal GROWS with horizon toward its half-life.
"""
from datetime import date

import numpy as np
import pytest

from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage4_simulate_deliver import eventstudy
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage2_normalize_signal import z_score as sig_mod
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

START, END = date(2021, 1, 1), date(2025, 12, 31)


def _pipeline(residual_bp, seed=11):
    cfg = SyntheticConfig(start=START, end=END, seed=seed, residual_bp=residual_bp)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(START, END)
    df, _rep, _tc = ingest.build_dataset(raw)
    df = normalize.add_normalized(df)
    rows, daily, prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    sr = carry.symbol_residual(rows)
    pairs = sig_mod.pair_spreads(sr, sig_mod.SignalConfig())
    return rows, daily, prem, pairs, src


@pytest.fixture(scope="module")
def null_world():
    return _pipeline(residual_bp=0.0)


@pytest.fixture(scope="module")
def edge_world():
    return _pipeline(residual_bp=22.0)


# --------------------------------------------------------------- the null result

def test_null_world_shows_no_convergence(null_world):
    """
    THE test. Zero-edge data must produce no economically meaningful convergence at
    any |z| band or horizon.
    """
    _rows, _daily, _prem, pairs, _src = null_world
    es = eventstudy.study(pairs)
    assert not es.empty

    extreme = es[es["z_band"].isin(["|z| 2-3", "|z| 3-inf"])]
    assert len(extreme) > 0
    # Nothing should reach the cost hurdle; a few bp of noise is acceptable.
    assert float(extreme["mean_conv_bp"].max()) < 5.0, \
        "null world shows convergence: look for a look-ahead or sign leak"


def test_null_world_hit_rate_is_a_coin_flip(null_world):
    _rows, _daily, _prem, pairs, _src = null_world
    es = eventstudy.study(pairs)
    main = es[es["z_band"] == "|z| 1-2"]
    assert len(main) > 0
    assert main["hit_rate"].between(0.40, 0.60).all(), \
        "hit rate should be ~0.5 when there is nothing to find"


def test_null_world_never_clears_costs(null_world):
    _rows, _daily, _prem, pairs, _src = null_world
    es = eventstudy.study(pairs)
    hurdle = eventstudy.cost_hurdle_table(es, round_trip_bp=33.0)
    assert not hurdle["survives_costs"].any()


def test_same_bar_entry_would_have_faked_a_signal(null_world):
    """
    Regression guard on the bug itself. With entry_lag=0 the null world produces a
    large spurious effect; with the correct lag it does not. If this test ever
    fails, the bias has been reintroduced somewhere.
    """
    _rows, _daily, _prem, pairs, _src = null_world
    biased = eventstudy.study(pairs, entry_lag=0)
    correct = eventstudy.study(pairs, entry_lag=1)

    b = biased[biased["z_band"] == "|z| 1-2"]["mean_conv_bp"].mean()
    c = correct[correct["z_band"] == "|z| 1-2"]["mean_conv_bp"].mean()
    assert b > 2.0, "expected the lag-0 bias to be visible on null data"
    assert abs(c) < 1.0, "lag-1 measurement should be clean"
    assert b > abs(c) * 3


def test_null_pseudo_convergence_is_flat_in_horizon(null_world):
    """
    Shape diagnostic. Under lag-0 measurement the null world's apparent convergence
    does not grow with horizon, because the noise reverts instantly. This is how to
    tell a real signal from a measurement artefact without any ground truth.
    """
    _rows, _daily, _prem, pairs, _src = null_world
    biased = eventstudy.study(pairs, entry_lag=0)
    band = biased[biased["z_band"] == "|z| 1-2"].sort_values("horizon")
    v = band["mean_conv_bp"].to_numpy()
    assert len(v) >= 4
    # ratio between the longest and shortest horizon stays near 1
    assert 0.5 < v[-1] / v[0] < 2.0


# ------------------------------------------------- the edge world, for contrast

def test_edge_world_shows_convergence_growing_with_horizon(edge_world):
    _rows, _daily, _prem, pairs, _src = edge_world
    es = eventstudy.study(pairs)
    band = es[es["z_band"] == "|z| 2-3"].sort_values("horizon")
    v = band["mean_conv_bp"].to_numpy()
    assert len(v) >= 4
    assert v[-1] > v[0] * 2.0, "a real signal must accumulate with horizon"
    assert float(band["hit_rate"].iloc[-1]) > 0.7


def test_edge_world_beats_its_own_control_band(edge_world):
    _rows, _daily, _prem, pairs, _src = edge_world
    es = eventstudy.study(pairs)
    h = 10
    ctrl = es[(es["z_band"] == "|z| 0-1") & (es["horizon"] == h)]["mean_conv_bp"]
    hot = es[(es["z_band"] == "|z| 2-3") & (es["horizon"] == h)]["mean_conv_bp"]
    assert len(ctrl) and len(hot)
    assert float(hot.iloc[0]) > 3.0 * float(ctrl.iloc[0])


def test_structural_premium_is_recovered_not_treated_as_signal(edge_world):
    """
    The fabrication premium must land in c_s, not in the residual. If it leaked into
    the residual, a parity strategy would sell GOLDPETAL forever and call it alpha.
    """
    _rows, _daily, prem, _pairs, src = edge_world
    truth = src.truth()
    true_bp = truth.groupby("symbol")["true_c_s"].first() * 10000
    est_bp = prem[prem["date"] > prem["date"].min()].groupby("symbol")["premium_bp"].mean()
    for sym in ("GOLDGUINEA", "GOLDPETAL"):
        if sym in est_bp.index and sym in true_bp.index:
            assert abs(float(est_bp[sym]) - float(true_bp[sym])) < 12.0, \
                "{} premium not recovered".format(sym)


def test_level_and_slope_recovered(edge_world):
    _rows, daily, _prem, _pairs, src = edge_world
    truth = src.truth()
    tl = truth.groupby("date")["true_L"].first().reset_index()
    m = daily.merge(tl, on="date").dropna(subset=["level"])
    assert float(m["level"].corr(m["true_L"])) > 0.99
    tb = truth.groupby("date")["true_b"].first().reset_index()
    mb = daily.merge(tb, on="date").dropna(subset=["slope"])
    assert float(mb["slope"].corr(mb["true_b"])) > 0.4
