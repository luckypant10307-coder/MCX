"""
Expected edge, learned walk-forward.

What must hold: an example's outcome is the trade's gain entered the session AFTER
the signal and held `horizon` sessions; the model trained for a month never uses an
outcome that was not yet known when the month began; nothing is predicted until
enough examples exist; and the backtest gate skips a trade whose expected gain does
not cover its cost, while leaving rows without a prediction alone.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from mcxrv.stage3_gates_decision import edge  # noqa: E402


def _pairs(n=400, seed=0, pair="GOLDM/GOLDTEN"):
    """A mean-reverting spread on weekdays, with the columns pair_spreads produces."""
    rng = np.random.default_rng(seed)
    days, d = [], date(2022, 1, 3)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.8 * s[i - 1] + rng.normal(0, 10)
    z = s / 15.0
    return pd.DataFrame({"date": days, "pair": pair, "spread_bp": s, "z": z,
                         "dislocation_bp": s, "halflife": 3.0, "maturity_matched": False,
                         "signal_ok": np.abs(z) >= 2})


def test_outcome_is_entered_next_session_and_known_after_the_horizon():
    p = pd.DataFrame({"date": [date(2026, 1, d) for d in (5, 6, 7, 8, 9)], "pair": "A/B",
                      "spread_bp": [30.0, 25.0, 20.0, 10.0, 5.0], "z": [2.5, 2, 1.5, 1, 0.5]})
    o = edge.outcomes(p, horizon=2)
    # signal on the 5th: sell the spread; enter at the 6th (25), exit at the 8th (10) -> +15 bp
    assert o.iloc[0]["outcome_bp"] == pytest.approx(15.0)
    assert o.iloc[0]["known_date"] == date(2026, 1, 8)
    # too close to the end of the data: no outcome yet
    assert o.iloc[-2:]["outcome_bp"].isna().all()


def test_no_prediction_uses_an_outcome_known_after_the_month_began(monkeypatch):
    p = _pairs()
    seen = []
    real_ridge = __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge

    class Spy(real_ridge):
        def fit(self, X, y):
            seen.append(len(y))
            return super().fit(X, y)

    monkeypatch.setattr("sklearn.linear_model.Ridge", Spy)
    cfg = edge.EdgeConfig(min_train=30, train_min_abs_z=0.0)
    out = edge.predict_walk_forward(p, cfg)
    ex = edge.outcomes(p, cfg.horizon)
    ex["known_date"] = pd.to_datetime(ex["known_date"])
    for _, r in out[out["edge_trained_on"] > 0].iterrows():
        month_start = pd.Timestamp(r["date"]).to_period("M").start_time
        allowed = ((ex["known_date"] < month_start) & ex["outcome_bp"].notna()).sum()
        assert r["edge_trained_on"] == allowed            # exactly the examples known by then
    assert seen and max(seen) < len(p)


def test_nothing_is_predicted_before_enough_examples():
    out = edge.predict_walk_forward(_pairs(n=120), edge.EdgeConfig(min_train=10_000))
    assert out["exp_gain_bp"].isna().all() and (out["edge_trained_on"] == 0).all()


def test_predictions_track_gap_size_on_a_reverting_spread():
    out = edge.predict_walk_forward(_pairs(n=900, seed=3), edge.EdgeConfig(min_train=100, train_min_abs_z=1.0))
    got = out[out["exp_gain_bp"].notna()]
    assert len(got) > 10
    # bigger gaps on a mean-reverting spread should be expected to close by more
    assert got["exp_gain_bp"].corr(got["dislocation_bp"].abs()) > 0.5


def test_rule_gain_closes_to_the_exit_band():
    p = pd.DataFrame({"z": [2.0, -4.0], "dislocation_bp": [40.0, -80.0]})
    assert list(edge.rule_gain(p, exit_z=0.5)) == pytest.approx([30.0, 70.0])


# ------------------------------------------------------------------ cost shown on alerts

def test_trade_cost_prices_the_boards_trade():
    from mcxrv import snapshot
    p = {"sym_a": "GOLDM", "sym_b": "GOLDTEN", "dislocation_bp": -60.0,
         "risk": {"contract_a": "A", "contract_b": "B", "lots_a": 1, "lots_b": 10}}
    liq = {"A": {"close": 140000.0, "half_spread_bp": 1.0, "max_lots": 50},
           "B": {"close": 141000.0, "half_spread_bp": 2.0, "max_lots": 50}}
    c = snapshot.trade_cost(p, liq)
    assert c["gap_bp"] == 60.0 and c["round_trip_bp"] > 6.0          # at least 2x(1+2) bp of spread
    assert c["round_trip_bp"] == pytest.approx(c["spread_cost_bp"] + c["charges_bp"])
    assert c["covers_cost"] is True and c["gap_after_cost_bp"] == pytest.approx(60.0 - c["round_trip_bp"])
    assert snapshot.trade_cost({**p, "risk": {}}, liq) is None        # nothing priced: nothing shown


def test_alert_message_says_whether_the_gap_pays_for_the_trade():
    from mcxrv import notify
    p = {"pair": "GOLDM/GOLDTEN", "sym_a": "GOLDM", "sym_b": "GOLDTEN", "direction": 1,
         "dislocation_bp": 20.0, "z": 2.2, "risk": {}, "cost": {"round_trip_bp": 33.0, "covers_cost": False}}
    msg = notify.format_message({"kind": "new_alert", "pair_state": p}, {"asof": "2026-11-04"})
    assert "Round trip costs ~33 bp: MORE than the gap" in msg


# ------------------------------------------------------------------ pre-registered forward test

def test_forward_test_counts_only_trades_entered_after_its_start(monkeypatch):
    from datetime import date as _date
    from mcxrv import pipeline
    cfg = pipeline.PipelineConfig(source="synthetic", start=_date(2025, 1, 1), end=_date(2026, 9, 18))
    monkeypatch.setitem(pipeline.FORWARD_TEST, "start", _date(2026, 3, 2))
    res = pipeline.run(cfg)
    ft = res.notes["forward_test"]
    assert ft["candidate_rule"] == "|z| >= 3" and ft["current_rule"] == "|z| >= 2"
    assert all(pd.Timestamp(t["entry_date"]) >= pd.Timestamp(2026, 3, 2) for t in ft["candidate_trades"])
    cur = res.trades[pd.to_datetime(res.trades["entry_date"]) >= pd.Timestamp(2026, 3, 2)]
    assert ft["current"]["n"] == len(cur)
    assert ft["candidate"]["n"] == len(ft["candidate_trades"])


def test_forward_test_dates_are_fixed():
    from datetime import date as _date
    from mcxrv import pipeline
    assert pipeline.FORWARD_TEST == {"start": _date(2026, 11, 5), "registered": _date(2026, 11, 4),
                                     "candidate_z_entry": 3.0}
