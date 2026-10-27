"""
The two scikit-learn additions: robust carry fitting and anomaly detection.

Both are OPTIONAL. scikit-learn absent must degrade gracefully -- the carry fit falls
back to weighted least squares and the detector returns nothing -- never raise. Those
fallback paths are tested explicitly, because an optional dependency that breaks the
pipeline when missing is not optional.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv.stage1_intake import anomaly
from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource

sk = pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _clean_cross_section(seed=0, n=7):
    """One well-behaved day: a straight carry line plus small noise."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.02, 0.51, n)
    a, b = np.log(48000.0), 0.06
    y = a + b * tau + rng.normal(0, 2e-4, n)
    w = np.linspace(9.0, 3.0, n)
    return tau, y, w, a, b


@pytest.fixture(scope="module")
def dataset():
    cfg = SyntheticConfig(start=date(2022, 1, 1), end=date(2024, 6, 30), seed=13)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    df, _rep, _tc = ingest.build_dataset(raw)
    return normalize.add_normalized(df), src


# ---------------------------------------------------------------------------
# 1. robust carry fit
# ---------------------------------------------------------------------------

def test_huber_matches_wls_on_clean_data():
    """Robustness must cost nothing when there is nothing to resist."""
    tau, y, w, a, b = _clean_cross_section()
    a_w, b_w, _r, _n = carry._fit_day(tau, y, w, robust=False)
    a_h, b_h, _r, _n = carry._fit_day(tau, y, w, robust=True)
    assert abs(a_h - a_w) * 10000 < 1.0, "levels should agree within 1 bp"
    assert abs(b_h - b_w) < 5e-4
    assert abs(a_w - a) * 10000 < 5.0
    assert abs(a_h - a) * 10000 < 5.0


def test_huber_resists_a_single_corrupt_print():
    """
    The whole justification. One bad settlement gives WLS unbounded leverage on that
    day's slope, and the error then propagates into every residual for the session.
    """
    tau, y, w, a_true, b_true = _clean_cross_section()
    y_bad = y.copy()
    y_bad[3] += 0.03            # 300 bp wrong: a fat-fingered settlement

    a_w, b_w, _r, _n = carry._fit_day(tau, y_bad, w, robust=False)
    a_h, b_h, _r, _n = carry._fit_day(tau, y_bad, w, robust=True)

    err_w = abs(a_w - a_true) * 10000
    err_h = abs(a_h - a_true) * 10000
    assert err_w > 10.0, "WLS should be badly pulled by the outlier"
    assert err_h < 2.0, "Huber should barely move"
    assert err_h < err_w / 5.0

    assert abs(b_h - b_true) < abs(b_w - b_true) / 3.0


def test_huber_falls_back_without_sklearn(monkeypatch):
    """An optional dependency must not be load-bearing."""
    monkeypatch.setattr(carry, "_huber_available", lambda: False)
    tau, y, w, _a, _b = _clean_cross_section()
    a_h, b_h, rmse, n = carry._fit_day(tau, y, w, robust=True)
    a_w, b_w, _r, _n = carry._fit_day(tau, y, w, robust=False)
    assert a_h == pytest.approx(a_w)
    assert b_h == pytest.approx(b_w)
    assert np.isfinite(rmse)


def test_robust_path_needs_enough_points():
    """Below five observations the robust path is skipped, not attempted."""
    tau = np.array([0.02, 0.10, 0.20])
    y = np.log(48000.0) + 0.06 * tau
    w = np.ones(3)
    a_h, b_h, _r, n = carry._fit_day(tau, y, w, robust=True)
    a_w, b_w, _r, _n = carry._fit_day(tau, y, w, robust=False)
    assert n == 3
    assert a_h == pytest.approx(a_w)


def test_rmse_stays_comparable_across_both_paths():
    """
    rmse feeds the measurement noise floor in `liquidity.annotate`, so it must keep
    meaning the same thing whichever solver ran: weighted RMSE about the fitted line.
    """
    tau, y, w, _a, _b = _clean_cross_section()
    for robust in (False, True):
        a, b, rmse, _n = carry._fit_day(tau, y, w, robust=robust)
        expect = float(np.sqrt(np.average((y - (a + b * tau)) ** 2, weights=w)))
        assert rmse == pytest.approx(expect, rel=1e-9)


def test_degenerate_and_empty_inputs_return_nan():
    y = np.log(48000.0) + np.zeros(6)
    w = np.ones(6)
    # no maturity spread at all
    a, b, rmse, n = carry._fit_day(np.full(6, 0.1), y, w, robust=True)
    assert not np.isfinite(a)
    # all weights zero
    a, b, rmse, n = carry._fit_day(np.linspace(0, 0.5, 6), y, np.zeros(6), robust=True)
    assert not np.isfinite(a)


def test_weighted_median_ignores_the_tail():
    v = np.array([1.0, 1.1, 0.9, 1.05, 50.0])
    w = np.ones(5)
    assert float(np.average(v, weights=w)) > 5.0
    assert carry._weighted_median(v, w) == pytest.approx(1.05, abs=0.1)
    assert carry._weighted_median(np.array([]), np.array([])) == 0.0


def test_robust_scope_none_reproduces_plain_wls(dataset):
    """`robust_scope="none"` must be an exact escape hatch back to the old path."""
    df, _src = dataset
    rows_a, daily_a, _p = carry.fit(df, carry.CarryConfig(robust_scope="none"))
    # recompute one day by hand through the WLS path and compare
    d0 = sorted(daily_a["date"].unique())[-1]
    g = rows_a[rows_a["date"] == d0]
    tau = g["T_years"].to_numpy()
    y = g["log_norm"].to_numpy() - g["premium_bp"].to_numpy() / 10000.0
    w = g["fit_weight"].to_numpy()
    a, b, _r, _n = carry._fit_day(tau, y, w, robust=False)
    got = daily_a[daily_a["date"] == d0].iloc[0]
    assert a == pytest.approx(float(got["level"]), rel=1e-9)
    assert b == pytest.approx(float(got["slope"]), rel=1e-9)


def test_robust_fit_flag_is_reported(dataset):
    df, _src = dataset
    _rows, daily, _p = carry.fit(df, carry.CarryConfig(robust_scope="day"))
    assert daily["robust_fit"].all()
    _rows, daily_n, _p = carry.fit(df, carry.CarryConfig(robust_scope="none"))
    assert not daily_n["robust_fit"].any()


def test_premium_recovery_survives_robust_fitting(dataset):
    """Robustness must not degrade what the model was already getting right."""
    df, src = dataset
    truth = src.truth()
    true_bp = truth.groupby("symbol")["true_c_s"].first() * 10000
    for scope in ("none", "day"):
        _rows, _daily, prem = carry.fit(df, carry.CarryConfig(robust_scope=scope))
        est = (prem[prem["date"] > prem["date"].min()]
               .groupby("symbol")["premium_bp"].mean())
        for sym in ("GOLDGUINEA", "GOLDPETAL"):
            if sym in est.index and sym in true_bp.index:
                assert abs(float(est[sym]) - float(true_bp[sym])) < 15.0, \
                    "{} premium drifted under scope={}".format(sym, scope)


# ---------------------------------------------------------------------------
# 2. anomaly detection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(dataset):
    df, _src = dataset
    rows, daily, _prem = carry.fit(df, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    return rows, anomaly.detect(rows, anomaly.AnomalyConfig(contamination=0.01))


def test_detector_returns_empty_without_sklearn(monkeypatch, dataset):
    monkeypatch.setattr(anomaly, "available", lambda: False)
    df, _src = dataset
    out = anomaly.detect(df)
    assert isinstance(out, pd.DataFrame) and out.empty


def test_detector_handles_empty_and_tiny_input():
    assert anomaly.detect(pd.DataFrame()).empty
    assert anomaly.detect(None).empty


def test_features_present_and_finite(scored):
    rows, det = scored
    assert not det.empty
    feats = anomaly.build_features(rows)
    for f in anomaly.FEATURES:
        assert f in feats.columns
    for f in anomaly.FEATURES:
        if f in det.columns:
            assert np.isfinite(det[f].to_numpy()).all(), \
                "{} reached the forest with a non-finite value".format(f)


def test_flag_rate_tracks_contamination(scored):
    rows, _det = scored
    for target in (0.005, 0.02):
        d = anomaly.detect(rows, anomaly.AnomalyConfig(contamination=target))
        rate = float(d["is_anomaly"].mean())
        assert 0.3 * target < rate < 3.0 * target


def test_detects_an_internally_consistent_corruption(scored):
    """
    Inject a price error that NO existing rule can catch: OHLC stays self-consistent,
    volume is healthy, the date and expiry are valid. Only the level is wrong. This is
    the class of problem the forest exists for.
    """
    rows, _det = scored
    bad = rows.copy()
    live = bad.index[(bad["volume"] > 100) & bad["residual_bp"].notna()]
    if len(live) < 10:
        pytest.skip("not enough liquid rows to corrupt")
    picks = list(live[:: max(len(live) // 5, 1)])[:5]
    for i in picks:
        for col in ("close", "open", "high", "low", "norm_close"):
            bad.loc[i, col] = bad.loc[i, col] * 1.035
    bad["log_norm"] = np.log(bad["norm_close"].astype(float))

    det = anomaly.detect(bad, anomaly.AnomalyConfig(contamination=0.01))
    idx = det.set_index(["contract_id", "date"])
    found = 0
    for i in picks:
        key = (bad.loc[i, "contract_id"], bad.loc[i, "date"])
        if key in idx.index:
            r = idx.loc[key]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            found += bool(r["is_anomaly"])
    # A corruption on a contract's FIRST observed session has no prior return to be
    # anomalous against, so perfect recall is not achievable and not claimed.
    assert found >= len(picks) - 1, \
        "expected at least {} of {} corruptions found, got {}".format(
            len(picks) - 1, len(picks), found)


def test_first_observation_is_not_flagged_merely_for_being_first(scored):
    """
    A contract's first session has no prior close, so its return features are missing.
    Imputing at the median keeps it from being flagged for absence of history, which
    would otherwise fill the queue with 242 meaningless hits.
    """
    rows, det = scored
    first = rows.groupby("contract_id")["date"].min().reset_index()
    first["is_first"] = True
    m = det.merge(first, on=["contract_id", "date"], how="left")
    m["is_first"] = m["is_first"].fillna(False)
    n_contracts = rows["contract_id"].nunique()
    flagged_first = int(m.loc[m["is_first"], "is_anomaly"].sum())
    assert flagged_first < 0.25 * n_contracts


def test_rule_flagged_column_matches_ingest(scored):
    rows, det = scored
    m = det.merge(rows[["contract_id", "date", "no_trade", "settle_untradeable"]],
                  on=["contract_id", "date"], how="left")
    known = m["no_trade"].fillna(False) | m["settle_untradeable"].fillna(False)
    assert (m.loc[known, "rule_flagged"]).all()


def test_summary_and_reasons_render(scored):
    _rows, det = scored
    s = anomaly.summary(det)
    assert not s.empty
    assert {"symbol", "flagged", "flagged_new"} <= set(s.columns)
    assert (s["flagged_new"] + s["flagged_already_known"] == s["flagged"]).all()

    top = anomaly.top_reasons(det, 5)
    assert len(top) <= 5
    if not top.empty:
        assert "top_feature" in top.columns

    fc = anomaly.feature_reason_counts(det)
    assert isinstance(fc, pd.DataFrame)


def test_summary_handles_empty():
    assert anomaly.summary(pd.DataFrame()).empty
    assert anomaly.top_reasons(pd.DataFrame()).empty
    assert anomaly.feature_reason_counts(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# the separation that keeps the backtest honest
# ---------------------------------------------------------------------------

def test_anomaly_flags_never_reach_the_model():
    """
    The forest is fit on the whole sample, which is fine for a retrospective audit but
    would be flat look-ahead if the flags filtered a backtest: a trade on day t would
    be excluded using information from later days. Enforce the separation structurally
    rather than by convention -- no modelling module may reference this one.
    """
    import pathlib
    import re

    from mcxrv.stage4_simulate_deliver import evaluation as attribution
    from mcxrv.stage4_simulate_deliver import replay_driver as backtest
    from mcxrv.stage4_simulate_deliver import eventstudy
    from mcxrv.stage2_normalize_signal import z_score as sig_mod
    # Modules are located by __file__, not by a hardcoded path, so this keeps
    # working wherever the package layout puts them.
    pattern = re.compile(r"import\s+anomaly\b|from\s+\S*anomaly\s+import")
    for mod in (carry, sig_mod, backtest, eventstudy, liquidity, attribution, ingest):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert not pattern.search(src), \
            "{} must not depend on anomaly detection".format(mod.__name__)
