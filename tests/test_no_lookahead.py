"""
No-look-ahead guarantees, tested mechanically rather than asserted in prose.

The strongest test available: mutate the FUTURE and confirm the past does not move.
If any statistic for day t changes when data after t changes, there is a leak.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv import carry, ingest, normalize
from mcxrv import signal as sig_mod
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource


def _dataset(seed=3, start=date(2022, 1, 1), end=date(2024, 6, 30)):
    cfg = SyntheticConfig(start=start, end=end, seed=seed)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(start, end)
    df, _rep, _tc = ingest.build_dataset(raw)
    return normalize.add_normalized(df), src


def test_trailing_stats_exclude_today():
    """
    A z-score must be measured against a distribution that does not contain today's
    observation. Including it shrinks extreme readings exactly when they matter.
    """
    s = pd.Series([0.0] * 120 + [1000.0])
    cfg = sig_mod.SignalConfig(lookback=100, min_history=50, use_robust_scale=False)
    st = sig_mod._trailing_stats(s, cfg)
    # trailing mean at the spike must be the pre-spike mean, i.e. zero
    assert st["trail_mean"].iloc[-1] == pytest.approx(0.0)
    assert st["trail_n"].iloc[-1] <= 100


def test_truncating_the_future_does_not_change_the_past():
    """
    Fit on the full sample, then on a truncated sample, and compare the overlap.
    Any difference in residuals or premiums on shared dates is a look-ahead leak.
    """
    df, _ = _dataset()
    cut = sorted(df["date"].unique())[-60]

    rows_full, daily_full, prem_full = carry.fit(df, carry.CarryConfig())
    rows_trunc, daily_trunc, prem_trunc = carry.fit(
        df[df["date"] < cut], carry.CarryConfig())

    a = rows_full[rows_full["date"] < cut][["date", "contract_id", "residual"]]
    b = rows_trunc[["date", "contract_id", "residual"]]
    m = a.merge(b, on=["date", "contract_id"], suffixes=("_full", "_trunc"))
    assert len(m) > 1000
    d = (m["residual_full"] - m["residual_trunc"]).abs()
    assert float(d.max()) < 1e-9, "residuals changed when future data was removed"

    da = daily_full[daily_full["date"] < cut][["date", "level", "slope"]]
    db = daily_trunc[["date", "level", "slope"]]
    md = da.merge(db, on="date", suffixes=("_f", "_t"))
    assert float((md["level_f"] - md["level_t"]).abs().max()) < 1e-9
    assert float((md["slope_f"] - md["slope_t"]).abs().max()) < 1e-9

    pa = prem_full[prem_full["date"] < cut]
    pb = prem_trunc
    mp = pa.merge(pb, on=["date", "symbol"], suffixes=("_f", "_t"))
    assert len(mp) > 0
    assert float((mp["premium_log_f"] - mp["premium_log_t"]).abs().max()) < 1e-9


def test_signal_scores_unchanged_by_future_data():
    df, _ = _dataset()
    rows, daily, _ = carry.fit(df, carry.CarryConfig())
    sr = carry.symbol_residual(rows)
    cut = sorted(sr["date"].unique())[-50]

    cfg = sig_mod.SignalConfig()
    full = sig_mod.score_symbols(sr, cfg)
    trunc = sig_mod.score_symbols(sr[sr["date"] < cut], cfg)

    m = (full[full["date"] < cut][["date", "symbol", "z"]]
         .merge(trunc[["date", "symbol", "z"]], on=["date", "symbol"],
                suffixes=("_f", "_t")))
    both = m.dropna(subset=["z_f", "z_t"])
    assert len(both) > 100
    assert float((both["z_f"] - both["z_t"]).abs().max()) < 1e-9


def test_execution_lag_moves_signal_forward():
    """A signal from day t must be actionable no earlier than t+1."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date
    sig = pd.DataFrame({"date": dates, "pair": ["A/B"] * 3,
                        "z": [3.0, 0.1, -2.0], "spread_bp": [30.0, 1.0, -20.0]})
    out = sig_mod.lag_for_execution(sig, sessions=1)
    assert len(out) == 2
    # the z computed on Jan 1 is acted on Jan 2
    row = out[out["date"] == dates[1]].iloc[0]
    assert row["z"] == pytest.approx(3.0)
    assert row["signal_date"] == dates[0]


def test_premium_estimate_requires_minimum_history():
    """Before the trailing window fills, no premium and hence no signal."""
    df, _ = _dataset(start=date(2023, 1, 1), end=date(2023, 4, 30))
    rows, daily, _ = carry.fit(df, carry.CarryConfig(min_window_days=40))
    early = rows[rows["date"] < sorted(rows["date"].unique())[40]]
    assert not early["has_premium_est"].any()
