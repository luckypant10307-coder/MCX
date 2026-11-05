"""
Spread estimators from daily high, low and close.

What must hold: on simulated prices with a KNOWN bid/ask spread, averaged over many
days, both estimators land near the true half-spread; wider true spreads give wider
estimates; and days with no genuine trading are left out.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from mcxrv.stage2_normalize_signal import spread_estimate as se


def _simulate(half_spread_bp: float, days: int = 400, trades: int = 300, vol_bp: float = 20.0, seed: int = 0):
    """An efficient price walking through the day; every trade prints at bid or ask."""
    rng = np.random.default_rng(seed)
    p, rows, d = np.log(100_000.0), [], date(2024, 1, 1)
    hs = half_spread_bp / 1e4
    for i in range(days):
        steps = rng.normal(0, vol_bp / 1e4 / np.sqrt(trades), trades)
        path = p + np.cumsum(steps)
        prints = path + np.where(rng.random(trades) < 0.5, hs, -hs)
        rows.append({"date": d + timedelta(days=i), "contract_id": "X|2030-01-01",
                     "high": float(np.exp(prints.max())), "low": float(np.exp(prints.min())),
                     "close": float(np.exp(prints[-1])), "settle_untradeable": False})
        p = path[-1]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("true_hs", [3.0, 10.0])
def test_both_estimators_recover_a_known_spread_on_average(true_hs):
    df = _simulate(true_hs)
    cs = se.average(se.corwin_schultz(df)["cs_half_spread_bp"])
    ar = se.abdi_ranaldo(df, window=400)["ar_half_spread_bp"].dropna().iloc[-1]
    assert cs == pytest.approx(true_hs, rel=0.6)       # noisy by design; right order of size
    assert ar == pytest.approx(true_hs, rel=0.6)


def test_wider_true_spreads_give_wider_estimates():
    narrow, wide = _simulate(2.0, seed=1), _simulate(15.0, seed=1)
    assert se.average(se.corwin_schultz(wide)["cs_half_spread_bp"]) > se.average(se.corwin_schultz(narrow)["cs_half_spread_bp"])
    assert (se.abdi_ranaldo(wide, 400)["ar_half_spread_bp"].dropna().iloc[-1]
            > se.abdi_ranaldo(narrow, 400)["ar_half_spread_bp"].dropna().iloc[-1])


def test_days_without_genuine_trading_are_dropped():
    df = _simulate(5.0, days=30)
    df.loc[10:19, "settle_untradeable"] = True
    used = se.corwin_schultz(df)
    assert len(used) == 30 - 10 - 1
