"""
Effective-dated specifications, and the stored per-contract snapshot.

The problem being solved: a single static spec per symbol means that the moment an
exchange revises anything -- a purity, a lot size, a quotation base -- the new value
is silently applied to ALL of history, rewriting every past normalised price with no
error raised. That is invisible corruption of exactly the kind this project is meant
to be careful about.

Two defences, tested here:

  1. Specs are effective-dated by the contract's LISTING date. A revision applies to
     contracts listed after it, never retroactively to contracts already trading,
     which is how exchanges actually work. Two expiries of the same symbol can be
     governed by different specs on the same session.
  2. The database stores each contract's spec as captured at ingest, and
     `normalize.add_normalized` prefers it over the in-code registry. A later edit to
     the registry therefore cannot rewrite stored history; where the two disagree,
     `spec_mismatch` surfaces it rather than one silently winning.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv.stage1_intake import calendar as cal
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage1_intake import specs


@pytest.fixture
def clean_registry():
    """Snapshot and restore the module-level registry around each test."""
    hist = {k: list(v) for k, v in specs.SPEC_HISTORY.items()}
    current = dict(specs.SPECS)
    yield
    specs.SPEC_HISTORY.clear()
    specs.SPEC_HISTORY.update(hist)
    specs.SPECS.clear()
    specs.SPECS.update(current)


# ---------------------------------------------------------------- registry

def test_default_registry_has_one_open_version_per_symbol():
    assert not specs.has_revisions()
    for sym in specs.ALL_SYMBOLS:
        v = specs.versions(sym)
        assert len(v) == 1
        assert v[0].valid_from is None and v[0].valid_to is None


def test_get_without_date_is_unchanged():
    """Every existing caller passes no date and must keep the same answer."""
    for sym in specs.ALL_SYMBOLS:
        assert specs.get(sym) is specs.SPECS[sym]
    assert specs.get("GOLDM").purity == 0.995
    assert specs.get("  goldm ").quote_base_g == 10.0


def test_add_version_splits_the_timeline(clean_registry):
    old = specs.get("GOLDM")
    # fine_g_per_lot is a DERIVED property, not a field: it recomputes from
    # trading_unit_g * purity, so a purity revision propagates on its own.
    new = replace(old, purity=0.999)
    assert new.fine_g_per_lot == pytest.approx(99.9)
    cut = date(2024, 7, 1)
    specs.add_version("GOLDM", new, valid_from=cut)

    v = specs.versions("GOLDM")
    assert len(v) == 2
    assert v[0].valid_from is None and v[0].valid_to == cut
    assert v[1].valid_from == cut and v[1].valid_to is None
    assert specs.has_revisions("GOLDM")

    # windows are contiguous and non-overlapping
    assert specs.get("GOLDM", on=date(2024, 6, 30)).purity == 0.995
    assert specs.get("GOLDM", on=cut).purity == 0.999
    assert specs.get("GOLDM", on=date(2025, 1, 1)).purity == 0.999
    # no date still means "current"
    assert specs.get("GOLDM").purity == 0.999


def test_gap_in_spec_history_raises_rather_than_guessing(clean_registry):
    """
    A date covered by no version is a data problem. Falling back to today's spec
    would be the exact silent corruption this machinery exists to prevent.
    """
    old = specs.get("GOLDPETAL")
    specs.SPEC_HISTORY["GOLDPETAL"] = [
        specs.SpecVersion(spec=old, valid_from=date(2024, 1, 1), valid_to=None)]
    with pytest.raises(KeyError, match="no GOLDPETAL specification covers"):
        specs.get("GOLDPETAL", on=date(2023, 6, 1))


# ------------------------------------------------- normalisation by listing date

def _frame(rows):
    df = pd.DataFrame(rows)
    df["contract_id"] = [cal.contract_id(s, e)
                         for s, e in zip(df["symbol"], df["expiry_date"])]
    return df


def test_revision_does_not_rewrite_older_contracts(clean_registry):
    """
    THE test. An old contract and a new one, same symbol, same session, different
    governing specs. The old one must keep the purity it was listed under.
    """
    old_contract = date(2024, 3, 5)      # listed well before the revision
    new_contract = date(2025, 3, 5)      # listed after it
    session = date(2024, 6, 3)

    df = _frame([
        {"symbol": "GOLDM", "expiry_date": old_contract, "date": date(2024, 1, 2),
         "close": 35485.0},
        {"symbol": "GOLDM", "expiry_date": old_contract, "date": session,
         "close": 35485.0},
        {"symbol": "GOLDM", "expiry_date": new_contract, "date": date(2024, 9, 2),
         "close": 35485.0},
        {"symbol": "GOLDM", "expiry_date": new_contract, "date": date(2024, 12, 2),
         "close": 35485.0},
    ])

    before = normalize.add_normalized(df)
    assert (before["purity"] == 0.995).all()
    baseline_old = float(before.loc[1, "norm_close"])

    # exchange revises GOLDM to 999 for contracts listed from 2024-07-01
    new_spec = replace(specs.get("GOLDM"), purity=0.999)
    specs.add_version("GOLDM", new_spec, valid_from=date(2024, 7, 1))

    after = normalize.add_normalized(df)

    old_rows = after[after["expiry_date"] == old_contract]
    new_rows = after[after["expiry_date"] == new_contract]

    assert (old_rows["purity"] == 0.995).all(), \
        "a contract listed before the revision must keep its original purity"
    assert (new_rows["purity"] == 0.999).all(), \
        "a contract listed after the revision must take the new purity"

    assert float(old_rows.loc[1, "norm_close"]) == pytest.approx(baseline_old), \
        "the revision retroactively rewrote a historical normalised price"
    assert float(new_rows["norm_close"].iloc[0]) != pytest.approx(baseline_old)


def test_same_session_two_specs(clean_registry):
    """Both contracts trading on one session, governed by different revisions."""
    session = date(2024, 12, 2)
    df = _frame([
        {"symbol": "GOLDM", "expiry_date": date(2025, 1, 6), "date": date(2024, 2, 1),
         "close": 35485.0},
        {"symbol": "GOLDM", "expiry_date": date(2025, 6, 5), "date": date(2024, 11, 1),
         "close": 35485.0},
    ])
    df = pd.concat([df, df.assign(date=session)], ignore_index=True)

    specs.add_version("GOLDM", replace(specs.get("GOLDM"), purity=0.999),
                      valid_from=date(2024, 7, 1))
    out = normalize.add_normalized(df)
    day = out[out["date"] == session]
    assert day["purity"].nunique() == 2, \
        "two expiries on one session should be able to carry different specs"


# ------------------------------------------------- stored snapshot precedence

def test_stored_snapshot_wins_over_the_registry(clean_registry):
    """
    Simulates reading from the database after the registry has been edited. The
    stored value governs, so history cannot be rewritten by a code change.
    """
    df = _frame([
        {"symbol": "GOLDM", "expiry_date": date(2024, 3, 5), "date": date(2024, 2, 1),
         "close": 35485.0},
    ])
    df["spec_purity"] = 0.995
    df["spec_quote_base_g"] = 10.0
    df["spec_trading_unit_g"] = 100.0
    df["spec_fine_g_per_lot"] = 99.5
    df["spec_tick_size_inr"] = 1.0

    # registry now says something different
    specs.add_version("GOLDM", replace(specs.get("GOLDM"), purity=0.999),
                      valid_from=date(2020, 1, 1))

    out = normalize.add_normalized(df, prefer_stored=True)
    assert float(out.loc[0, "purity"]) == 0.995, "stored snapshot must win"
    assert bool(out.loc[0, "spec_mismatch"]), "the disagreement must be surfaced"
    assert out.loc[0, "spec_source"] == "stored"

    ignored = normalize.add_normalized(df, prefer_stored=False)
    assert float(ignored.loc[0, "purity"]) == 0.999
    assert ignored.loc[0, "spec_source"] == "registry"


def test_no_mismatch_flag_when_they_agree():
    df = _frame([
        {"symbol": "GOLDPETAL", "expiry_date": date(2026, 9, 30),
         "date": date(2026, 9, 1), "close": 3596.5},
    ])
    sp = specs.get("GOLDPETAL")
    df["spec_purity"] = sp.purity
    df["spec_quote_base_g"] = sp.quote_base_g
    df["spec_trading_unit_g"] = sp.trading_unit_g
    df["spec_fine_g_per_lot"] = sp.fine_g_per_lot
    df["spec_tick_size_inr"] = sp.tick_size_inr

    out = normalize.add_normalized(df)
    assert not bool(out.loc[0, "spec_mismatch"])
    assert out.loc[0, "spec_source"] == "stored"
    assert float(out.loc[0, "norm_close"]) == pytest.approx(
        sp.to_inr_per_10g_fine(3596.5))


def test_missing_snapshot_falls_back_to_registry():
    """A frame with no spec columns at all must behave exactly as before."""
    df = _frame([
        {"symbol": "GOLDTEN", "expiry_date": date(2026, 9, 30),
         "date": date(2026, 9, 1), "close": 35616.0},
    ])
    out = normalize.add_normalized(df)
    assert out.loc[0, "spec_source"] == "registry"
    assert not bool(out.loc[0, "spec_mismatch"])
    assert float(out.loc[0, "norm_close"]) == pytest.approx(
        specs.get("GOLDTEN").to_inr_per_10g_fine(35616.0))


def test_partial_snapshot_fills_the_gaps():
    """tick_size may be absent from an older snapshot; the rest must still apply."""
    df = _frame([
        {"symbol": "GOLDM", "expiry_date": date(2026, 10, 5),
         "date": date(2026, 9, 1), "close": 35485.0},
    ])
    df["spec_purity"] = 0.995
    df["spec_quote_base_g"] = 10.0
    out = normalize.add_normalized(df)
    assert float(out.loc[0, "purity"]) == 0.995
    assert float(out.loc[0, "tick_size_inr"]) == specs.get("GOLDM").tick_size_inr
    assert np.isfinite(float(out.loc[0, "tick_bp_of_norm"]))


def test_normalisation_unchanged_for_the_current_dataset():
    """
    Regression guard: with no revisions recorded, every number must match what the
    plain per-symbol lookup produced before effective dating existed.
    """
    df = _frame([
        {"symbol": s, "expiry_date": date(2026, 9, 30), "date": date(2026, 9, 1),
         "close": 10000.0}
        for s in specs.ALL_SYMBOLS
    ])
    out = normalize.add_normalized(df)
    for r in out.itertuples(index=False):
        sp = specs.get(r.symbol)
        assert r.norm_close == pytest.approx(sp.to_inr_per_10g_fine(10000.0))
        assert r.fine_g_per_lot == pytest.approx(sp.fine_g_per_lot)
