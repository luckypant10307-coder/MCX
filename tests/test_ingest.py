"""
Data-trap detection. Each test corresponds to a warning in the problem statement.
"""
from datetime import date

import pandas as pd
import pytest

from mcxrv import calendar as cal
from mcxrv import ingest
from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource


# ---------------------------------------------------------------- expiry format

@pytest.mark.parametrize("token,expected", [
    ("04SEP2026", date(2026, 9, 4)),
    ("5Sep2026", date(2026, 9, 5)),
    (" 30SEP2026 ", date(2026, 9, 30)),
    ("03may2024", date(2024, 5, 3)),
])
def test_parse_expiry(token, expected):
    assert cal.parse_expiry(token) == expected


@pytest.mark.parametrize("bad", ["2026-09-04", "04/09/2026", "04XXX2026", "", "SEP2026"])
def test_parse_expiry_rejects_rather_than_guesses(bad):
    """A misparsed expiry reassigns a row to the wrong instrument: fail loudly."""
    with pytest.raises((ValueError, TypeError)):
        cal.parse_expiry(bad)


def test_expiry_roundtrip():
    d = date(2026, 9, 4)
    assert cal.parse_expiry(cal.format_expiry(d)) == d


def test_expiry_window_enforcement():
    assert cal.expiry_window_ok("GOLDM", date(2026, 9, 4))
    assert not cal.expiry_window_ok("GOLDM", date(2026, 9, 29))
    assert cal.expiry_window_ok("GOLDPETAL", date(2026, 9, 29))
    assert not cal.expiry_window_ok("GOLDPETAL", date(2026, 9, 4))
    # February clips to the month's real last day
    assert cal.expiry_window_ok("GOLDPETAL", date(2026, 2, 27))


# ------------------------------------------------------------------ date format

def test_response_date_is_mm_dd_by_default():
    d, flag = ingest.parse_response_date("12/31/2024")
    assert d == date(2024, 12, 31)
    assert flag == "ok"


def test_detects_swapped_day_month():
    """First field above 12 violates the documented MM/DD contract."""
    d, flag = ingest.parse_response_date("31/12/2024")
    assert d == date(2024, 12, 31)
    assert flag == "swapped"


def test_ambiguous_date_resolved_by_requested_date():
    """
    Below day 13 both orderings parse. The requested date is what disambiguates,
    and without it a format bug is indistinguishable from a date echo.
    """
    d, flag = ingest.parse_response_date("01/12/2024", requested=date(2024, 1, 12))
    assert d == date(2024, 1, 12)
    assert flag == "ambiguous_ok"

    # only the swap matches the request -> the field ordering is wrong
    d, flag = ingest.parse_response_date("01/12/2024", requested=date(2024, 12, 1))
    assert d == date(2024, 12, 1)
    assert flag == "swapped"


def test_unparseable_date_reported():
    d, flag = ingest.parse_response_date("not-a-date")
    assert d is None
    assert flag == "unparseable"


# -------------------------------------------------------------------- date echo

def test_date_echo_is_detected():
    """
    THE critical trap. MCX answers a holiday request with the last trading day.
    Accepting it silently duplicates one session across many dates, which
    manufactures zero-return days and inflates every Sharpe downstream.
    """
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 2, 29), seed=5)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    _df, rep, _tc = ingest.build_dataset(raw)
    assert rep.counts.get("date_echo_rows", 0) > 0, "echo must be flagged"
    assert "date_echo_examples" in rep.details


def test_no_echo_when_only_trading_days_requested():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 3, 31), seed=5)
    src = SyntheticSource(cfg)
    frames = [src.fetch_day(d).assign(_requested_date=d.isoformat(), _source="synthetic")
              for d in src.session_list()]
    raw = pd.concat(frames, ignore_index=True)
    _df, rep, _tc = ingest.build_dataset(raw)
    assert rep.counts.get("date_echo_rows", 0) == 0


# ------------------------------------------------------------ symbol and dupes

def test_symbol_whitespace_trimmed_and_counted():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 1, 31), seed=5,
                          inject_space_padding=True)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    assert raw["Symbol"].iloc[0] != raw["Symbol"].iloc[0].strip()
    df, rep, _ = ingest.build_dataset(raw)
    assert rep.counts.get("symbol_whitespace", 0) > 0
    assert (df["symbol"] == df["symbol"].str.strip()).all()


def test_duplicate_key_collapsed_to_one_row_per_contract_day():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 2, 29), seed=9,
                          inject_duplicate_rows=0.25)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    df, rep, _ = ingest.build_dataset(raw)
    assert rep.counts.get("duplicate_rows_collapsed", 0) > 0
    assert not df.duplicated(subset=["date", "contract_id"]).any()


def test_contract_id_tracks_expiry_not_near_month():
    """
    Instruments are keyed by (symbol, expiry). A 'near month' key would splice two
    contracts together and inject an artificial price jump at every roll.
    """
    cid = cal.contract_id("GOLDM  ", date(2026, 9, 4))
    assert cid == "GOLDM|2026-09-04"
    assert cal.contract_id("GOLDM", date(2026, 10, 5)) != cid


# ------------------------------------------------------- derived settlements

def test_zero_volume_rows_flagged_untradeable():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 3, 31), seed=4)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    df, rep, _ = ingest.build_dataset(raw)
    assert rep.counts.get("no_trade_rows", 0) > 0
    assert df.loc[df["volume"] <= 0, "settle_untradeable"].all()


def test_lifecycle_columns_and_tender():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 4, 30), seed=4)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    df, _rep, _tc = ingest.build_dataset(raw)
    assert (df["sessions_to_expiry"] >= 0).all()
    assert (df["days_to_expiry"] >= 0).all()
    # tender fires only near expiry
    assert df.loc[df["in_tender"], "sessions_to_expiry"].max() <= 3


def test_strict_mode_refuses_on_echo():
    cfg = SyntheticConfig(start=date(2024, 1, 1), end=date(2024, 2, 29), seed=5)
    src = SyntheticSource(cfg)
    raw = src.fetch_range(cfg.start, cfg.end)
    with pytest.raises(ValueError, match="date echo"):
        ingest.tidy(raw, strict=True)
