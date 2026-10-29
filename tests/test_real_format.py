"""
The current MCX bhavcopy download, as it actually arrives.

The fixture is eight real rows from the 22 Oct 2026 file. That format broke the
reader in three ways the synthetic source never exercised: "22 Oct 2026" dates,
"Volume(Lots)" / "Open Interest(Lots)" headers, and options on the same symbols
mixed into the file. Real downloads also arrive as many identical copies.
"""
from __future__ import annotations

import os
import shutil
from datetime import date

import pytest

from mcxrv.stage1_intake import calendar as cal
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage1_intake import specs
from mcxrv.sources.csv_dir import CSVDirectorySource

FIXTURE = os.path.join(os.path.dirname(__file__), "data",
                       "bhavcopy_22102026_current_format.csv")


@pytest.fixture()
def raw_dir(tmp_path):
    # Three copies, named the way a browser saves repeated downloads.
    for name in ("BhavCopyDateWise_22102026.csv", "BhavCopyDateWise_22102026 (1).csv",
                 "BhavCopyDateWise_22102026 (2).csv"):
        shutil.copy(FIXTURE, tmp_path / name)
    return str(tmp_path)


def _build(raw_dir):
    raw = CSVDirectorySource(raw_dir).load_all()
    return raw, *ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)[:2]


def test_text_month_dates_parse():
    assert ingest.parse_response_date("22 Oct 2026") == (date(2026, 10, 22), "ok")
    assert ingest.parse_response_date("01 October 2026") == (date(2026, 10, 1), "ok")


def test_headers_with_lots_suffix_are_read(raw_dir):
    raw, _, _ = _build(raw_dir)
    assert raw["Volume"].notna().all() and raw["OpenInterest"].notna().all()
    assert raw["Instrument"].notna().all()


def test_options_dropped_futures_kept_copies_collapsed(raw_dir):
    _, df, rep = _build(raw_dir)
    # 5 gold futures rows, once each, despite three copies of the file
    assert len(df) == 5
    assert sorted(df["symbol"]) == ["GOLDGUINEA", "GOLDGUINEA", "GOLDM", "GOLDPETAL", "GOLDTEN"]
    assert rep.counts["non_futures_rows_dropped"] == 6          # 2 options x 3 copies
    assert rep.counts["duplicate_rows_collapsed"] == 10
    assert set(df["date"]) == {date(2026, 10, 22)}
    goldm = df[df["symbol"] == "GOLDM"].iloc[0]
    assert goldm["close"] == 141802.0 and goldm["volume"] == 45850
    assert goldm["open_interest"] == 48432


def test_real_prices_line_up_per_gram_of_fine_gold(raw_dir):
    _, df, _ = _build(raw_dir)
    n = normalize.add_normalized(df)
    oct30 = n[n["expiry_date"] == date(2026, 10, 30)].set_index("symbol")["norm_close"] / 10
    # GOLDPETAL (1 g) and GOLDGUINEA (8 g) at the same expiry agree to within 1 bp
    assert abs(oct30["GOLDPETAL"] / oct30["GOLDGUINEA"] - 1) < 1e-4
    # GOLDTEN trades a persistent ~45 bp under them in the real file: a symbol
    # premium for the carry model's c_s term, not a spec error (published spec: 999)
    gap_bp = (oct30["GOLDTEN"] / oct30["GOLDPETAL"] - 1) * 1e4
    assert -60 < gap_bp < -30


def test_feb_expiry_on_last_business_day_is_not_flagged(raw_dir):
    _, df, rep = _build(raw_dir)
    assert not df["expiry_window_bad"].any()
    assert rep.counts.get("expiry_outside_window", 0) == 0


def test_expiry_window_weekend_rule():
    # 27-28 Feb 2027 are Sat/Sun, so Friday the 26th is the last business day
    assert cal.expiry_window_ok("GOLDTEN", date(2027, 2, 26))
    # 29 Jan 2027 is a Friday inside the window: the 26th would be misplaced
    assert not cal.expiry_window_ok("GOLDTEN", date(2027, 1, 26))
    assert cal.expiry_window_ok("GOLDTEN", date(2027, 1, 29))
    # GOLDM's 3rd-5th window still rejects month-end dates
    assert not cal.expiry_window_ok("GOLDM", date(2026, 10, 30))


def test_short_real_history_reports_warmup_not_missing_data(raw_dir):
    from mcxrv import notify, pipeline, snapshot
    cfg = pipeline.PipelineConfig(source="csv", csv_dir=raw_dir,
                                  start=date(2026, 10, 1), end=date(2026, 10, 31))
    snap = snapshot.build(pipeline.run(cfg))
    assert snap["meta"]["sample_data"] is False and snap["meta"]["sessions"] == 1
    assert snap["summary"]["WARMUP"] == snap["summary"]["pairs"] == 6
    assert all(p["status"] == "WARMUP" and "1 of" in p["reason"] for p in snap["pairs"])
    assert "Not enough history" in snap["backtest"]["verdict"]["headline"]
    # a warming-up run is not "no usable data"
    assert [e["kind"] for e in notify.changes(None, snap)] == []
