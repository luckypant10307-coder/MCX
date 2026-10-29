"""
scripts/fetch_mtf.py: the image build must not publish a site a session behind.

Two failure modes, both seen on 28 Oct 2026. The archive's index sits behind a CDN
and can lag the API by a session, so a session the index does not list yet must
still be fetched. And a session the index does list can come back empty while the
mirror is mid-update, which must fail the build (--require-latest) rather than
publish a stale site. What must hold: only weekdays after the index's last date are
probed, never today, never an unbounded range; and --require-latest holds the build
to the newest LISTED session, never to a probed day that may be a holiday.
"""
from __future__ import annotations

import importlib.util
import os
from datetime import date

import pytest

_P = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "fetch_mtf.py")
spec = importlib.util.spec_from_file_location("fetch_mtf", _P)
fetch_mtf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_mtf)

CSV = b"Date,Symbol\n2026-10-26,GOLDM\n2026-10-26,GOLDTEN\n"


# ------------------------------------------------------------------ probing past the index

def test_unlisted_weekdays_skip_weekends_and_today():
    listed = [date(2026, 10, 22), date(2026, 10, 23)]          # Thu, Fri
    # Run on Wed 28 Oct: Mon 26 and Tue 27 are worth asking for; Sat/Sun and today are not.
    assert fetch_mtf.unlisted_weekdays(listed, today=date(2026, 10, 28)) == [
        date(2026, 10, 26), date(2026, 10, 27)]


def test_nothing_to_probe_when_the_index_is_current():
    assert fetch_mtf.unlisted_weekdays([date(2026, 10, 27)], today=date(2026, 10, 28)) == []
    assert fetch_mtf.unlisted_weekdays([], today=date(2026, 10, 28)) == []


def test_probing_is_bounded():
    out = fetch_mtf.unlisted_weekdays([date(2026, 1, 1)], today=date(2026, 12, 31))
    assert len(out) == fetch_mtf.PROBE_DAYS


# ------------------------------------------------------------------ --require-latest

@pytest.fixture()
def mirror(monkeypatch):
    """Index lists 26 and 27 Oct; nothing to probe unless a test says so."""
    days = [date(2026, 10, 26), date(2026, 10, 27)]
    answers = {}
    probed = []
    monkeypatch.setattr(fetch_mtf, "available_dates", lambda s: (days, "2026-10-28"))
    monkeypatch.setattr(fetch_mtf, "unlisted_weekdays", lambda listed: list(probed))
    monkeypatch.setattr(fetch_mtf, "fetch_one", lambda s, d: answers.get(d))
    monkeypatch.setattr(fetch_mtf.time, "sleep", lambda *_: None)
    return answers, probed


def test_newest_session_empty_fails_with_require_latest(mirror, tmp_path, capsys):
    answers, _ = mirror
    answers[date(2026, 10, 26)] = CSV            # 27 Oct listed but served empty
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 1
    assert "2026-10-27 came back empty" in capsys.readouterr().err
    assert os.path.exists(tmp_path / "mtf_mcx_20261026.csv")


def test_without_the_flag_an_empty_day_is_skipped(mirror, tmp_path):
    answers, _ = mirror
    answers[date(2026, 10, 26)] = CSV
    assert fetch_mtf.main(["--out-dir", str(tmp_path)]) == 0


def test_complete_download_passes(mirror, tmp_path):
    answers, _ = mirror
    answers[date(2026, 10, 26)] = CSV
    answers[date(2026, 10, 27)] = CSV.replace(b"26", b"27")
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0
    assert os.path.exists(tmp_path / "mtf_mcx_20261027.csv")


def test_newest_already_on_disk_passes(mirror, tmp_path):
    answers, _ = mirror
    (tmp_path / "mtf_mcx_20261027.csv").write_bytes(CSV)
    answers[date(2026, 10, 26)] = CSV
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0


def test_a_probed_holiday_does_not_fail_the_build(mirror, tmp_path):
    answers, probed = mirror
    answers[date(2026, 10, 26)] = CSV
    answers[date(2026, 10, 27)] = CSV.replace(b"26", b"27")
    probed.append(date(2026, 10, 28))            # not listed; no session (404) that day
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0


def test_a_probed_session_is_saved(mirror, tmp_path):
    answers, probed = mirror
    answers[date(2026, 10, 26)] = CSV
    answers[date(2026, 10, 27)] = CSV.replace(b"26", b"27")
    probed.append(date(2026, 10, 28))
    answers[date(2026, 10, 28)] = CSV.replace(b"26", b"28")
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0
    assert os.path.exists(tmp_path / "mtf_mcx_20261028.csv")
