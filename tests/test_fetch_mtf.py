"""
The bhavcopy downloader's date list.

The archive's index can lag the API by a session (it sits behind a CDN), and a
session the index has not listed yet must still be fetched. What must hold: only
weekdays after the index's last date are probed, never today (its bhavcopy does not
exist until the evening session closes), and never an unbounded range.
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import fetch_mtf  # noqa: E402


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
