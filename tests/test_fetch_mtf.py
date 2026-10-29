"""scripts/fetch_mtf.py: the image build must not publish a site a session behind."""
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


@pytest.fixture()
def mirror(monkeypatch):
    days = [date(2026, 10, 26), date(2026, 10, 27)]
    answers = {}
    monkeypatch.setattr(fetch_mtf, "available_dates", lambda s: (days, "2026-10-28"))
    monkeypatch.setattr(fetch_mtf, "fetch_one", lambda s, d: answers.get(d))
    monkeypatch.setattr(fetch_mtf.time, "sleep", lambda *_: None)
    return answers


def test_newest_session_empty_fails_with_require_latest(mirror, tmp_path, capsys):
    mirror[date(2026, 10, 26)] = CSV            # 27 Oct listed but served empty
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 1
    assert "2026-10-27 came back empty" in capsys.readouterr().err
    assert os.path.exists(tmp_path / "mtf_mcx_20261026.csv")


def test_without_the_flag_an_empty_day_is_skipped(mirror, tmp_path):
    mirror[date(2026, 10, 26)] = CSV
    assert fetch_mtf.main(["--out-dir", str(tmp_path)]) == 0


def test_complete_download_passes(mirror, tmp_path):
    mirror[date(2026, 10, 26)] = CSV
    mirror[date(2026, 10, 27)] = CSV.replace(b"26", b"27")
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0
    assert os.path.exists(tmp_path / "mtf_mcx_20261027.csv")


def test_newest_already_on_disk_passes(mirror, tmp_path):
    (tmp_path / "mtf_mcx_20261027.csv").write_bytes(CSV)
    mirror[date(2026, 10, 26)] = CSV
    assert fetch_mtf.main(["--out-dir", str(tmp_path), "--require-latest"]) == 0
