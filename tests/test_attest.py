"""
Timestamped daily records (OpenTimestamps).

What must hold: the record carries what was said for the session and nothing that
changes between identical runs' meaning; the bytes are the same however the dict was
built; a second run on the same data does not create a second record; and a proof
verifies against the right block and fails for a file changed after stamping.

The proof fixture is the OpenTimestamps project's own example (hello-world.txt,
Bitcoin block 358391); the block explorer is faked, so nothing here needs a network.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

pytest.importorskip("opentimestamps")

from mcxrv import attest  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
BLOCK = {"height": 358391,
         "hash": "000000000000000003e892881a8cdcdc117c06d444057c98b6f04a9ee75a2319",
         "merkle_root": "8a1b66ecb7cbd07d8139a7e7d7f2c41aab1f5009b8364aaf61d03ad245e47e00",
         "timestamp": 1432827678}

SNAP = {
    "meta": {"asof": "2026-10-26", "run_id": "f427a137e820", "data_fingerprint": "f427a137e820aa",
             "generated_at": "2026-10-27T18:21:08+00:00", "source": "csv", "span_parameters": "x"},
    "summary": {"ALERT": 1, "QUIET": 5},
    "pairs": [{"pair": "GOLDM/GOLDTEN", "status": "ALERT", "z": 2.31, "reason": "z beyond 2", "extra": 1}],
    "decision_log": [{"date": "2026-10-26", "pair": "GOLDM/GOLDTEN", "status": "TRADED"},
                     {"date": "2026-10-23", "pair": "GOLDM/GOLDPETAL", "status": "SIGNAL"}],
    "backtest": {"verdict": {"headline": "No edge", "edge_survives_costs": False}},
}


class _Explorer:
    def __init__(self, block=BLOCK): self.b = block

    class _R:
        def __init__(self, text=None, data=None): self.text, self._d = text, data
        def json(self): return self._d

    def get(self, url, timeout=None):
        if "/block-height/" in url:
            return self._R(text=self.b["hash"])
        return self._R(data={"merkle_root": self.b["merkle_root"], "timestamp": self.b["timestamp"]})


def test_record_holds_the_session_and_only_its_decisions():
    r = attest.record(SNAP, b"{}")
    assert r["asof"] == "2026-10-26" and r["run_id"] == "f427a137e820"
    assert r["pairs"] == [{"pair": "GOLDM/GOLDTEN", "status": "ALERT", "z": 2.31, "reason": "z beyond 2"}]
    assert [d["date"] for d in r["decisions"]] == ["2026-10-26"]
    assert r["snapshot_sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def test_canonical_bytes_do_not_depend_on_key_order():
    r = attest.record(SNAP)
    assert attest.canonical(r) == attest.canonical(dict(reversed(list(r.items()))))
    assert json.loads(attest.canonical(r)) == r


def test_same_session_on_same_data_is_recorded_once(tmp_path):
    r = attest.record(SNAP)
    first = attest.write(r, str(tmp_path))
    assert os.path.basename(first) == "2026-10-26_f427a137e820.json"
    assert attest.write(r, str(tmp_path)) is None
    # Revised data for the same session is a new file, not an overwrite.
    r2 = dict(r, run_id="0123456789ab")
    assert os.path.basename(attest.write(r2, str(tmp_path))) == "2026-10-26_0123456789ab.json"


def test_a_record_needs_a_session_and_run():
    with pytest.raises(ValueError):
        attest.write(attest.record({"meta": {}}), "unused")


def test_proof_verifies_against_its_block(tmp_path):
    for f in ("hello-world.txt", "hello-world.txt.ots"):
        shutil.copy(os.path.join(FIX, f), tmp_path / f)
    res = attest.verify(str(tmp_path / "hello-world.txt"), session=_Explorer())
    assert res == {"ok": True, "height": 358391, "block_time": 1432827678, "block": BLOCK["hash"]}


def test_a_changed_file_fails(tmp_path):
    for f in ("hello-world.txt", "hello-world.txt.ots"):
        shutil.copy(os.path.join(FIX, f), tmp_path / f)
    with open(tmp_path / "hello-world.txt", "ab") as fh:
        fh.write(b"edited later")
    res = attest.verify(str(tmp_path / "hello-world.txt"), session=_Explorer())
    assert res["ok"] is False and "changed" in res["reason"]


def test_a_proof_against_the_wrong_block_fails(tmp_path):
    for f in ("hello-world.txt", "hello-world.txt.ots"):
        shutil.copy(os.path.join(FIX, f), tmp_path / f)
    wrong = dict(BLOCK, merkle_root="00" * 32)
    res = attest.verify(str(tmp_path / "hello-world.txt"), session=_Explorer(wrong))
    assert res["ok"] is False and "358391" in res["reason"]


def test_stamp_needs_one_calendar_and_reports_the_rest(tmp_path, monkeypatch):
    from opentimestamps import calendar
    from opentimestamps.core.notary import PendingAttestation
    from opentimestamps.core.timestamp import Timestamp

    def submit(self, digest, timeout=None):
        if "down" in self.url:
            raise OSError("unreachable")
        t = Timestamp(digest)
        t.attestations.add(PendingAttestation(self.url))
        return t
    monkeypatch.setattr(calendar.RemoteCalendar, "submit", submit)
    p = attest.write(attest.record(SNAP), str(tmp_path))
    res = attest.stamp(p, calendars=("https://up.example", "https://down.example"))
    assert res["calendars"] == ["https://up.example"] and len(res["errors"]) == 1
    assert os.path.exists(p + ".ots")
    assert attest.verify(p, session=_Explorer())["pending"] is True
    with pytest.raises(RuntimeError):
        attest.stamp(p, calendars=("https://down.example",))
