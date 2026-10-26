"""
The MCXCCL drop folder and the circular watch.

Drop folder: a file a person saves must end up either ACCEPTED (and then used only for
its own business date) or REJECTED with the parser's reason -- never half-used, never
silently skipped. Circular watch: deduplicated, relevant, new-vs-seen tracked, and a
failure or a missing key never raises.
"""
from __future__ import annotations

import json
from datetime import date

from mcxrv import circulars, risk_inbox, spanfile

MARGIN_CSV = """MULTI COMMODITY EXCHANGE CLEARING CORPORATION
Applicable Margins as on 18-09-2026
Commodity,Price Scan Range (%),Extreme Loss Margin (%),Additional Margin (%),Delivery Period Margin (%)
GOLDM,5.50,1.25,0.00,3.00
GOLDTEN,5.50,1.25,0.00,3.00
GOLDGUINEA,6.10,1.50,0.00,3.00
GOLDPETAL,6.40,1.50,0.50,3.00
"""


def _drop(root, name, body):
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(body, encoding="utf-8")


def test_a_readable_file_is_accepted_and_filed_by_business_date(tmp_path):
    _drop(tmp_path, "margins.csv", MARGIN_CSV)
    out = risk_inbox.process(str(tmp_path))
    assert [d["status"] for d in out] == ["ACCEPTED"]
    assert out[0]["business_date"] == "2026-09-18"
    assert (tmp_path / "accepted" / "20260918_margins.csv").exists()
    assert not any((tmp_path / "inbox").iterdir())


def test_an_unreadable_file_is_rejected_with_the_reason_beside_it(tmp_path):
    _drop(tmp_path, "mystery.csv", "this,is,not\na,margin,file\n")
    out = risk_inbox.process(str(tmp_path))
    assert out[0]["status"] == "REJECTED" and out[0]["reason"]
    assert (tmp_path / "rejected" / "mystery.csv").exists()
    assert (tmp_path / "rejected" / "mystery.csv.reason.txt").read_text(encoding="utf-8").strip()


def test_every_decision_is_logged_and_reported(tmp_path):
    _drop(tmp_path, "good.csv", MARGIN_CSV)
    _drop(tmp_path, "bad.csv", "nope\n")
    risk_inbox.process(str(tmp_path))
    st = risk_inbox.status(str(tmp_path), sessions=[date(2026, 9, 17), date(2026, 9, 18)])
    assert st["accepted"] == 1 and st["rejected"] == 1 and st["waiting"] == 0
    assert st["dates"] == ["2026-09-18"]
    assert st["coverage"]["covered"] == 1 and st["coverage"]["sessions"] == 2
    assert len(json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))) == 2


def test_published_scan_ranges_apply_only_to_their_own_date(tmp_path):
    _drop(tmp_path, "margins.csv", MARGIN_CSV)
    risk_inbox.process(str(tmp_path))
    sched = spanfile.schedule_from_files(risk_inbox.accepted_files(str(tmp_path)))
    assert {d for d, _s in sched} == {date(2026, 9, 18)}
    assert sched[(date(2026, 9, 18), "GOLDPETAL")] == 0.064


# ------------------------------------------------------------------ circulars

class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _Session:
    def __init__(self, results): self.results, self.calls = results, 0
    def get(self, *_a, **_k):
        self.calls += 1
        return _Resp({"results": self.results})


RESULTS = [
    {"title": "MCX Circular: additional margin on Gold futures withdrawn", "url": "https://x.example/a", "snippet": "MCXCCL circular"},
    {"title": "Live MCX Gold Future Price Chart", "url": "https://x.example/b", "snippet": "gold margin contract"},
    {"title": "Crude oil margin circular", "url": "https://x.example/c", "snippet": "crude"},
]


def test_circulars_keep_only_relevant_results_once(tmp_path):
    s = _Session(RESULTS)
    r = circulars.check(api_key="k", seen_path=str(tmp_path / "seen.json"), session=s)
    assert r["available"] and s.calls == len(circulars.QUERIES)
    assert [x["url"] for x in r["items"]] == ["https://x.example/a"]      # no price page, no crude
    assert len(r["items"][0]["topics"]) == len(circulars.QUERIES)         # deduplicated across queries
    assert r["new"] == 1 and circulars.message(r)


def test_a_second_run_marks_nothing_new(tmp_path):
    seen = str(tmp_path / "seen.json")
    circulars.check(api_key="k", seen_path=seen, session=_Session(RESULTS))
    r = circulars.check(api_key="k", seen_path=seen, session=_Session(RESULTS))
    assert r["new"] == 0 and circulars.message(r) is None


def test_no_key_and_failures_never_raise(monkeypatch):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    assert circulars.check()["available"] is False

    class Boom:
        def get(self, *_a, **_k): raise ConnectionError("offline")
    r = circulars.check(api_key="k", session=Boom())
    assert r["available"] and r["items"] == [] and len(r["errors"]) == len(circulars.QUERIES)
