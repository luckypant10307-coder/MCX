"""
Telegram alerts fire on transitions only, and delivery can never break the nightly job.
"""
from __future__ import annotations

import copy

from mcxrv import notify


def _snap(statuses, asof="2026-09-18", fp="aaa", sample=True):
    pairs = []
    for pair, st in statuses.items():
        a, b = pair.split("/")
        pairs.append({"pair": pair, "sym_a": a, "sym_b": b, "status": st,
                      "reason": "all gates passed; may add" if st == "ALERT" else "|z| below 2.0",
                      "z": -2.6 if st == "ALERT" else 0.4, "dislocation_bp": -38.2,
                      "direction": 1, "risk": {"lots_a": 1, "lots_b": 10,
                                               "marginal_margin": 41250.0,
                                               "buffer_state": "ok"}})
    return {"meta": {"asof": asof, "data_fingerprint": fp, "run_id": fp[:12],
                     "sample_data": sample},
            "summary": {"pairs": len(pairs)}, "pairs": pairs}


def test_first_run_reports_existing_alerts():
    ev = notify.changes(None, _snap({"GOLDM/GOLDTEN": "ALERT", "GOLDTEN/GOLDPETAL": "QUIET"}))
    assert [(e["kind"], e["pair"]) for e in ev] == [("new_alert", "GOLDM/GOLDTEN")]


def test_unchanged_alert_is_not_resent():
    s = _snap({"GOLDM/GOLDTEN": "ALERT"})
    assert notify.changes(s, copy.deepcopy(s)) == []


def test_alert_cleared_and_new_alert_in_one_run():
    prev = _snap({"GOLDM/GOLDTEN": "ALERT", "GOLDTEN/GOLDPETAL": "QUIET"})
    new = _snap({"GOLDM/GOLDTEN": "NO_TRADE", "GOLDTEN/GOLDPETAL": "ALERT"}, asof="2026-09-19")
    kinds = {(e["kind"], e["pair"]) for e in notify.changes(prev, new)}
    assert kinds == {("alert_cleared", "GOLDM/GOLDTEN"), ("new_alert", "GOLDTEN/GOLDPETAL")}


def test_no_data_and_same_session_data_change():
    assert [e["kind"] for e in notify.changes(None, _snap({}))] == ["no_data"]
    prev = _snap({"GOLDM/GOLDTEN": "QUIET"}, fp="aaa")
    new = _snap({"GOLDM/GOLDTEN": "QUIET"}, fp="bbb")
    assert [e["kind"] for e in notify.changes(prev, new)] == ["data_changed_same_session"]
    later = _snap({"GOLDM/GOLDTEN": "QUIET"}, asof="2026-09-19", fp="bbb")
    assert notify.changes(prev, later) == []        # new session, new data: normal


def test_alert_message_says_which_leg_to_buy():
    s = _snap({"GOLDM/GOLDTEN": "ALERT"})
    ev = notify.changes(None, s)[0]
    msg = notify.format_message(ev, s["meta"])
    assert msg.startswith("GOLD SPREAD ALERT (sample data) - 2026-09-18")
    assert "Buy GOLDM / sell GOLDTEN" in msg and "(1 x GOLDM, 10 x GOLDTEN)" in msg
    assert "Rs 41,250" in msg and "z -2.60" in msg
    ev["pair_state"]["direction"] = -1
    assert "Buy GOLDTEN / sell GOLDM" in notify.format_message(ev, s["meta"])
    assert "(sample data)" not in notify.format_message(ev, {**s["meta"], "sample_data": False})


def test_other_messages_render():
    s = _snap({"GOLDM/GOLDTEN": "NO_TRADE"})
    cleared = {"kind": "alert_cleared", "pair": "GOLDM/GOLDTEN", "pair_state": s["pairs"][0]}
    assert "GOLDM/GOLDTEN is now NO_TRADE" in notify.format_message(cleared, s["meta"])
    assert "No usable data" in notify.format_message({"kind": "no_data"}, s["meta"])
    assert "changed since the last run" in notify.format_message(
        {"kind": "data_changed_same_session"}, s["meta"])


def test_send_without_credentials_is_a_no_op(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    r = notify.send("hello")
    assert r["sent"] is False and "not set" in r["reason"]


class _FakeSession:
    def __init__(self, status=200, exc=None):
        self.status, self.exc, self.calls = status, exc, []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if self.exc:
            raise self.exc

        class R:
            status_code = self.status
        return R()


def test_send_posts_to_the_bot_api():
    fs = _FakeSession()
    r = notify.send("hi", token="T0K", chat_id="42", session=fs)
    assert r == {"sent": True, "status": 200}
    url, body, _ = fs.calls[0]
    assert url == "https://api.telegram.org/botT0K/sendMessage"
    assert body["chat_id"] == "42" and body["text"] == "hi"


def test_send_never_raises():
    assert notify.send("x", token="t", chat_id="c", session=_FakeSession(status=401)) == \
        {"sent": False, "status": 401}
    r = notify.send("x", token="t", chat_id="c", session=_FakeSession(exc=TimeoutError("slow")))
    assert r["sent"] is False and "TimeoutError" in r["reason"]


def test_notify_sends_one_message_per_event():
    fs = _FakeSession()
    out = notify.notify(None, _snap({"GOLDM/GOLDTEN": "ALERT", "GOLDTEN/GOLDPETAL": "ALERT"}),
                        token="t", chat_id="c", session=fs)
    assert len(out) == len(fs.calls) == 2 and all(o["sent"] for o in out)
