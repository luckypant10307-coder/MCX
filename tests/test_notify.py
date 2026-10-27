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


# ------------------------------------------------------------------ ntfy push

class _NtfySession:
    def __init__(self, status=200, exc=None):
        self.status, self.exc, self.calls = status, exc, []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, headers))
        if self.exc:
            raise self.exc

        class R:
            status_code = self.status
        return R()


def test_ntfy_without_a_topic_is_a_no_op(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    r = notify.send_ntfy("hello")
    assert r["sent"] is False and "NTFY_TOPIC" in r["reason"]


def test_ntfy_posts_text_with_title_priority_and_tags():
    s = _NtfySession()
    r = notify.send_ntfy("GOLDM/GOLDTEN alert", title="Gold spread ALERT", priority="high",
                         tags="rotating_light", topic="mcx-abc123", session=s)
    assert r == {"sent": True, "status": 200}
    url, data, headers = s.calls[0]
    assert url == "https://ntfy.sh/mcx-abc123"
    assert data == "GOLDM/GOLDTEN alert".encode("utf-8")
    assert headers == {"Title": "Gold spread ALERT", "Priority": "high", "Tags": "rotating_light"}


def test_ntfy_self_hosted_server_and_token():
    s = _NtfySession()
    notify.send_ntfy("x", topic="t", server="https://push.example.com/", token="tk", session=s)
    url, _d, headers = s.calls[0]
    assert url == "https://push.example.com/t" and headers["Authorization"] == "Bearer tk"


def test_ntfy_never_raises():
    assert notify.send_ntfy("x", topic="t", session=_NtfySession(status=429)) == {"sent": False, "status": 429}
    r = notify.send_ntfy("x", topic="t", session=_NtfySession(exc=ConnectionError("down")))
    assert r["sent"] is False and "ConnectionError" in r["reason"]


def test_deliver_reaches_every_configured_channel_and_styles_alerts(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    ns = _NtfySession()
    r = notify.deliver("alert text", kind="new_alert", ntfy_kw={"topic": "t", "session": ns})
    # Telegram is not configured here, ntfy is: one working channel is enough.
    assert r["sent"] is True and r["telegram"]["sent"] is False and r["ntfy"]["sent"] is True
    assert ns.calls[0][2]["Priority"] == "high"


def test_notify_pushes_each_event_to_ntfy_too():
    fs, ns = _FakeSession(), _NtfySession()
    out = notify.notify(None, _snap({"GOLDM/GOLDTEN": "ALERT"}), ntfy_kw={"topic": "t", "session": ns},
                        token="t", chat_id="c", session=fs)
    assert len(out) == 1 and len(fs.calls) == 1 and len(ns.calls) == 1
    assert out[0]["telegram"]["sent"] and out[0]["ntfy"]["sent"]


# ------------------------------------------------------------------ evening digest

def _refused(snap, pair, z=2.4, reason="half-life too long"):
    for p in snap["pairs"]:
        if p["pair"] == pair:
            p.update(status="NO_TRADE", z=z, reason=reason)
    return snap


def test_quiet_session_sends_no_digest():
    assert notify.digest(None, _snap({"GOLDM/GOLDTEN": "QUIET"})) is None


def test_digest_lists_refused_pairs_with_their_reason():
    s = _refused(_snap({"GOLDM/GOLDTEN": "QUIET", "GOLDTEN/GOLDPETAL": "QUIET"}), "GOLDM/GOLDTEN")
    text = notify.digest(None, s)
    assert "Unusual, not traded (1)" in text
    assert "GOLDM/GOLDTEN z +2.40: half-life too long" in text
    assert "GOLDTEN/GOLDPETAL" not in text


def test_digest_includes_only_todays_data_anomalies_and_the_kill_switch():
    s = _snap({"GOLDM/GOLDTEN": "QUIET"})
    s["anomalies"] = {"latest": [
        {"date": "2026-09-18", "contract_id": "GOLDM|2026-11-05", "top_feature": "ret_in_ticks"},
        {"date": "2026-09-17", "contract_id": "GOLDTEN|2026-10-30", "top_feature": "d_log_oi"}]}
    s["account"] = {"halted": True, "drawdown_pct": 0.122}
    text = notify.digest(None, s)
    assert "Data to review (1)" in text and "GOLDM 2026-11-05 price jump" in text
    assert "GOLDTEN" not in text
    assert "Kill switch ON: account 12.2% below its peak" in text


def test_digest_is_not_repeated_for_the_same_session():
    s = _refused(_snap({"GOLDM/GOLDTEN": "QUIET"}), "GOLDM/GOLDTEN")
    assert notify.digest(None, s) is not None
    assert notify.digest(copy.deepcopy(s), s) is None
    nxt = _refused(_snap({"GOLDM/GOLDTEN": "QUIET"}, asof="2026-09-19", fp="bbb"), "GOLDM/GOLDTEN")
    assert notify.digest(s, nxt) is not None


def test_digest_is_delivered_quietly():
    ns = _NtfySession()
    notify.deliver("summary", kind="digest", ntfy_kw={"topic": "t", "session": ns})
    assert ns.calls[0][2]["Priority"] == "low"
