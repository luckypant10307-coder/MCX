"""
The "Ask" assistant.

What must hold: the tools read the snapshot and nothing else; the model is given the
rules and the tools on every call; a user cannot exceed the daily limit, and a failure
on our side does not use up a question; with accounts on, nobody unsigned can ask; and
an unconfigured or unreachable model is a clear 503, never a crash.

Gemini is faked throughout; nothing here calls Google.
"""
from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from fastapi.testclient import TestClient  # noqa: E402

from mcxrv import assistant, supa  # noqa: E402

SNAP = {
    "meta": {"asof": "2026-10-26", "sessions": 1236, "source": "csv", "sample_data": False,
             "span_parameters": "published broker margin rates on 720 of 1236 session(s)"},
    "summary": {"QUIET": 5, "NO_TRADE": 1},
    "pairs": [{"pair": "GOLDM/GOLDTEN", "status": "QUIET", "z": -0.43, "reason": "inside +/-2",
               "binding_gate": None, "telegram": "x"},
              {"pair": "GOLDM/GOLDPETAL", "status": "NO_TRADE", "z": 2.4, "reason": "runway",
               "binding_gate": "runway"}],
    "pair_history": {"GOLDM/GOLDTEN": [{"date": "2026-10-26", "z": -0.43}]},
    "decision_log": [{"date": "2026-10-26", "pair": "GOLDM/GOLDPETAL", "status": "NO_TRADE"},
                     {"date": "2026-10-20", "pair": "GOLDM/GOLDTEN", "status": "TRADED"}],
    "backtest": {"performance": {"n_trades": 146, "mean_net_bp": -5.3},
                 "verdict": {"headline": "No edge survives costs", "edge_survives_costs": False}},
    "research": {"by_year": [{"year": 2025, "n": 40}], "by_pair": []},
    "account": {"halted": True, "drawdown_pct": 0.12},
    "published_margins": {"available": True, "latest": {"GOLDM": 9.25},
                          "history": [{"date": "2026-02-06", "GOLDM": 27.83}]},
}


def tool(name):
    return {f.__name__: f for f in assistant.tools_for(SNAP)}[name]


# ------------------------------------------------------------------ tools

def test_pair_lookup_accepts_any_spelling_and_hides_internal_fields():
    for spelling in ("GOLDM/GOLDTEN", "goldm-goldten", "GOLDTEN/GOLDM"):
        p = tool("get_pair")(spelling)
        assert p["pair"] == "GOLDM/GOLDTEN" and "telegram" not in p
    assert "unknown pair" in tool("get_pair")("SILVER/GOLDM")["error"]


def test_decisions_are_newest_first_and_filter_by_pair():
    assert tool("get_decisions")(5)[0]["date"] == "2026-10-26"
    assert [d["pair"] for d in tool("get_decisions")(5, "goldten-goldm")] == ["GOLDM/GOLDTEN"]


def test_overview_and_margins_come_from_the_snapshot():
    ov = tool("get_overview")()
    assert ov["asof"] == "2026-10-26" and ov["verdict"]["edge_survives_costs"] is False
    m = tool("get_margins")()
    assert m["latest"]["GOLDM"] == 9.25 and m["recent_history"][0]["GOLDM"] == 27.83


# ------------------------------------------------------------------ the model call

class FakeChats:
    """Stands in for client.chats: create(...) then send_message(question)."""
    def __init__(self, reply="As of 26 Oct 2026, GOLDM/GOLDTEN is QUIET.", call_tool=None, fail=None):
        self.reply, self.call_tool, self.fail, self.calls = reply, call_tool, fail, []

    def create(self, model, config, history):
        outer = self

        class Chat:
            def send_message(self, message):
                outer.calls.append({"model": model, "config": config,
                                    "contents": list(history) + [message]})
                if outer.fail:
                    raise outer.fail
                if outer.call_tool:   # what the SDK's automatic function calling would do
                    {f.__name__: f for f in config.tools}[outer.call_tool]()
                return type("R", (), {"text": outer.reply})()
        return Chat()


class FakeClient:
    def __init__(self, **kw): self.models = self.chats = FakeChats(**kw)


def test_answer_passes_rules_tools_and_history(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    c = FakeClient(call_tool="get_pairs")
    out = assistant.answer("Why is GOLDM/GOLDTEN quiet?",
                           [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}], SNAP, client=c)
    call = c.models.calls[0]
    assert "Answer only from the tools" in call["config"].system_instruction
    assert {f.__name__ for f in call["config"].tools} >= {"get_pair", "get_decisions", "get_margins"}
    assert [x.role for x in call["contents"][:-1]] == ["user", "model"]
    assert call["contents"][-1] == "Why is GOLDM/GOLDTEN quiet?"
    assert out["tools"] == ["get_pairs"] and out["asof"] == "2026-10-26"


def test_bad_questions_are_refused_before_calling_the_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    c = FakeClient()
    for q in ("", "   ", "x" * 1001):
        with pytest.raises(ValueError):
            assistant.answer(q, [], SNAP, client=c)
    assert c.models.calls == []


def test_a_disabled_google_api_is_explained(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    err = RuntimeError("403 PERMISSION_DENIED ... SERVICE_DISABLED ... has not been used in project 1")
    with pytest.raises(assistant.ChatUnavailable, match="not enabled"):
        assistant.answer("hi", [], SNAP, client=FakeClient(fail=err))


def test_no_key_means_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(assistant.ChatUnavailable):
        assistant.answer("hi", [], SNAP)


# ------------------------------------------------------------------ daily limit

def test_daily_limit_and_give_back():
    assistant._USE.clear()
    assert assistant.take_question("u1", limit=2) == 1
    assert assistant.take_question("u1", limit=2) == 0
    with pytest.raises(assistant.ChatLimit):
        assistant.take_question("u1", limit=2)
    assistant.give_back("u1")
    assert assistant.take_question("u1", limit=2) == 0
    assert assistant.take_question("u2", limit=2) == 1          # per user


# ------------------------------------------------------------------ endpoint

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    monkeypatch.setattr(supa, "user_from_token",
                        lambda tok, session=None: {"id": "u1", "email": "me@example.com"} if tok == "good" else None)
    from api import main
    monkeypatch.setattr(main.store, "get", lambda: SNAP)
    assistant._USE.clear()
    return TestClient(main.app)


GOOD = {"Authorization": "Bearer good"}


def test_chat_needs_sign_in(client):
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_chat_answers_and_counts(client, monkeypatch):
    monkeypatch.setattr(assistant, "answer", lambda q, h, s: {"text": "ok " + q, "tools": [], "model": "m", "asof": "2026-10-26"})
    r = client.post("/api/chat", json={"message": "hi"}, headers=GOOD)
    assert r.status_code == 200 and r.json()["text"] == "ok hi" and r.json()["questions_left"] == 39
    assert client.get("/api/config").json()["chat_enabled"] is True


def test_model_outage_is_503_and_does_not_use_a_question(client, monkeypatch):
    def down(q, h, s):
        raise assistant.ChatUnavailable("The assistant could not reach Gemini.")
    monkeypatch.setattr(assistant, "answer", down)
    r = client.post("/api/chat", json={"message": "hi"}, headers=GOOD)
    assert r.status_code == 503 and "Gemini" in r.json()["detail"]
    assert assistant._USE == {k: 0 for k in assistant._USE}


def test_over_the_limit_is_429(client, monkeypatch):
    monkeypatch.setenv("CHAT_DAILY_LIMIT", "1")
    monkeypatch.setattr(assistant, "answer", lambda q, h, s: {"text": "ok", "tools": [], "model": "m", "asof": None})
    assert client.post("/api/chat", json={"message": "a"}, headers=GOOD).status_code == 200
    assert client.post("/api/chat", json={"message": "b"}, headers=GOOD).status_code == 429


def test_overload_and_timeout_read_as_busy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    for err in ("503 UNAVAILABLE. This model is currently experiencing high demand.",
                "504 DEADLINE_EXCEEDED. Deadline expired before operation could complete."):
        with pytest.raises(assistant.ChatUnavailable, match="busy"):
            assistant.answer("hi", [], SNAP, client=FakeClient(fail=RuntimeError(err)))


def test_ai_studio_is_the_default_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.looks-like-vertex-but-is-ai-studio")
    monkeypatch.delenv("GEMINI_USE_VERTEX", raising=False)
    assert assistant._client()._api_client.vertexai is False
    monkeypatch.setenv("GEMINI_USE_VERTEX", "1")
    assert assistant._client()._api_client.vertexai is True


def test_margin_highs_and_lows_are_computed_not_guessed():
    snap = dict(SNAP, published_margins={"available": True, "history": [
        {"date": "2026-01-29", "GOLDM": 11.8}, {"date": "2026-02-06", "GOLDM": 27.83, "GOLDGUINEA": 28.9},
        {"date": "2026-10-27", "GOLDM": 9.25}, {"date": "2025-06-09", "GOLDM": 7.38}]})
    ex = {f.__name__: f for f in assistant.tools_for(snap)}["get_margins"]()["extremes_by_year"]
    assert ex["2026"]["GOLDM"] == {"high_pct": 27.83, "high_date": "2026-02-06",
                                   "low_pct": 9.25, "low_date": "2026-10-27"}
    assert ex["2025"]["GOLDM"]["high_pct"] == 7.38 and ex["2026"]["GOLDGUINEA"]["high_pct"] == 28.9
