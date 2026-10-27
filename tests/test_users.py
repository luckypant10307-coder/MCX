"""
Accounts and per-user notifications.

The properties that matter: with accounts on, no market data leaves the API without a
valid sign-in; each user hears only what they asked for; a dead device is dropped
rather than retried forever; and none of it can break the nightly run.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mcxrv import supa, user_notify

URL = "https://example.supabase.co"


# ------------------------------------------------------------------ login gate

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    monkeypatch.setattr(supa, "user_from_token",
                        lambda tok, session=None: {"id": "u1", "email": "me@example.com"} if tok == "good" else None)
    from api import main
    return TestClient(main.app)


def test_config_and_health_stay_public(client):
    assert client.get("/api/health").status_code in (200, 404)     # 404 only if no snapshot exists
    cfg = client.get("/api/config").json()
    assert cfg["auth_enabled"] is True and cfg["supabase_url"] == URL
    assert "secret" not in str(cfg).lower()


def test_data_needs_a_signed_in_user(client):
    assert client.get("/api/overview").status_code == 401
    assert client.get("/api/overview", headers={"Authorization": "Bearer bad"}).status_code == 401
    me = client.get("/api/me", headers={"Authorization": "Bearer good"})
    assert me.status_code == 200 and me.json()["email"] == "me@example.com"


def test_without_supabase_the_dashboard_stays_open(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    from api import main
    c = TestClient(main.app)
    assert c.get("/api/config").json()["auth_enabled"] is False
    assert c.get("/api/overview").status_code in (200, 404)


# ------------------------------------------------------------------ fan-out

class _Resp:
    def __init__(self, data=None, status=200): self._d, self.status_code = data, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self): return self._d


class _DB:
    """Stands in for Supabase's REST API: two users, one device each, one gone."""
    def __init__(self, prefs):
        self.prefs, self.inserted, self.deleted = prefs, [], []
        self.subs = [{"id": 1, "user_id": "a", "endpoint": "https://push/a", "p256dh": "k", "auth": "s"},
                     {"id": 2, "user_id": "b", "endpoint": "https://push/b-gone", "p256dh": "k", "auth": "s"}]

    def get(self, url, params=None, headers=None, timeout=None):
        return _Resp(self.prefs if url.endswith("notification_prefs") else self.subs)

    def post(self, url, json=None, params=None, headers=None, timeout=None):
        self.inserted.extend(json)
        return _Resp(json, 201)

    def delete(self, url, params=None, headers=None, timeout=None):
        self.deleted.append(params)
        return _Resp(None, 204)


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test")
    monkeypatch.setenv("VAPID_PRIVATE_KEY", "x")
    # device "b" was unsubscribed in the browser: the push service answers 410 Gone.
    monkeypatch.setattr(user_notify, "_push", lambda sub, payload, session=None: 410 if "gone" in sub["endpoint"] else 201)


ALERT = {"kind": "new_alert", "body": "GOLDM/GOLDTEN alert", "pair": "GOLDM/GOLDTEN", "url": "/pairs/GOLDM-GOLDTEN"}
DIGEST = {"kind": "digest", "body": "summary", "url": "/decisions"}


def test_each_user_gets_only_what_they_asked_for(configured):
    db = _DB([{"user_id": "a", "trade_alerts": True, "digest": False, "circulars": True, "pairs": None},
              {"user_id": "b", "trade_alerts": True, "digest": True, "circulars": True, "pairs": ["GOLDTEN/GOLDPETAL"]}])
    out = user_notify.fanout([ALERT, DIGEST], run_id="r1", session=db)
    got = {(r["user_id"], r["kind"]) for r in db.inserted}
    # a: alert yes, digest off. b: follows another pair, so no alert; digest yes.
    assert got == {("a", "new_alert"), ("b", "digest")}
    assert out["notifications"] == 2 and out["pushed"] == 1


def test_a_device_the_push_service_says_is_gone_is_removed(configured):
    db = _DB([{"user_id": "b", "trade_alerts": True, "digest": True, "circulars": True, "pairs": None}])
    out = user_notify.fanout([DIGEST], session=db)
    assert out["devices_removed"] == 1 and db.deleted == [{"id": "eq.2"}]


def test_fanout_is_a_no_op_without_supabase(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    out = user_notify.fanout([ALERT])
    assert out["configured"] is False and out["notifications"] == 0


def test_a_database_failure_is_reported_not_raised(configured):
    class Down:
        def get(self, *a, **k): raise ConnectionError("supabase down")
    out = user_notify.fanout([ALERT], session=Down())
    assert out["errors"] and out["notifications"] == 0


def test_events_open_the_right_page():
    evs = user_notify.events_from_run([{"event": "new_alert", "pair": "GOLDM/GOLDTEN", "text": "t"}], "digest", "circ")
    assert [e["url"] for e in evs] == ["/pairs/GOLDM-GOLDTEN", "/decisions", "/data"]


# ------------------------------------------------------------------ token check cache

def test_token_check_is_cached(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "p")
    supa._TOKEN_CACHE.clear()
    calls = []

    class S:
        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            return _Resp({"id": "u1"}, 200)
    assert supa.user_from_token("tok", session=S())["id"] == "u1"
    assert supa.user_from_token("tok", session=S())["id"] == "u1"
    assert len(calls) == 1
