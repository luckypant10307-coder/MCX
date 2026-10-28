"""
Server-side access to Supabase: the user database (PostgREST) and token checks (Auth).

Uses SUPABASE_URL with SUPABASE_SECRET_KEY, which bypasses row-level security. It is
for the server and the nightly job ONLY -- the browser uses the publishable key and is
held to RLS. Nothing here runs when SUPABASE_URL is unset, so the project still works
as a single-user tool without accounts.
"""
from __future__ import annotations

import os
import time

TIMEOUT = 15.0


def configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SECRET_KEY"))


def public_config() -> dict:
    """What the browser may know: URL and publishable key (both public by design)."""
    return {"auth_enabled": bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_PUBLISHABLE_KEY")),
            "supabase_url": os.environ.get("SUPABASE_URL"),
            "supabase_publishable_key": os.environ.get("SUPABASE_PUBLISHABLE_KEY"),
            "vapid_public_key": os.environ.get("VAPID_PUBLIC_KEY"),
            "chat_enabled": bool(os.environ.get("GEMINI_API_KEY"))}


def _base() -> str:
    return os.environ["SUPABASE_URL"].rstrip("/")


def _headers(extra: dict | None = None) -> dict:
    key = os.environ["SUPABASE_SECRET_KEY"]
    h = {"apikey": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
    h.update(extra or {})
    return h


def _session(session):
    if session is not None:
        return session
    import requests
    return requests


def select(table: str, params: dict | None = None, session=None) -> list[dict]:
    r = _session(session).get(f"{_base()}/rest/v1/{table}", params=params or {"select": "*"},
                               headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def insert(table: str, rows, session=None, upsert_on: str | None = None) -> list[dict]:
    prefer = "return=representation"
    params = {}
    if upsert_on:
        prefer += ",resolution=merge-duplicates"
        params["on_conflict"] = upsert_on
    r = _session(session).post(f"{_base()}/rest/v1/{table}", json=rows, params=params,
                                headers=_headers({"Prefer": prefer}), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def update(table: str, match: dict, values: dict, session=None) -> None:
    params = {k: f"eq.{v}" for k, v in match.items()}
    r = _session(session).patch(f"{_base()}/rest/v1/{table}", params=params, json=values,
                                 headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()


def delete(table: str, match: dict, session=None) -> None:
    params = {k: f"eq.{v}" for k, v in match.items()}
    r = _session(session).delete(f"{_base()}/rest/v1/{table}", params=params,
                                  headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()


# ---------------------------------------------------------------- access tokens
_TOKEN_CACHE: dict = {}
TOKEN_TTL_S = 60


def user_from_token(token: str, session=None) -> dict | None:
    """
    The user a browser access token belongs to, or None if it is invalid or expired.

    Asks Supabase Auth itself (/auth/v1/user) rather than verifying the JWT locally, so
    it works whichever signing scheme the project uses and honours sign-outs. Results
    are cached for a minute so a page that makes eight API calls checks once.
    """
    if not token or not os.environ.get("SUPABASE_URL"):
        return None
    hit = _TOKEN_CACHE.get(token)
    if hit and hit[0] > time.time():
        return hit[1]
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_SECRET_KEY", "")
    try:
        r = _session(session).get(f"{_base()}/auth/v1/user", timeout=TIMEOUT,
                                  headers={"apikey": key, "Authorization": "Bearer " + token})
    except Exception:
        return None
    user = r.json() if getattr(r, "status_code", 0) == 200 else None
    if len(_TOKEN_CACHE) > 1000:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[token] = (time.time() + TOKEN_TTL_S, user)
    return user
