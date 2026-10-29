"""
Per-user notifications: every registered user gets the events they asked for, on
every device they enabled, plus a copy in the in-site notification centre.

The nightly job decides WHAT happened (notify.changes, notify.digest, circulars);
this module decides WHO hears about it and delivers it:

  * notification_prefs  trade_alerts / digest / circulars on or off, and which pairs
  * notifications       one row per user per event: the bell list in the site
  * push_subscriptions  each browser or phone that turned notifications on; a device
                        the push service says is gone (404/410) is removed

Runs only when Supabase is configured (supa.configured()). Never raises: a delivery
problem is reported, not allowed to fail the nightly run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from mcxrv import supa
from mcxrv.notify import PUSH_STYLE

# Event kind -> the preference that controls it.
PREF_FOR = {"new_alert": "trade_alerts", "alert_cleared": "trade_alerts", "digest": "digest",
            "circulars": "circulars", "no_data": "trade_alerts",
            "data_changed_same_session": "trade_alerts", "test": None}


def _wants(pref: dict, kind: str, pair: str | None) -> bool:
    key = PREF_FOR.get(kind)
    if key and not pref.get(key, True):
        return False
    pairs = pref.get("pairs")
    return not (pair and pairs and pair not in pairs)


def _push(sub: dict, payload: dict, session=None) -> int | None:
    """Send one web push; returns the HTTP status (None if it could not be attempted)."""
    if not os.environ.get("VAPID_PRIVATE_KEY"):
        return None
    from pywebpush import WebPushException, webpush
    try:
        r = webpush(subscription_info={"endpoint": sub["endpoint"],
                                       "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]}},
                    data=json.dumps(payload),
                    vapid_private_key=os.environ["VAPID_PRIVATE_KEY"],
                    vapid_claims={"sub": os.environ.get("VAPID_SUBJECT", "mailto:admin@example.com")},
                    ttl=24 * 3600, requests_session=session)
        return getattr(r, "status_code", 201)
    except WebPushException as exc:
        return getattr(getattr(exc, "response", None), "status_code", 0) or 0
    except Exception:
        return 0


def fanout(events: list[dict], run_id: str = "", only_user: str | None = None,
           session=None, push_session=None) -> dict:
    """
    Deliver events to users. Each event: {kind, title?, body, pair?, url?}.
    Returns counts per outcome; never raises.
    """
    out = {"configured": supa.configured(), "users": 0, "notifications": 0,
           "pushed": 0, "push_failed": 0, "devices_removed": 0, "errors": []}
    if not out["configured"] or not events:
        return out
    try:
        params = {"select": "*"}
        if only_user:
            params["user_id"] = f"eq.{only_user}"
        prefs = supa.select("notification_prefs", params, session=session)
        subs = supa.select("push_subscriptions", params, session=session)
    except Exception as exc:
        out["errors"].append(f"read users: {type(exc).__name__}: {exc}")
        return out

    devices: dict = {}
    for s in subs:
        devices.setdefault(s["user_id"], []).append(s)
    out["users"] = len(prefs)

    rows = []
    delivered: set = set()
    for pref in prefs:
        uid = pref["user_id"]
        for ev in events:
            if not _wants(pref, ev["kind"], ev.get("pair")):
                continue
            title = ev.get("title") or PUSH_STYLE.get(ev["kind"], ("Gold Spread Monitor",))[0]
            url = ev.get("url") or "/"
            rows.append({"user_id": uid, "kind": ev["kind"], "title": title,
                         "body": ev["body"], "url": url, "run_id": run_id})
            for dev in devices.get(uid, []):
                status = _push(dev, {"title": title, "body": ev["body"], "url": url,
                                     "kind": ev["kind"], "tag": ev["kind"] + ":" + (ev.get("pair") or "")},
                               session=push_session)
                if status in (200, 201, 202):
                    out["pushed"] += 1
                    delivered.add(dev["id"])
                elif status in (404, 410):          # the browser unsubscribed or was reset
                    try:
                        supa.delete("push_subscriptions", {"id": dev["id"]}, session=session)
                        out["devices_removed"] += 1
                    except Exception as exc:
                        out["errors"].append(f"remove device: {exc}")
                elif status is not None:
                    out["push_failed"] += 1
    # When each device last took a push: a phone that stops answering shows up here
    # long before anyone notices the missing alerts.
    now = datetime.now(timezone.utc).isoformat()
    for dev_id in sorted(delivered):
        try:
            supa.update("push_subscriptions", {"id": dev_id}, {"last_ok_at": now}, session=session)
        except Exception as exc:
            out["errors"].append(f"mark device: {exc}")
    if rows:
        try:
            supa.insert("notifications", rows, session=session)
            out["notifications"] = len(rows)
        except Exception as exc:
            out["errors"].append(f"write notifications: {type(exc).__name__}: {exc}")
    return out


def events_from_run(alert_events: list[dict], digest_text: str | None, circular_text: str | None) -> list[dict]:
    """Shape the nightly job's messages into fan-out events, with where each should open."""
    evs = []
    for m in alert_events:
        pair = m.get("pair")
        evs.append({"kind": m["event"], "body": m["text"], "pair": pair,
                    "url": "/pairs/" + pair.replace("/", "-") if pair else "/"})
    if digest_text:
        evs.append({"kind": "digest", "body": digest_text, "url": "/decisions"})
    if circular_text:
        evs.append({"kind": "circulars", "body": circular_text, "url": "/data"})
    return evs
