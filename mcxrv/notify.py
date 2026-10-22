"""
Telegram alerts, sent only when something changes.

The nightly job compares today's snapshot with yesterday's and messages only the
transitions -- a pair becoming an ALERT, an ALERT going away, or a data-quality
problem on the newest file. Re-sending an unchanged state every night trains people
to ignore the channel.

Credentials come from the environment (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). With
either missing, `send` does nothing and says so; the job never fails because a
message could not be delivered.
"""
from __future__ import annotations

import os

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _by_pair(snap: dict | None) -> dict:
    if not snap:
        return {}
    return {p["pair"]: p for p in snap.get("pairs", [])}


def changes(prev: dict | None, new: dict) -> list[dict]:
    """State transitions worth a message. Pure function; tested directly."""
    events = []
    old, cur = _by_pair(prev), _by_pair(new)
    for pair, p in cur.items():
        before = old.get(pair, {}).get("status")
        if p["status"] == "ALERT" and before != "ALERT":
            events.append({"kind": "new_alert", "pair": pair, "pair_state": p})
        elif before == "ALERT" and p["status"] != "ALERT":
            events.append({"kind": "alert_cleared", "pair": pair, "pair_state": p})

    # A new session whose file came back under a different date, or a run that
    # produced no data at all, is worth knowing about the same evening.
    meta_new, meta_old = new.get("meta", {}), (prev or {}).get("meta", {})
    if new.get("summary", {}).get("pairs", 0) == 0:
        events.append({"kind": "no_data", "pair": None, "pair_state": None})
    if meta_old and meta_new.get("asof") == meta_old.get("asof") \
            and meta_new.get("data_fingerprint") != meta_old.get("data_fingerprint"):
        events.append({"kind": "data_changed_same_session", "pair": None,
                       "pair_state": None})
    return events


def _fmt_num(v, fmt):
    return fmt.format(v) if v is not None else "n/a"


def format_message(event: dict, meta: dict) -> str:
    asof = meta.get("asof", "?")
    tag = " (sample data)" if meta.get("sample_data") else ""
    run = meta.get("run_id", "")
    k = event["kind"]
    if k == "new_alert":
        p = event["pair_state"]
        risk = p.get("risk") or {}
        a, b = p["sym_a"], p["sym_b"]
        # direction = +1: spread too low -> buy a, sell b. -1: the reverse.
        buy, sell = (a, b) if p.get("direction", 0) > 0 else (b, a)
        lots = ""
        if risk.get("lots_a") is not None and risk.get("lots_b") is not None:
            la, lb = int(risk["lots_a"]), int(risk["lots_b"])
            lots = " ({} x {}, {} x {})".format(la, a, lb, b)
        return ("GOLD SPREAD ALERT{tag} - {asof}\n"
                "{pair}: extra gap {d} bp, z {z}\n"
                "Buy {buy} / sell {sell}{lots}\n"
                "Margin {m}, buffer {buf}\n"
                "Checks: history, stationarity, half-life, size, risk - all passed\n"
                "run {run}").format(
                    tag=tag, asof=asof, pair=p["pair"],
                    d=_fmt_num(p.get("dislocation_bp"), "{:+.1f}"),
                    z=_fmt_num(p.get("z"), "{:+.2f}"), buy=buy, sell=sell, lots=lots,
                    m=_fmt_num(risk.get("marginal_margin"), "Rs {:,.0f}"),
                    buf=risk.get("buffer_state", "n/a"), run=run)
    if k == "alert_cleared":
        p = event["pair_state"]
        return ("Alert cleared{tag} - {asof}\n{pair} is now {status}: {reason}\nrun {run}"
                .format(tag=tag, asof=asof, pair=p["pair"], status=p["status"],
                        reason=p["reason"], run=run))
    if k == "no_data":
        return ("No usable data{tag} for {asof}. Check the downloaded bhavcopy.\nrun {run}"
                .format(tag=tag, asof=asof, run=run))
    if k == "data_changed_same_session":
        return ("Data for {asof} changed since the last run (new fingerprint).{tag}\n"
                "Re-check any alert sent earlier for this session.\nrun {run}"
                .format(tag=tag, asof=asof, run=run))
    return "{}: {}".format(k, event.get("pair"))


def send(text: str, token: str | None = None, chat_id: str | None = None,
         session=None, timeout: float = 10.0) -> dict:
    """Send one message. Returns a status dict; never raises."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"}
    if session is None:
        import requests
        session = requests
    try:
        r = session.post(TELEGRAM_API.format(token=token),
                         json={"chat_id": chat_id, "text": text,
                               "disable_web_page_preview": True},
                         timeout=timeout)
        ok = getattr(r, "status_code", 0) == 200
        return {"sent": ok, "status": getattr(r, "status_code", None)}
    except Exception as exc:     # delivery must never break the nightly run
        return {"sent": False, "reason": "{}: {}".format(type(exc).__name__, exc)}


def notify(prev: dict | None, new: dict, **send_kw) -> list[dict]:
    """Diff, format and send. Returns what was (or would have been) sent."""
    out = []
    for ev in changes(prev, new):
        text = format_message(ev, new.get("meta", {}))
        out.append({"event": ev["kind"], "pair": ev.get("pair"), "text": text,
                    **send(text, **send_kw)})
    return out
