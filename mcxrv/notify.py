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


NTFY_DEFAULT_SERVER = "https://ntfy.sh"

# How each event shows up as a phone push: title, priority (1-5) and ntfy emoji tags.
PUSH_STYLE = {
    "new_alert": ("Gold spread ALERT", "high", "rotating_light"),
    "alert_cleared": ("Gold spread alert cleared", "default", "white_check_mark"),
    "no_data": ("Gold monitor: no usable data", "high", "warning"),
    "data_changed_same_session": ("Gold monitor: data changed", "default", "warning"),
    "circulars": ("MCX circular watch", "default", "page_facing_up"),
    # Unusual but not actionable: delivered silently so it never competes with an alert.
    "digest": ("Gold monitor: evening summary", "low", "memo"),
}

# What a data-anomaly feature means, in the words a phone notification can carry.
_FEATURE_WORDS = {
    "ret_in_ticks": "price jump", "ret_vs_peers_bp": "moved unlike its peers",
    "peer_dev_bp": "far from its peers", "log_ret": "large daily move", "range_frac": "wide range",
    "close_in_range": "odd close", "d_log_volume": "volume swing", "d_log_oi": "open-interest swing",
    "resid_bp": "large model residual",
}


def digest(prev: dict | None, new: dict, max_lines: int = 8) -> str | None:
    """
    One quiet evening summary of what was unusual but NOT an alert: pairs that crossed
    the entry level and were refused (with the reason), and data anomalies flagged on
    the newest session. None when there is nothing to say, or when this session was
    already summarised (a re-run of the same data never repeats it).
    """
    meta = new.get("meta", {})
    old = (prev or {}).get("meta", {})
    if old and old.get("asof") == meta.get("asof") and \
            old.get("data_fingerprint") == meta.get("data_fingerprint"):
        return None

    refused = [p for p in new.get("pairs", []) if p.get("status") == "NO_TRADE"]
    an = new.get("anomalies") or {}
    flags = [a for a in (an.get("latest") or []) if str(a.get("date")) == str(meta.get("asof"))]
    if not refused and not flags:
        return None

    tag = " (sample data)" if meta.get("sample_data") else ""
    lines = ["Evening summary{} - {}".format(tag, meta.get("asof"))]
    if refused:
        lines.append("Unusual, not traded ({}):".format(len(refused)))
        for p in refused[:max_lines]:
            lines.append("- {} z {}: {}".format(p["pair"], _fmt_num(p.get("z"), "{:+.2f}"), p.get("reason")))
    if flags:
        lines.append("Data to review ({}):".format(len(flags)))
        for a in flags[:max_lines]:
            lines.append("- {} {}".format(str(a.get("contract_id", "")).replace("|", " "),
                                          _FEATURE_WORDS.get(a.get("top_feature"), a.get("top_feature"))))
    acct = new.get("account") or {}
    if acct.get("halted"):
        lines.append("Kill switch ON: account {:.1f}% below its peak; no new trades.".format(
            100 * (acct.get("drawdown_pct") or 0)))
    lines.append("run {}".format(meta.get("run_id", "")))
    return "\n".join(lines)


def send_ntfy(text: str, title: str | None = None, priority: str | None = None,
              tags: str | None = None, topic: str | None = None, server: str | None = None,
              token: str | None = None, session=None, timeout: float = 10.0) -> dict:
    """
    Push one message through ntfy (https://ntfy.sh, or a self-hosted server).

    NTFY_TOPIC is the channel: anyone who knows the name can read it, so pick a long
    random one and treat it like a password. NTFY_TOKEN is only for protected topics or
    a private server. Returns a status dict; never raises.
    """
    topic = topic or os.environ.get("NTFY_TOPIC")
    server = (server or os.environ.get("NTFY_SERVER") or NTFY_DEFAULT_SERVER).rstrip("/")
    token = token or os.environ.get("NTFY_TOKEN")
    if not topic:
        return {"sent": False, "reason": "NTFY_TOPIC not set"}
    headers = {}
    if title:
        headers["Title"] = title
    if priority:
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = tags
    if token:
        headers["Authorization"] = "Bearer " + token
    if session is None:
        import requests
        session = requests
    try:
        r = session.post("{}/{}".format(server, topic), data=text.encode("utf-8"),
                         headers=headers, timeout=timeout)
        ok = getattr(r, "status_code", 0) == 200
        return {"sent": ok, "status": getattr(r, "status_code", None)}
    except Exception as exc:     # delivery must never break the nightly run
        return {"sent": False, "reason": "{}: {}".format(type(exc).__name__, exc)}


def deliver(text: str, kind: str | None = None, ntfy_kw: dict | None = None, **send_kw) -> dict:
    """
    Send one message to every configured channel (Telegram, ntfy). The top-level
    "sent" is True when at least one channel accepted it; each channel's own result
    is kept under its name. Telegram's fields stay at the top level as before.
    """
    title, priority, tags = PUSH_STYLE.get(kind or "", (None, None, None))
    tg = send(text, **send_kw)
    nt = send_ntfy(text, title=title, priority=priority, tags=tags, **(ntfy_kw or {}))
    return {**tg, "sent": bool(tg.get("sent") or nt.get("sent")), "telegram": tg, "ntfy": nt}


def notify(prev: dict | None, new: dict, ntfy_kw: dict | None = None, **send_kw) -> list[dict]:
    """Diff, format and send to every configured channel. Returns what was sent."""
    out = []
    for ev in changes(prev, new):
        text = format_message(ev, new.get("meta", {}))
        out.append({"event": ev["kind"], "pair": ev.get("pair"), "text": text,
                    **deliver(text, kind=ev["kind"], ntfy_kw=ntfy_kw, **send_kw)})
    return out
