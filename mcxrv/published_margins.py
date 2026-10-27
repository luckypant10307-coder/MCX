"""
Published MCX margin RATES, as brokers republish them from MCXCCL each day.

This is not the SPAN risk-parameter file. It carries the outcome of SPAN for one
outright lot -- the percentage of contract value blocked -- and, in the Aditya Birla
Money sheet, its parts (initial, additional, special, ELM, tender, delivery). It does
NOT carry the calendar or inter-commodity spread credits; those exist only in the
MCXCCL file, so the credit stays an assumption however many of these are held.

Two sources, both public pages with no bot protection:

  zerodha   https://zerodha.com/margin-calculator/Commodity/   today only, total %
  abml      .../SPANMarginPDF/Mcx Span_DDMMYY.pdf              a dated archive with
            gaps, full breakdown; parsed from the PDF text (needs `pypdf`)

Store: one JSON file, {"YYYY-MM-DD": {"source", "fetched_at", "contracts": [...]}}.
Every value is kept as published; the conversion to a scan range happens at use.
"""
from __future__ import annotations

import io
import json
import os
import re
import statistics
from datetime import date, datetime, timedelta, timezone
from html import unescape

GOLD = ("GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL")
DEFAULT_PATH = "data/reference/published_margins.json"
ZERODHA_URL = "https://zerodha.com/margin-calculator/Commodity/"
ABML_URL = ("https://stocksandsecurities.adityabirlacapital.com/pdf/SPANMarginPDF/"
            "Mcx%20Span_{:%d%m%y}.pdf")
UA = {"User-Agent": "mcx-gold-rv research (daily margin check)"}

# A published rate holds until the next revision, but not forever: past this many
# days without a newer sheet, the session falls back to the volatility estimate
# rather than wearing a stale number as if it were published.
MAX_CARRY_DAYS = 14
# Smallest move in the total rate (percentage points) worth telling anyone about.
CHANGE_ALERT_PP = 0.25


# ------------------------------------------------------------------ parsing

_ROW = re.compile(r"<tr\b([^>]*\bdata-scrip=\"[^\"]+\"[^>]*)>", re.S)
_ATTR = re.compile(r"data-([a-z_]+)=\"([^\"]*)\"")
_UPDATED = re.compile(r"Last updated:\s*(\d{1,2} \w{3} \d{4})")


def parse_zerodha(html: str, symbols=GOLD) -> dict:
    """Zerodha's commodity margin table -> {"date", "contracts"}. Raises on no rows."""
    m = _UPDATED.search(html)
    if not m:
        raise ValueError("zerodha: no 'Last updated' date on the page")
    asof = datetime.strptime(m.group(1), "%d %b %Y").date()
    out = []
    for row in _ROW.finditer(html):
        a = {k: unescape(v) for k, v in _ATTR.findall(row.group(1))}
        sym = a.get("scrip", "").strip().upper()
        if symbols and sym not in symbols:
            continue
        try:
            out.append({"symbol": sym, "expiry": a.get("expiry"),
                        "total_pct": float(a["margin"]),
                        "margin_inr": float(a["nrml_margin"]),
                        "price": float(a["price"])})
        except (KeyError, ValueError):
            continue
    if not out:
        raise ValueError("zerodha: no {} rows in the table".format("/".join(symbols)))
    return {"date": asof, "contracts": out}


_ABML_DATE = re.compile(r"\b(\d{2}-[A-Z][a-z]{2}-\d{2})\b")
_ABML_ROW = re.compile(r"^\s*\d+\s+MCX\s+([A-Z0-9]+)\s+(\d{2}-[A-Z][a-z]{2}-\d{2})\s+(.*)$")
_NUM = re.compile(r"^-?[\d,]+(?:\.\d+)?$")


def parse_abml(text: str, symbols=GOLD) -> dict:
    """
    Text of an Aditya Birla Money MCX margin sheet -> {"date", "contracts"}.

    Each row ends in the same 15 numbers: initial, tender, delivery, additional L/S,
    special L/S, ELM L/S, daily vol, annual vol, total % L/S, total INR L/S. The
    lot-size and quotation columns before them vary in shape, so the row is read
    from the right.
    """
    contracts = []
    for line in text.splitlines():
        m = _ABML_ROW.match(line)
        if not m:
            continue
        sym = m.group(1)
        if symbols and sym not in symbols:
            continue
        toks = m.group(3).split()
        tail = toks[-15:]
        if len(tail) < 15 or not all(_NUM.match(t) for t in tail):
            continue
        n = [float(t.replace(",", "")) for t in tail]
        contracts.append({
            "symbol": sym,
            "expiry": datetime.strptime(m.group(2), "%d-%b-%y").date().isoformat(),
            "initial_pct": n[0], "tender_pct": n[1], "delivery_pct": n[2],
            "additional_pct": max(n[3], n[4]), "special_pct": max(n[5], n[6]),
            "elm_pct": max(n[7], n[8]), "total_pct": max(n[11], n[12]),
            "margin_inr": max(n[13], n[14])})
    # The sheet's date sits on a line of its own, near the footer.
    stamp = None
    for line in text.splitlines():
        if _ABML_DATE.fullmatch(line.strip()):
            stamp = datetime.strptime(line.strip(), "%d-%b-%y").date()
    if not contracts:
        raise ValueError("abml: no {} rows in the sheet".format("/".join(symbols)))
    if stamp is None:
        raise ValueError("abml: sheet date not found")
    return {"date": stamp, "contracts": contracts}


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader          # optional: only the archive backfill needs it
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


# ------------------------------------------------------------------ fetching

def fetch_zerodha(session=None, timeout: float = 20.0) -> dict:
    import requests
    r = (session or requests).get(ZERODHA_URL, headers=UA, timeout=timeout)
    r.raise_for_status()
    return parse_zerodha(r.text)


def fetch_abml(day: date, session=None, timeout: float = 30.0) -> dict | None:
    """One dated sheet, or None when the archive has no PDF for that day."""
    import requests
    r = (session or requests).get(ABML_URL.format(day), headers=UA, timeout=timeout,
                                  stream=True)
    try:
        head = next(r.iter_content(8), b"") if r.status_code == 200 else b""
        if not head.startswith(b"%PDF"):
            return None                   # missing days are served as a ~240 KB HTML page
        data = head + r.content
    finally:
        r.close()
    sheet = parse_abml(pdf_text(data))
    if sheet["date"] != day:
        raise ValueError("abml: file for {} is stamped {}".format(day, sheet["date"]))
    return sheet


# ------------------------------------------------------------------ store

def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(store: dict, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(store.items())), fh, indent=1)
    os.replace(tmp, path)


def add(store: dict, sheet: dict, source: str) -> bool:
    """
    Put one sheet in the store. The archive's breakdown is kept over a total-only
    page for the same day, never the other way round. Returns True if it changed.
    """
    key = sheet["date"].isoformat() if isinstance(sheet["date"], date) else str(sheet["date"])
    old = store.get(key)
    if old and old.get("source") == "abml" and source != "abml":
        return False
    entry = {"source": source, "fetched_at": datetime.now(timezone.utc).isoformat(),
             "contracts": sheet["contracts"]}
    if old and old.get("contracts") == entry["contracts"] and old.get("source") == source:
        return False
    store[key] = entry
    return True


# ------------------------------------------------------------------ use

def _day(x) -> date:
    return x.date() if isinstance(x, datetime) else x       # pandas Timestamps too


def symbol_rates(entry: dict) -> dict:
    """
    One rate per symbol for a day: the median total over contracts not in their
    tender or delivery period. Those carry 25-50% delivery margins the backtest
    never pays (it exits before expiry), and Zerodha does not flag them, so the
    median of the listed months is what keeps one such row from setting the rate.
    """
    by: dict = {}
    for c in entry.get("contracts", []):
        if c.get("tender_pct") or c.get("delivery_pct"):
            continue
        by.setdefault(c["symbol"], []).append(float(c["total_pct"]))
    return {s: statistics.median(v) / 100.0 for s, v in by.items() if v}


def schedule(store: dict, sessions, elm_for, additional_for,
             max_carry_days: int = MAX_CARRY_DAYS) -> dict:
    """
    {(session, symbol): scan_pct} from published totals, shaped like
    `margin.volatility_schedule` so it drops in beside it.

    A sheet dated D states the margin blocked during trading on D, so it is known
    before session D opens. Session s uses the newest sheet dated on or before the
    NEXT session (that is when the backtest's t+1 fill happens), and no older than
    `max_carry_days`. The scan range is the published total less the ELM and
    additional margin the engine adds on top, so an outright lot comes out at
    exactly the published rate.
    """
    if not store:
        return {}
    days = sorted((date.fromisoformat(k), symbol_rates(v)) for k, v in store.items())
    sess = sorted(set(sessions))
    out: dict = {}
    j = -1
    for i, s in enumerate(sess):
        horizon = _day(sess[i + 1]) if i + 1 < len(sess) else _day(s) + timedelta(days=4)
        while j + 1 < len(days) and days[j + 1][0] <= horizon:
            j += 1
        if j < 0 or (horizon - days[j][0]).days > max_carry_days:
            continue
        for sym, total in days[j][1].items():
            out[(s, sym)] = max(total - elm_for(sym) - additional_for(sym), 0.0)
    return out


def latest_change(store: dict, threshold_pp: float = CHANGE_ALERT_PP) -> list[dict]:
    """Symbols whose rate moved between the two newest sheets, for an alert."""
    keys = sorted(store)
    if len(keys) < 2:
        return []
    a, b = symbol_rates(store[keys[-2]]), symbol_rates(store[keys[-1]])
    out = []
    for sym in sorted(set(a) & set(b)):
        if abs(b[sym] - a[sym]) * 100 >= threshold_pp:
            out.append({"symbol": sym, "from_pct": a[sym] * 100, "to_pct": b[sym] * 100,
                        "from_date": keys[-2], "to_date": keys[-1]})
    return out


def message(changes: list[dict]) -> str | None:
    if not changes:
        return None
    lines = ["MCX margin change ({} -> {}):".format(changes[0]["from_date"], changes[0]["to_date"])]
    for c in changes:
        lines.append("  {}  {:.2f}% -> {:.2f}%".format(c["symbol"], c["from_pct"], c["to_pct"]))
    lines.append("Capital per spread changes with it; the dashboard uses the new rate.")
    return "\n".join(lines)


def status(store: dict, sessions=None) -> dict:
    """What the dashboard shows: the newest rates, history size and coverage."""
    if not store:
        return {"available": False, "days": 0,
                "reason": "no published margin sheets stored yet"}
    keys = sorted(store)
    last = store[keys[-1]]
    out = {
        "available": True, "days": len(keys), "first": keys[0], "last": keys[-1],
        "last_source": last.get("source"),
        "latest": {s: round(r * 100, 4) for s, r in symbol_rates(last).items()},
        "contracts": last.get("contracts", []),
        "sources": sorted({v.get("source") for v in store.values()}),
        "changes": latest_change(store),
        "note": ("Outright margin rates as brokers republish them from MCXCCL. "
                 "Spread credits are not in these sheets and remain assumed."),
    }
    if sessions is not None:
        out["history"] = [{"date": k, **{s: round(r * 100, 4)
                                         for s, r in symbol_rates(store[k]).items()}}
                          for k in keys]
    return out
