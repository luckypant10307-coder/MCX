"""
Watch for MCX / MCXCCL circulars that change this model's inputs: margins on gold
futures, contract-specification revisions, new gold contracts.

How: TinyFish web search (https://api.search.tinyfish.ai), the same public search a
person would run. It returns titles, links and snippets from public pages -- broker
bulletins, regulatory trackers, news. It does not fetch mcxccl.com or mcxindia.com,
which refuse scripted access; a result may LINK to them, for a person to open.

The API key comes from the environment (TINYFISH_API_KEY), never from source. With
no key the check is skipped and says so; a search failure never fails the nightly run.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

SEARCH_URL = "https://api.search.tinyfish.ai"

# Each query targets one kind of change the model cares about.
QUERIES = {
    "margin": "MCX MCXCCL additional margin gold futures circular",
    "margin_withdrawal": "MCX withdrawal of additional margin gold circular",
    "specification": "MCX gold contract specification modification circular GOLDGUINEA GOLDPETAL GOLDM GOLDTEN",
    "launch": "MCX new gold futures contract launch circular",
}
# A result is kept only if it is about gold AND about something that moves an input.
_GOLD = re.compile(r"\bgold|goldm|goldten|goldguinea|goldpetal|bullion\b", re.I)
_TOPIC = re.compile(r"margin|circular|specification|modif|launch|introduc|tender|delivery|\bELM\b|\bSPAN\b", re.I)
# Live-price and product pages mention "contract" and "margin" on every visit and are
# never news about a change.
_NOISE = re.compile(r"price chart|live .*(rate|price)|price today|futures price|stock price|quote", re.I)


def _search(query: str, api_key: str, session=None, timeout: float = 30.0) -> list[dict]:
    if session is None:
        import requests
        session = requests
    r = session.get(SEARCH_URL, params={"query": query, "location": "IN", "language": "en"},
                    headers={"X-API-Key": api_key}, timeout=timeout)
    r.raise_for_status()
    return (r.json() or {}).get("results", []) or []


def _relevant(item: dict) -> bool:
    text = " ".join(str(item.get(k) or "") for k in ("title", "snippet", "url"))
    return bool(_GOLD.search(text) and _TOPIC.search(text)
                and not _NOISE.search(str(item.get("title") or "")))


def check(api_key: str | None = None, seen_path: str | None = None, session=None) -> dict:
    """
    Run every query; return deduplicated, relevant results, each marked `new` when its
    link was not seen on an earlier run. Never raises.
    """
    api_key = api_key or os.environ.get("TINYFISH_API_KEY")
    checked_at = datetime.now(timezone.utc).isoformat()
    if not api_key:
        return {"available": False, "reason": "TINYFISH_API_KEY is not set", "checked_at": checked_at}

    seen: set = set()
    if seen_path and os.path.isfile(seen_path):
        try:
            with open(seen_path, encoding="utf-8") as fh:
                seen = set(json.load(fh))
        except (OSError, ValueError):
            seen = set()

    items, errors, by_url = [], [], {}
    for topic, q in QUERIES.items():
        try:
            results = _search(q, api_key, session=session)
        except Exception as exc:              # one failed query must not lose the rest
            errors.append(f"{topic}: {type(exc).__name__}: {exc}")
            continue
        for r in results:
            url = r.get("url")
            if not url or not _relevant(r):
                continue
            if url in by_url:
                by_url[url]["topics"].append(topic)
                continue
            by_url[url] = {"title": r.get("title"), "url": url, "site": r.get("site_name"),
                           "snippet": r.get("snippet"), "date": r.get("date"),
                           "topics": [topic], "new": url not in seen}
            items.append(by_url[url])

    if seen_path and items:
        os.makedirs(os.path.dirname(os.path.abspath(seen_path)), exist_ok=True)
        with open(seen_path, "w", encoding="utf-8") as fh:
            json.dump(sorted(seen | set(by_url)), fh)

    items.sort(key=lambda x: (not x["new"], x.get("date") or ""), reverse=False)
    return {"available": True, "checked_at": checked_at, "items": items,
            "new": sum(1 for x in items if x["new"]), "errors": errors,
            "source": "TinyFish web search"}


def message(result: dict, limit: int = 5) -> str | None:
    """Telegram text for newly seen circulars, or None when there is nothing new."""
    new = [x for x in result.get("items", []) if x.get("new")]
    if not new:
        return None
    lines = ["MCX circular watch: {} new result(s)".format(len(new))]
    for x in new[:limit]:
        lines.append("- {} ({})\n  {}".format(x.get("title") or "untitled", ", ".join(x["topics"]), x["url"]))
    if len(new) > limit:
        lines.append("...and {} more on the dashboard".format(len(new) - limit))
    return "\n".join(lines)
