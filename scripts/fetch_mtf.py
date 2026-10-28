"""
Download MCX futures bhavcopy from mtf.trading into data/raw/, one CSV per session.

    python scripts/fetch_mtf.py                      # everything not already on disk
    python scripts/fetch_mtf.py --start 2025-01-01   # a range
    python scripts/fetch_mtf.py --end 2026-10-23 --start 2026-10-01

WHY THIS SOURCE
---------------
mcxindia.com refuses scripted requests (Akamai). mtf.trading republishes the exchange's
own daily bhavcopy, captured by its maintainer through a real browser, behind a public
API with no key. Spot checks against a file downloaded from MCX directly match row for
row: prices, volume, open interest and ExpiryDate.

It is still a THIRD PARTY. Coverage depends on when its maintainer last ran the
capture (see `generated` in the index), and a mirror can lag or miss a day. So the
files are stored exactly as served and then go through the same parse-and-flag path
as a manual download (`--source csv`). Nothing here trusts the mirror more than it
would trust the exchange.

Attribution, per the site's terms: "MCX bhavcopy via mtf.trading".

WHAT IS SAVED
-------------
Only FUTCOM rows (all commodities, not just gold): options are premiums, not futures
prices, and the parser would drop them anyway. Each file is the API's CSV response,
byte for byte, named mtf_mcx_YYYYMMDD.csv so `csv_dir` can cross-check the date inside
the file against the date in the name -- the date-echo guard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

BASE = "https://mtf.trading"
INDEX = BASE + "/mcx/index.json"
API = BASE + "/api/v1/mcx"
HEADERS = {"User-Agent": "mcxrv-research/1.0 (bhavcopy archive; MCX gold relative value)"}


def _d(s: str) -> date:
    return datetime.strptime(s.replace("-", ""), "%Y%m%d").date()


IST = timezone(timedelta(hours=5, minutes=30))
# How far past the index's last date to look for sessions it does not list yet.
PROBE_DAYS = 10


def available_dates(session: requests.Session) -> tuple[list[date], str]:
    # The index sits behind a CDN that keeps a copy for up to a day, so a plain
    # request can be told the archive ends a session early. Ask for a fresh copy.
    r = session.get(INDEX, params={"t": int(time.time())},
                    headers={"Cache-Control": "no-cache"}, timeout=60)
    r.raise_for_status()
    idx = r.json()
    return [_d(x["date"]) for x in idx.get("dates", [])], idx.get("generated", "?")


def unlisted_weekdays(listed: list[date], today: date | None = None) -> list[date]:
    """
    Weekdays after the index's last date and before today (IST) that it does not
    list. The API serves a session before the index is rebuilt to include it, so
    these are worth one request each; a day with no session answers 404. Today is
    left out: its bhavcopy does not exist until the evening session has closed.
    """
    if not listed:
        return []
    today = today or datetime.now(IST).date()
    out, d = [], max(listed) + timedelta(days=1)
    while d < today and len(out) < PROBE_DAYS:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def fetch_one(session: requests.Session, d: date, retries: int = 3) -> bytes | None:
    params = {"date": d.isoformat(), "instr": "FUTCOM", "format": "csv", "limit": 100000}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=60)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 404:
            return None                       # no session that day
        if r is not None and r.ok:
            return r.content
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{d}: failed after {retries} attempts"
                       + (f" (HTTP {r.status_code})" if r is not None else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: earliest available)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: latest available)")
    p.add_argument("--delay", type=float, default=0.25, help="seconds between requests")
    p.add_argument("--refetch", action="store_true", help="overwrite files already on disk")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    s = requests.Session()
    s.headers.update(HEADERS)

    dates, generated = available_dates(s)
    extra = unlisted_weekdays(dates)
    dates = dates + extra
    if args.start:
        dates = [d for d in dates if d >= _d(args.start)]
    if args.end:
        dates = [d for d in dates if d <= _d(args.end)]
    if not dates:
        print("no sessions in that range")
        return 0
    print(f"mtf.trading index generated {generated}: {len(dates)} sessions "
          f"{dates[0]} .. {dates[-1]}"
          + (f" (incl. {len(extra)} day(s) after the index, checked directly)" if extra else ""))

    got = skipped = empty = 0
    for i, d in enumerate(dates, 1):
        path = os.path.join(args.out_dir, f"mtf_mcx_{d:%Y%m%d}.csv")
        if os.path.exists(path) and not args.refetch:
            skipped += 1
            continue
        body = fetch_one(s, d)
        if body is None or body.count(b"\n") < 2:
            empty += 1
        else:
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, path)             # never leave a half-written session
            got += 1
        if i % 50 == 0:
            print(f"  {i}/{len(dates)}  saved {got}  already had {skipped}")
        time.sleep(args.delay)

    print(f"done: saved {got}, already on disk {skipped}, empty {empty} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
