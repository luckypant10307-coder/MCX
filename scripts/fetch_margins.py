"""
Collect published MCX margin rates for the four gold contracts.

  python scripts/fetch_margins.py                       # today's rates (Zerodha page)
  python scripts/fetch_margins.py --backfill 2022-01-01 # dated archive (Aditya Birla
                                                        # Money PDFs; needs `pypdf`)

Writes data/reference/published_margins.json. Safe to re-run: days already stored
are skipped, and a total-only page never replaces a full breakdown for the same day.
The backfill asks for one file per weekday, one second apart.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import published_margins as pm  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=pm.DEFAULT_PATH)
    p.add_argument("--backfill", metavar="START", help="YYYY-MM-DD: walk the PDF archive")
    p.add_argument("--end", default=None, help="last day for --backfill (default today)")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between archive requests")
    a = p.parse_args(argv)
    store = pm.load(a.out)

    if not a.backfill:
        try:
            sheet = pm.fetch_zerodha()
        except Exception as e:              # a page change must not stop the nightly run
            print("margins: Zerodha page not read ({})".format(e))
            return 0
        changed = pm.add(store, sheet, "zerodha")
        pm.save(store, a.out)
        print("margins {}: {} ({} contracts) {}".format(
            sheet["date"], ", ".join("{} {:.2f}%".format(s, r * 100)
                                     for s, r in sorted(pm.symbol_rates(sheet).items())),
            len(sheet["contracts"]), "stored" if changed else "unchanged"))
        return 0

    import requests
    day = datetime.strptime(a.backfill, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date() if a.end else date.today()
    found = asked = 0
    with requests.Session() as s:
        while day <= end:
            if day.weekday() < 5 and store.get(day.isoformat(), {}).get("source") != "abml":
                asked += 1
                try:
                    sheet = pm.fetch_abml(day, session=s)
                except Exception as e:
                    print("{}: unreadable ({})".format(day, e))
                    sheet = None
                if sheet:
                    pm.add(store, sheet, "abml")
                    found += 1
                    print("{}: {}".format(day, ", ".join(
                        "{} {:.2f}%".format(k, v * 100)
                        for k, v in sorted(pm.symbol_rates(sheet).items()))))
                    if found % 20 == 0:
                        pm.save(store, a.out)
                time.sleep(a.delay)
            day += timedelta(days=1)
    pm.save(store, a.out)
    print("backfill: {} sheets found in {} weekdays asked; store has {} days".format(
        found, asked, len(store)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
