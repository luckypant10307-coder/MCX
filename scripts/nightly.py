"""
Nightly job: run the pipeline, write the snapshot the API serves, and send Telegram
messages for anything that changed since the last run.

  python scripts/nightly.py                                  # synthetic sample data
  python scripts/nightly.py --source csv --csv-dir data/raw  # downloaded bhavcopy
  python scripts/nightly.py --source db                      # validated Postgres store
  python scripts/nightly.py --no-notify                      # build only

Writes data/processed/snapshot.json (atomically) and keeps a dated copy under
data/processed/history/ so any evening's state can be reproduced and compared.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import circulars, notify, pipeline, published_margins, risk_inbox, snapshot, user_notify  # noqa: E402
from mcxrv.stage1_intake.config import load_dotenv  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "live", "db"])
    p.add_argument("--csv-dir", default="data/raw")
    p.add_argument("--start", default=None,
                   help="default: all stored history for csv/db (the research view needs "
                        "every session); 450 days before --end for synthetic/live")
    p.add_argument("--end", default=None,
                   help="last session to include. Default: today for real data (csv, "
                        "db, live); 2026-09-18 for synthetic, matching the README")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--account-capital", type=float, default=1_000_000.0)
    p.add_argument("--out", default=os.environ.get("MCX_SNAPSHOT_PATH",
                                                   "data/processed/snapshot.json"))
    p.add_argument("--no-notify", action="store_true")
    p.add_argument("--risk-dir", default="data/risk",
                   help="drop folder for MCXCCL risk files (inbox/ accepted/ rejected/)")
    p.add_argument("--margins-file", default=published_margins.DEFAULT_PATH,
                   help="published margin rates (scripts/fetch_margins.py); '' to ignore")
    p.add_argument("--no-margins-fetch", action="store_true",
                   help="use the stored rates without reading today's Zerodha page")
    p.add_argument("--no-circulars", action="store_true",
                   help="skip the TinyFish search for new MCX circulars")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.end:
        end = datetime.strptime(a.end, "%Y-%m-%d").date()
    elif a.source == "synthetic":
        end = datetime(2026, 9, 18).date()
    else:
        # Real data: everything up to the latest session on hand. A fixed default
        # here silently cut off every session after it.
        end = datetime.now(timezone.utc).date()
    if a.start:
        start = datetime.strptime(a.start, "%Y-%m-%d").date()
    elif a.source in ("csv", "db"):
        start = None                    # everything on disk / in the store
    else:
        start = end - timedelta(days=450)
    load_dotenv()                       # TINYFISH_API_KEY / TELEGRAM_* from a local .env
    # 1. Risk files someone saved into the inbox since the last run.
    inbox = risk_inbox.process(a.risk_dir)
    for d in inbox:
        print("risk file {}: {}{}".format(d["file"], d["status"],
                                          "" if d["status"] == "ACCEPTED" else " - " + d["reason"]))
    # 2. Today's published margin rates. Real data only: synthetic sessions are
    # invented dates, and a real rate on them would be meaningless.
    real = a.source != "synthetic" and bool(a.margins_file)
    margin_changes = []
    if real and not a.no_margins_fetch:
        store = published_margins.load(a.margins_file)
        try:
            sheet = published_margins.fetch_zerodha()
            seen = store.get(sheet["date"].isoformat())
            if published_margins.add(store, sheet, "zerodha"):
                published_margins.save(store, a.margins_file)
                # The page's prices move between the two daily runs; only new RATES
                # are news, so a re-read of the same day never alerts twice.
                if seen is None or (published_margins.symbol_rates(seen)
                                    != published_margins.symbol_rates(sheet)):
                    margin_changes = published_margins.latest_change(store)
            print("margins {}: {}".format(sheet["date"], ", ".join(
                "{} {:.2f}%".format(s, r * 100)
                for s, r in sorted(published_margins.symbol_rates(sheet).items()))))
        except Exception as e:              # the page can change; the run must not stop
            print("margins: today's rates not read ({}); using stored ones".format(e))
    cfg = pipeline.PipelineConfig(source=a.source, start=start, end=end,
                                  csv_dir=a.csv_dir, seed=a.seed, z_entry=a.z_entry,
                                  account_capital=a.account_capital, risk_dir=a.risk_dir,
                                  margins_file=a.margins_file if real else None)
    t0 = time.time()
    result = pipeline.run(cfg)
    archive = snapshot.archive_log(a.csv_dir) if a.source == "csv" else None
    snap = snapshot.build(result, archive=archive)
    sessions = sorted(result.quotes["date"].unique()) if result.quotes is not None and not result.quotes.empty else []
    snap["risk_files"] = snapshot.clean(risk_inbox.status(a.risk_dir, sessions))
    snap["published_margins"] = snapshot.clean(
        published_margins.status(published_margins.load(a.margins_file), sessions) if real
        else {"available": False, "days": 0, "reason": "sample data: published rates not applied"})
    # 3. New MCX circulars, via TinyFish search (skipped without TINYFISH_API_KEY).
    circ = ({"available": False, "reason": "skipped (--no-circulars)"} if a.no_circulars else
            circulars.check(seen_path=os.path.join(os.path.dirname(a.out) or ".", "circulars_seen.json")))
    snap["circulars"] = snapshot.clean(circ)
    prev = snapshot.load(a.out)
    snapshot.save(snap, a.out)

    hist_dir = os.path.join(os.path.dirname(a.out) or ".", "history")
    snapshot.save(snap, os.path.join(hist_dir, "snapshot_{}_{}.json".format(
        snap["meta"]["asof"] or "none", snap["meta"]["run_id"])))

    s = snap["summary"]
    print("asof {}  source {}  run {}  ({:.0f}s)".format(
        snap["meta"]["asof"], a.source, snap["meta"]["run_id"], time.time() - t0))
    print("pairs: {} ALERT, {} NO_TRADE, {} QUIET, {} WARMUP".format(
        s.get("ALERT", 0), s.get("NO_TRADE", 0), s.get("QUIET", 0), s.get("WARMUP", 0)))
    for p in snap["pairs"]:
        print("  {:<22s} {:<8s} z {:>6}  {}".format(
            p["pair"], p["status"], "n/a" if p["z"] is None else "{:+.2f}".format(p["z"]),
            p["reason"]))
    print(snap["backtest"]["verdict"]["headline"])
    print("margin parameters: {}".format(snap["meta"].get("span_parameters")))
    if circ.get("available"):
        print("circulars: {} relevant, {} new".format(len(circ["items"]), circ["new"]))
    else:
        print("circulars: {}".format(circ.get("reason")))
    print("snapshot -> {}".format(a.out))

    if not a.no_notify:
        def report(event, r):
            for ch in ("telegram", "ntfy"):
                res = r.get(ch, {})
                print("{} [{}] {}".format(ch, event, "sent" if res.get("sent")
                                          else res.get("reason", "not sent (status {})".format(res.get("status")))))
        text = circulars.message(circ)
        if text:
            report("circulars", notify.deliver(text, kind="circulars"))
        mtext = published_margins.message(margin_changes)
        if mtext:
            report("margin_changed", notify.deliver(mtext, kind="margin_changed"))
            # Users hear about it under "exchange notices", with the circulars.
            text = "\n\n".join(t for t in (text, mtext) if t)
        sent = notify.notify(prev, snap)
        if not sent:
            print("notify: nothing changed, no message")
        for m in sent:
            report(m["event"], m)
        summary = notify.digest(prev, snap)
        if summary:
            report("digest", notify.deliver(summary, kind="digest"))
        else:
            print("notify: nothing unusual to summarise for this session")
        # Registered users: each gets what their settings ask for, on their own devices
        # and in the site's notification centre. Skipped when Supabase is not set up.
        if user_notify.supa.configured():
            res = user_notify.fanout(user_notify.events_from_run(sent, summary, text),
                                     run_id=snap["meta"]["run_id"])
            print("users: {users} | notifications {notifications} | pushed {pushed} | "
                  "failed {push_failed} | devices removed {devices_removed}".format(**res))
            for e in res["errors"]:
                print("users: " + e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
