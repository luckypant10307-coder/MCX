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
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import notify, pipeline, snapshot  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "live", "db"])
    p.add_argument("--csv-dir", default="data/raw")
    p.add_argument("--start", default=None,
                   help="default: all stored history for csv/db (the research view needs "
                        "every session); 450 days before --end for synthetic/live")
    p.add_argument("--end", default="2026-09-18",
                   help="last session to include (synthetic default matches the README)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--account-capital", type=float, default=1_000_000.0)
    p.add_argument("--out", default=os.environ.get("MCX_SNAPSHOT_PATH",
                                                   "data/processed/snapshot.json"))
    p.add_argument("--no-notify", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
    if a.start:
        start = datetime.strptime(a.start, "%Y-%m-%d").date()
    elif a.source in ("csv", "db"):
        start = None                    # everything on disk / in the store
    else:
        start = end - timedelta(days=450)
    cfg = pipeline.PipelineConfig(source=a.source, start=start, end=end,
                                  csv_dir=a.csv_dir, seed=a.seed, z_entry=a.z_entry,
                                  account_capital=a.account_capital)
    t0 = time.time()
    result = pipeline.run(cfg)
    archive = snapshot.archive_log(a.csv_dir) if a.source == "csv" else None
    snap = snapshot.build(result, archive=archive)
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
    print("snapshot -> {}".format(a.out))

    if not a.no_notify:
        sent = notify.notify(prev, snap)
        if not sent:
            print("telegram: nothing changed, no message")
        for m in sent:
            print("telegram [{}] {}: {}".format(
                m["event"], "sent" if m.get("sent") else "not sent",
                m.get("reason", m.get("status"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
