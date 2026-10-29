"""
Timestamp the latest session's signals in Bitcoin (OpenTimestamps).

  python scripts/attest.py                          # record + stamp today, upgrade pending
  python scripts/attest.py --verify attestations/2026-10-26_f427a137e820.json

Records go to attestations/ (the nightly job keeps them on the `attestations` branch).
A new proof is "pending" for a few hours until the calendars commit it to a block;
every later run upgrades pending proofs, and --verify checks one against the block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import attest  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default=os.environ.get("MCX_SNAPSHOT_PATH",
                                                        "data/processed/snapshot.json"))
    p.add_argument("--dir", default="attestations")
    p.add_argument("--verify", metavar="RECORD", help="check one record's proof and exit")
    p.add_argument("--no-stamp", action="store_true", help="only upgrade pending proofs")
    a = p.parse_args(argv)

    if a.verify:
        res = attest.verify(a.verify)
        if res.get("ok"):
            when = datetime.fromtimestamp(res["block_time"], timezone.utc)
            print("VERIFIED: {} existed by Bitcoin block {} ({:%Y-%m-%d %H:%M} UTC)".format(
                os.path.basename(a.verify), res["height"], when))
            return 0
        print("NOT VERIFIED: " + res["reason"])
        return 0 if res.get("pending") else 1

    _explain(a.dir)
    if not a.no_stamp:
        with open(a.snapshot, "rb") as fh:
            raw = fh.read()
        snap = json.loads(raw)
        if (snap.get("meta") or {}).get("sample_data"):
            print("attest: sample data, nothing recorded")
        else:
            path = attest.write(attest.record(snap, raw), a.dir)
            if path is None:
                print("attest: session {} on this data already recorded".format(snap["meta"]["asof"]))
            else:
                res = attest.stamp(path)
                print("attest: {} stamped via {} calendar(s){}".format(
                    os.path.basename(path), len(res["calendars"]),
                    "; " + "; ".join(res["errors"]) if res["errors"] else ""))

    done = pending = 0
    for ots in sorted(glob.glob(os.path.join(a.dir, "*.ots"))):
        try:
            if attest.upgrade(ots):
                done += 1
            else:
                pending += 1
        except Exception as e:
            print("attest: {} not upgraded ({})".format(os.path.basename(ots), e))
    print("attest: {} proof(s) in Bitcoin, {} still pending".format(done, pending))
    return 0


README = """# Timestamped daily records

One file per MCX session: what the Gold Spread Monitor said (pair status, z-score,
reason, decisions, backtest verdict) and the SHA-256 of the full snapshot. Each
`.json.ots` is an OpenTimestamps proof that the `.json` beside it existed, byte for
byte, by the time of a Bitcoin block. Written automatically by the nightly job.

Verify one record (the proof is pending for a few hours after it is made):

    python scripts/attest.py --verify attestations/<session>_<run>.json

or drop the `.ots` file and the `.json` on https://opentimestamps.org.

A proof shows when the record existed and that it has not changed since. It does not
show that a signal was right or that the prices behind it were correct.
"""


def _explain(folder: str) -> None:
    """A README for the branch, and no line-ending conversion: the bytes are the proof."""
    os.makedirs(folder, exist_ok=True)
    for name, text in (("README.md", README), (".gitattributes", "* -text\n")):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)


if __name__ == "__main__":
    sys.exit(main())
