"""
A drop folder for MCXCCL risk-parameter files, so the only manual step is saving a file.

    data/risk/inbox/      you save files here, any name, from the MCXCCL site in a browser
    data/risk/accepted/   parsed cleanly; the margin model uses these
    data/risk/rejected/   could not be read with confidence; <file>.reason.txt says why
    data/risk/log.json    every decision, newest last

Why a drop folder and not a downloader: mcxccl.com refuses scripted requests (Akamai,
HTTP 403 even for robots.txt). Getting around that is not something this project
does, so a person fetches the file and everything after that is automatic.

Each file goes through the existing strict parser (spanfile.load). A file it cannot
read with confidence is REJECTED with the parser's own explanation rather than half
used: a margin parameter read from the wrong column is worse than an assumed one,
because it looks real.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

from mcxrv.stage3_gates_decision import spanfile

DEFAULT_ROOT = os.path.join("data", "risk")
LOG_KEEP = 200


def _dirs(root: str) -> dict:
    d = {k: os.path.join(root, k) for k in ("inbox", "accepted", "rejected")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def _unique(path: str) -> str:
    base, ext = os.path.splitext(path)
    n, out = 1, path
    while os.path.exists(out):
        out = f"{base}_{n}{ext}"
        n += 1
    return out


def _read_log(root: str) -> list:
    try:
        with open(os.path.join(root, "log.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def _write_log(root: str, log: list) -> None:
    tmp = os.path.join(root, "log.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log[-LOG_KEEP:], fh, indent=1, default=str)
    os.replace(tmp, os.path.join(root, "log.json"))


def process(root: str = DEFAULT_ROOT) -> list[dict]:
    """Validate everything waiting in the inbox; return this run's decisions."""
    d = _dirs(root)
    now = datetime.now(timezone.utc).isoformat()
    decisions = []
    for name in sorted(os.listdir(d["inbox"])):
        src = os.path.join(d["inbox"], name)
        if not os.path.isfile(src) or name.startswith("."):
            continue
        try:
            f = spanfile.load(src)
            dst = _unique(os.path.join(d["accepted"], f"{f.business_date:%Y%m%d}_{name}"))
            shutil.move(src, dst)
            decisions.append({
                "at": now, "file": name, "status": "ACCEPTED", "kind": f.kind,
                "business_date": f.business_date.isoformat(),
                "commodities": sorted(f.commodities), "credits": len(f.credits),
                "stored_as": os.path.basename(dst)})
        except Exception as exc:          # a parse failure is a decision, not a crash
            dst = _unique(os.path.join(d["rejected"], name))
            shutil.move(src, dst)
            reason = f"{type(exc).__name__}: {exc}"
            with open(dst + ".reason.txt", "w", encoding="utf-8") as fh:
                fh.write(reason + "\n\nRun scripts/load_risk_params.py --sniff on the file "
                         "to see how it is laid out.\n")
            decisions.append({"at": now, "file": name, "status": "REJECTED",
                              "reason": reason, "stored_as": os.path.basename(dst)})
    if decisions:
        _write_log(root, _read_log(root) + decisions)
    return decisions


def accepted_files(root: str = DEFAULT_ROOT) -> list:
    """Parsed RiskFiles from the accepted folder, oldest business date first."""
    folder = os.path.join(root, "accepted")
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            try:
                out.append(spanfile.load(p))
            except Exception:              # accepted earlier; a parser change can retire it
                continue
    return sorted(out, key=lambda r: r.business_date)


def status(root: str = DEFAULT_ROOT, sessions=None) -> dict:
    """What the dashboard shows: counts, recent decisions, and session coverage."""
    files = accepted_files(root)
    log = _read_log(root)
    out = {
        "root": root,
        "accepted": len(files),
        "rejected": sum(1 for x in log if x.get("status") == "REJECTED"),
        "waiting": len([n for n in os.listdir(os.path.join(root, "inbox"))
                        if not n.startswith(".")]) if os.path.isdir(os.path.join(root, "inbox")) else 0,
        "recent": list(reversed(log[-20:])),
        "dates": sorted({f.business_date.isoformat() for f in files}),
    }
    if sessions is not None and files:
        out["coverage"] = spanfile.coverage(files, sessions)
    return out
