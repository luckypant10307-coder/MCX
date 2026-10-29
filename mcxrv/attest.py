"""
Tamper-evident daily record of what the monitor said, timestamped in Bitcoin through
OpenTimestamps (https://opentimestamps.org).

Each session gets a small JSON record: the pair statuses and reasons, that session's
decisions, the backtest headline, and the SHA-256 of the full snapshot. The record's
hash -- only the hash -- is sent to the public OpenTimestamps calendars, which commit
it into a Bitcoin transaction within a few hours. The `.ots` proof file then shows
that this exact record existed by that block's time: an alert cannot be added, removed
or edited afterwards without the proof failing.

What it does not show: that a signal was right, that the input prices were right, or
that no other record was made. Records are written by the scheduled job and committed
to a public branch every session, which is what makes cherry-picking visible.

  record(snapshot, snapshot_bytes)  -> dict        what gets stamped
  write(record, folder)             -> path|None   canonical JSON, one file per session+data
  stamp(path)                       -> .ots path   submit to the calendars
  upgrade(ots_path)                 -> bool        fetch the Bitcoin part once it exists
  verify(path)                      -> dict        check the proof against a block explorer
"""
from __future__ import annotations

import hashlib
import json
import os

CALENDARS = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://ots.btc.catallaxy.com",
)
EXPLORER = "https://blockstream.info/api"
RECORD_VERSION = 1


# ------------------------------------------------------------------ the record

def record(snap: dict, snapshot_bytes: bytes | None = None) -> dict:
    """The part of a snapshot worth proving: what was said, for which session."""
    meta = snap.get("meta", {})
    asof = meta.get("asof")
    return {
        "record_version": RECORD_VERSION,
        "asof": asof,
        "run_id": meta.get("run_id"),
        "data_fingerprint": meta.get("data_fingerprint"),
        "generated_at": meta.get("generated_at"),
        "source": meta.get("source"),
        "margin_parameters": meta.get("span_parameters"),
        "summary": snap.get("summary"),
        "pairs": [{k: p.get(k) for k in ("pair", "status", "z", "reason")}
                  for p in snap.get("pairs", [])],
        "decisions": [d for d in snap.get("decision_log", []) if d.get("date") == asof],
        "backtest": (snap.get("backtest") or {}).get("verdict"),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest() if snapshot_bytes else None,
    }


def canonical(rec: dict) -> bytes:
    """One byte sequence per record, whatever produced it: sorted keys, no spacing."""
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def write(rec: dict, folder: str) -> str | None:
    """
    Save the record as `<asof>_<run_id>.json`. Returns None when that file exists:
    the same session on the same data was already recorded (the job runs twice a
    day), and re-stamping it would only add noise. Different data for the same
    session gets its own file, so a revision is visible rather than overwritten.
    """
    if not rec.get("asof") or not rec.get("run_id"):
        raise ValueError("record has no session date or run id")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "{}_{}.json".format(rec["asof"], rec["run_id"]))
    if os.path.exists(path):
        return None
    with open(path, "wb") as fh:
        fh.write(canonical(rec))
    return path


# ------------------------------------------------------------------ OpenTimestamps

def stamp(path: str, calendars=CALENDARS, timeout: float = 15.0) -> dict:
    """
    Submit the file's SHA-256 to the calendars and write `<path>.ots`.

    Same construction as `ots stamp`: the file hash, a random nonce appended so the
    calendars learn nothing about the file, hashed again, and that digest submitted.
    Succeeds if at least one calendar answers.
    """
    from opentimestamps.calendar import RemoteCalendar
    from opentimestamps.core.op import OpAppend, OpSHA256
    from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).digest()
    dfile = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
    tip = dfile.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    ok, errors = [], []
    for url in calendars:
        try:
            tip.merge(RemoteCalendar(url).submit(tip.msg, timeout=timeout))
            ok.append(url)
        except Exception as e:             # one calendar down must not lose the stamp
            errors.append("{}: {}".format(url, e))
    if not ok:
        raise RuntimeError("no calendar accepted the timestamp: " + "; ".join(errors))
    _save(dfile, path + ".ots")
    return {"ots": path + ".ots", "calendars": ok, "errors": errors}


def upgrade(ots_path: str, timeout: float = 15.0) -> bool:
    """
    Ask each calendar for the Bitcoin part of a pending proof. True if the file now
    contains a Bitcoin attestation (it changed, or already had one).
    """
    from opentimestamps.calendar import RemoteCalendar
    from opentimestamps.core.notary import PendingAttestation

    dfile = _load(ots_path)
    if _bitcoin(dfile.timestamp):
        return True
    changed = False
    for ts in _walk(dfile.timestamp):
        for att in list(ts.attestations):
            if isinstance(att, PendingAttestation):
                try:
                    ts.merge(RemoteCalendar(att.uri).get_timestamp(ts.msg, timeout=timeout))
                    changed = True
                except Exception:          # not committed yet, or calendar busy
                    continue
    if changed:
        _save(dfile, ots_path)
    return bool(_bitcoin(dfile.timestamp))


def verify(path: str, session=None, timeout: float = 20.0) -> dict:
    """
    Check `<path>.ots` against `path` and, for each Bitcoin attestation, against the
    block's merkle root from a public block explorer. Returns the earliest block's
    height and time when it verifies; never raises for a pending or failed proof.
    """
    import requests

    dfile = _load(path + ".ots")
    with open(path, "rb") as fh:
        if hashlib.sha256(fh.read()).digest() != dfile.file_digest:
            return {"ok": False, "reason": "file does not match the proof (changed since stamping)"}
    http = session or requests
    found = []
    for msg, height in _bitcoin(dfile.timestamp):
        try:
            bhash = http.get("{}/block-height/{}".format(EXPLORER, height), timeout=timeout).text.strip()
            blk = http.get("{}/block/{}".format(EXPLORER, bhash), timeout=timeout).json()
        except Exception as e:
            return {"ok": False, "reason": "block explorer unreachable ({})".format(e)}
        # Header fields are little-endian; explorers print the merkle root reversed.
        if bytes.fromhex(blk["merkle_root"])[::-1] == msg:
            found.append({"height": height, "block_time": blk["timestamp"], "block": bhash})
        else:
            return {"ok": False, "reason": "proof does not match block {}".format(height)}
    if not found:
        return {"ok": False, "pending": True,
                "reason": "not in Bitcoin yet; run upgrade in a few hours"}
    first = min(found, key=lambda b: b["height"])
    return {"ok": True, **first}


# ------------------------------------------------------------------ helpers

def _walk(ts):
    yield ts
    for sub in ts.ops.values():
        yield from _walk(sub)


def _bitcoin(ts) -> list:
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
    return [(msg, att.height) for msg, att in ts.all_attestations()
            if isinstance(att, BitcoinBlockHeaderAttestation)]


def _load(ots_path: str):
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile
    with open(ots_path, "rb") as fh:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(fh))


def _save(dfile, ots_path: str) -> None:
    from opentimestamps.core.serialize import StreamSerializationContext
    tmp = ots_path + ".tmp"
    with open(tmp, "wb") as fh:
        dfile.serialize(StreamSerializationContext(fh))
    os.replace(tmp, ots_path)
