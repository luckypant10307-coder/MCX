"""
Read side of the API, independent of any web framework.

Everything here works on the snapshot the nightly job writes, so a request never
re-runs the model and the dashboard can never show a number the job did not compute.
`api/main.py` is only routing on top of these functions; the tests exercise this
module directly.
"""
from __future__ import annotations

import os

from mcxrv import snapshot as snap_mod

DEFAULT_PATH = os.environ.get("MCX_SNAPSHOT_PATH", "data/processed/snapshot.json")


class NotFound(Exception):
    pass


class SnapshotStore:
    """Loads the snapshot file and reloads it when the nightly job replaces it."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._mtime = None
        self._snap = None

    def get(self) -> dict:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            raise NotFound("no snapshot yet at {}; run scripts/nightly.py".format(self.path))
        if self._snap is None or mtime != self._mtime:
            self._snap = snap_mod.load(self.path)
            self._mtime = mtime
        return self._snap


# ---------------------------------------------------------------- endpoints
def health(store: SnapshotStore) -> dict:
    try:
        s = store.get()
    except NotFound as e:
        return {"ok": False, "reason": str(e)}
    m = s["meta"]
    return {"ok": True, "asof": m["asof"], "generated_at": m["generated_at"],
            "run_id": m["run_id"], "source": m["source"], "sample_data": m["sample_data"]}


def overview(store: SnapshotStore) -> dict:
    s = store.get()
    dq = s["data_quality"]
    return {
        "meta": s["meta"],
        "summary": s["summary"],
        "contracts": s["contracts"],
        "verdict": s["backtest"]["verdict"],
        "data_quality": {"rows_in": dq["rows_in"], "rows_out": dq["rows_out"],
                         "date_echo_rows": dq["flags"].get("date_echo_rows", 0)},
        "gold_level": s["gold_level"],
    }


def pairs(store: SnapshotStore, status: str | None = None) -> list[dict]:
    rows = store.get()["pairs"]
    if status:
        want = status.upper()
        rows = [p for p in rows if p["status"] == want]
    return [{k: v for k, v in p.items() if k != "risk"} for p in rows]


def _pair_key(pair: str) -> str:
    # URLs cannot carry a bare "/", so accept GOLDM-GOLDTEN as well as GOLDM/GOLDTEN.
    return pair.upper().replace("-", "/").replace("_", "/")


def pair_detail(store: SnapshotStore, pair: str) -> dict:
    s = store.get()
    key = _pair_key(pair)
    for p in s["pairs"]:
        if p["pair"] == key:
            return {**p, "history": s["pair_history"].get(key, []),
                    "z_entry": s["meta"]["z_entry"],
                    "recent_decisions": [d for d in s["decision_log"]
                                         if d["pair"] == key][:20]}
    raise NotFound("unknown pair {}".format(pair))


def decisions(store: SnapshotStore, limit: int = 50, status: str | None = None,
              pair: str | None = None) -> dict:
    log = store.get()["decision_log"]
    if status:
        log = [d for d in log if d["status"] == status.upper()]
    if pair:
        log = [d for d in log if d["pair"] == _pair_key(pair)]
    counts = {}
    for d in store.get()["decision_log"]:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return {"total": len(log), "counts": counts, "items": log[:max(0, int(limit))]}


def risk_board(store: SnapshotStore) -> list[dict]:
    return [{"pair": p["pair"], "status": p["status"], "z": p["z"], **(p["risk"] or {})}
            for p in store.get()["pairs"]]


def data_quality(store: SnapshotStore) -> dict:
    s = store.get()
    return {**s["data_quality"], "run_id": s["meta"]["run_id"],
            "data_fingerprint": s["meta"]["data_fingerprint"], "asof": s["meta"]["asof"]}


def backtest(store: SnapshotStore) -> dict:
    return store.get()["backtest"]


def research(store: SnapshotStore) -> dict:
    s = store.get()
    if not s.get("research"):
        raise NotFound("this snapshot has no research section; rebuild it with "
                       "scripts/nightly.py")
    return {**s["research"], "verdict": s["backtest"]["verdict"],
            "performance": s["backtest"]["performance"], "meta": s["meta"]}
