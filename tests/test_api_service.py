"""
The API's read side (api/service.py) against a saved snapshot. api/main.py is routing
only, so these tests cover what every endpoint returns.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from api import service
from mcxrv import snapshot


@pytest.fixture()
def store(small_snapshot, tmp_path):
    p = tmp_path / "snapshot.json"
    snapshot.save(small_snapshot, str(p))
    return service.SnapshotStore(str(p))


def test_health_without_a_snapshot(tmp_path):
    st = service.SnapshotStore(str(tmp_path / "none.json"))
    h = service.health(st)
    assert h["ok"] is False and "nightly" in h["reason"]
    with pytest.raises(service.NotFound):
        service.overview(st)


def test_health(store, small_snapshot):
    h = service.health(store)
    assert h["ok"] and h["run_id"] == small_snapshot["meta"]["run_id"]
    assert h["sample_data"] is True


def test_overview(store, small_snapshot):
    o = service.overview(store)
    assert set(o) == {"meta", "summary", "contracts", "verdict", "data_quality", "gold_level"}
    assert o["data_quality"]["rows_out"] == small_snapshot["data_quality"]["rows_out"]
    json.dumps(o, allow_nan=False)


def test_pairs_filter_and_hide_risk(store):
    allp = service.pairs(store)
    assert len(allp) == 6 and all("risk" not in p for p in allp)
    quiet = service.pairs(store, status="quiet")
    assert all(p["status"] == "QUIET" for p in quiet)
    assert len(quiet) == sum(p["status"] == "QUIET" for p in allp)


@pytest.mark.parametrize("key", ["GOLDM-GOLDTEN", "goldm_goldten", "GOLDM/GOLDTEN"])
def test_pair_detail_accepts_url_safe_names(store, key):
    d = service.pair_detail(store, key)
    assert d["pair"] == "GOLDM/GOLDTEN"
    assert d["history"] and d["z_entry"] == 2.0
    assert all(x["pair"] == "GOLDM/GOLDTEN" for x in d["recent_decisions"])


def test_pair_detail_unknown(store):
    with pytest.raises(service.NotFound):
        service.pair_detail(store, "GOLDM-SILVER")


def test_decisions_filters_and_counts(store, small_snapshot):
    log = small_snapshot["decision_log"]
    d = service.decisions(store, limit=3)
    assert d["total"] == len(log) and len(d["items"]) == min(3, len(log))
    assert sum(d["counts"].values()) == len(log)
    one = service.decisions(store, limit=1000, pair="GOLDM-GOLDTEN")
    assert all(x["pair"] == "GOLDM/GOLDTEN" for x in one["items"])
    nt = service.decisions(store, limit=1000, status="no_trade")
    assert all(x["status"] == "NO_TRADE" for x in nt["items"])
    assert service.decisions(store, limit=0)["items"] == []


def test_risk_board_rows(store):
    rb = service.risk_board(store)
    assert len(rb) == 6
    assert all({"pair", "status", "z", "action"} <= set(r) for r in rb)


def test_data_quality_and_backtest(store, small_snapshot):
    dq = service.data_quality(store)
    assert dq["data_fingerprint"] == small_snapshot["meta"]["data_fingerprint"]
    assert dq["rows_in"] >= dq["rows_out"] > 0
    assert service.backtest(store) == small_snapshot["backtest"]


def test_store_reloads_when_the_file_is_replaced(store, small_snapshot):
    assert service.health(store)["asof"] == small_snapshot["meta"]["asof"]
    changed = json.loads(json.dumps(small_snapshot))
    changed["meta"]["asof"] = "2099-01-01"
    snapshot.save(changed, store.path)
    os.utime(store.path, (time.time() + 5, time.time() + 5))   # make the mtime move
    assert service.health(store)["asof"] == "2099-01-01"
