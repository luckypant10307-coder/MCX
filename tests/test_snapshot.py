"""
The snapshot is the only thing the API and the dashboard ever read, so it has to be
valid JSON, reproducible from the same data, and every pair has to carry the reason
for its status. The decision rule itself is tested as a pure function.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from datetime import date

from mcxrv import pipeline, snapshot

GATES_PASS = {"gate_history": True, "gate_stationary": True, "gate_halflife": True,
              "gate_magnitude": True}


# ---------------------------------------------------------------- decide()
def test_alert_needs_signal_and_an_addable_board():
    row = {"z": -2.6, "signal_ok": True, **GATES_PASS}
    d = snapshot.decide(row, {"action": "may add", "reason": ""}, 2.0)
    assert d["status"] == "ALERT"
    d = snapshot.decide(row, {"action": "add reduced size", "reason": ""}, 2.0)
    assert d["status"] == "ALERT"


def test_signal_blocked_by_risk_names_the_risk_reason():
    row = {"z": 2.4, "signal_ok": True, **GATES_PASS}
    d = snapshot.decide(row, {"action": "do not add exposure",
                              "reason": "margin utilisation above limit"}, 2.0)
    assert d == {"status": "NO_TRADE", "reason": "margin utilisation above limit",
                 "binding_gate": "risk"}
    d = snapshot.decide(row, None, 2.0)
    assert d["status"] == "NO_TRADE" and "unavailable" in d["reason"]


def test_unusual_but_gated_lists_each_failed_gate():
    row = {"z": 3.1, "signal_ok": False, **GATES_PASS,
           "gate_stationary": False, "gate_halflife": False}
    d = snapshot.decide(row, {"action": "may add"}, 2.0)
    assert d["status"] == "NO_TRADE" and d["binding_gate"] == "signal"
    assert "spread not stationary" in d["reason"] and "half-life too long" in d["reason"]
    assert "not enough history" not in d["reason"]


def test_quiet_below_entry_and_on_missing_z():
    assert snapshot.decide({"z": 1.2, "signal_ok": False}, None, 2.0)["status"] == "QUIET"
    assert snapshot.decide({"z": float("nan"), "signal_ok": False}, None, 2.0)["status"] == "QUIET"
    assert snapshot.decide({"z": None, "signal_ok": False}, None, 2.0)["status"] == "QUIET"


# ---------------------------------------------------------------- clean() / fingerprint()
def test_clean_makes_everything_json_safe():
    raw = {"a": np.float64("nan"), "b": np.inf, "c": np.int64(3), "d": np.bool_(True),
           "e": pd.Timestamp("2026-01-02"), "f": [np.float32(1.5), pd.NaT], 7: None}
    out = snapshot.clean(raw)
    assert out == {"a": None, "b": None, "c": 3, "d": True, "e": "2026-01-02T00:00:00",
                   "f": [1.5, None], "7": None}
    json.dumps(out, allow_nan=False)


def test_fingerprint_ignores_row_order_and_sees_any_price_change():
    q = pd.DataFrame({"date": ["2026-01-02", "2026-01-02"], "contract_id": ["A", "B"],
                      "close": [100.0, 200.0], "volume": [5, 6]})
    f = snapshot.fingerprint(q)
    assert f == snapshot.fingerprint(q.iloc[::-1])
    q2 = q.copy()
    q2.loc[1, "close"] = 200.5
    assert f != snapshot.fingerprint(q2)
    assert snapshot.fingerprint(pd.DataFrame()) == "empty"


# ---------------------------------------------------------------- build() on a real run
def test_snapshot_is_strict_json(small_snapshot):
    text = json.dumps(small_snapshot, allow_nan=False)      # raises on NaN / inf
    assert json.loads(text)["schema_version"] == snapshot.SCHEMA_VERSION


def test_meta_and_summary_agree_with_pairs(small_snapshot):
    s = small_snapshot
    m = s["meta"]
    assert m["source"] == "synthetic" and m["sample_data"] is True
    assert m["asof"] == "2026-09-18"
    assert len(m["run_id"]) == 12 and m["data_fingerprint"].startswith(m["run_id"])
    counts = {k: sum(p["status"] == k for p in s["pairs"]) for k in ("ALERT", "NO_TRADE", "QUIET")}
    for k, v in counts.items():
        assert s["summary"][k] == v
    assert s["summary"]["pairs"] == len(s["pairs"]) == 6


def test_every_pair_carries_a_reason_and_alerts_sort_first(small_snapshot):
    order = {"ALERT": 0, "NO_TRADE": 1, "QUIET": 2}
    ranks = [order[p["status"]] for p in small_snapshot["pairs"]]
    assert ranks == sorted(ranks)
    for p in small_snapshot["pairs"]:
        assert p["reason"]
        assert set(p["gates"]) == set(snapshot.GATE_LABELS)
        if p["status"] == "ALERT":
            assert p["signal_ok"] and p["risk"]["action"] in snapshot.ADDABLE


def test_contracts_are_priced_per_gram_of_fine_gold(small_snapshot):
    cs = small_snapshot["contracts"]
    assert {c["symbol"] for c in cs} == {"GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"}
    for c in cs:
        expect = c["close"] / (c["quote_base_g"] * c["purity"])
        assert math.isclose(c["inr_per_g_fine"], expect, rel_tol=1e-9)
    per_g = [c["inr_per_g_fine"] for c in cs]
    assert max(per_g) / min(per_g) < 1.05       # same metal, so within a few percent


def test_history_never_runs_past_asof(small_snapshot):
    asof = small_snapshot["meta"]["asof"]
    for hist in small_snapshot["pair_history"].values():
        assert len(hist) <= snapshot.HISTORY_SESSIONS
        assert all(h["date"] <= asof for h in hist)
    statuses = {d["status"] for d in small_snapshot["decision_log"]}
    assert statuses <= {"TRADED", "SIGNAL", "NO_TRADE"}     # history never says ALERT


def test_verdict_matches_performance(small_snapshot):
    bt = small_snapshot["backtest"]
    net = bt["performance"].get("mean_net_bp")
    if net is not None:
        assert bt["verdict"]["edge_survives_costs"] == (net > 0)
    assert bt["verdict"]["headline"]


def test_save_is_atomic_and_round_trips(small_snapshot, tmp_path):
    p = tmp_path / "out" / "snapshot.json"
    snapshot.save(small_snapshot, str(p))
    assert not (tmp_path / "out" / "snapshot.json.tmp").exists()
    assert snapshot.load(str(p)) == small_snapshot
    assert snapshot.load(str(tmp_path / "missing.json")) is None


def test_build_is_reproducible(small_result, small_snapshot):
    again = snapshot.build(small_result, generated_at=None)
    for k in ("pairs", "contracts", "decision_log", "backtest"):
        assert again[k] == small_snapshot[k]
    assert again["meta"]["run_id"] == small_snapshot["meta"]["run_id"]


def test_session_where_every_pair_is_unavailable():
    """Regression: 30 June 2026 leaves no pair hedgeable; risk.board used to KeyError."""
    r = pipeline.run(pipeline.PipelineConfig(start=date(2025, 3, 1), end=date(2026, 6, 30)))
    assert set(r.board["action"]) == {"unavailable"}
    s = snapshot.build(r)
    assert s["summary"]["pairs"] == 6
    assert all(p["risk"]["action"] == "unavailable" for p in s["pairs"])
