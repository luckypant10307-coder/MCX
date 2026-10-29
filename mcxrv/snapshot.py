"""
A pipeline run reduced to one JSON document: what the API serves, what the dashboard
draws, and what the alerting job diffs against yesterday.

Built once per run and read many times, so a request never re-runs the model. Every
value is JSON-safe (NaN and inf become null) and every decision carries the reason
that produced it -- the same "explain the no" rule the risk board follows.

Decisions, per pair on the latest session:

  ALERT     the statistical gates cleared AND the risk board says the position is
            fundable ("may add" / "add reduced size")
  NO_TRADE  something unusual happened but a gate refused it: a statistical gate
            (history, stationarity, half-life, magnitude) or the risk board
            (capital, buffer, lot granularity). The binding reason is named.
  QUIET     nothing unusual. The default, and most days the answer.

The decision log covers past sessions, where only the statistical gates are known:
TRADED (the backtest entered next session), SIGNAL (gates passed, not entered) or
NO_TRADE (a gate refused it).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from mcxrv import normalize, specs

SCHEMA_VERSION = 1
HISTORY_SESSIONS = 250
DECISION_LOG_SESSIONS = 120

GATE_LABELS = {
    "gate_history": "not enough history",
    "gate_stationary": "spread not stationary",
    "gate_halflife": "half-life too long",
    "gate_magnitude": "dislocation too small",
}
ADDABLE = ("may add", "add reduced size")


# ---------------------------------------------------------------- JSON safety
def clean(v):
    """Recursively convert to JSON-safe builtins. NaN / inf -> None."""
    if v is None or v is pd.NaT:          # NaT is a datetime subclass; catch it first
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _records(df: pd.DataFrame, cols) -> list[dict]:
    cols = [c for c in cols if c in df.columns]
    return [clean(r) for r in df[cols].to_dict("records")]


def fingerprint(quotes: pd.DataFrame) -> str:
    """Hash of the validated input: same data -> same id, any change -> new id."""
    if quotes is None or quotes.empty:
        return "empty"
    cols = [c for c in ("date", "contract_id", "close", "volume") if c in quotes.columns]
    blob = quotes[cols].sort_values(cols[:2]).to_csv(index=False).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- decisions
def failed_gates(row) -> list[str]:
    return [GATE_LABELS[g] for g in GATE_LABELS if g in row and not bool(row[g])]


def decide(pair_row, board_row, z_entry: float) -> dict:
    """One pair, one session -> {status, reason}. Pure; tested directly."""
    z = pair_row.get("z")
    unusual = z is not None and np.isfinite(z) and abs(z) >= z_entry
    if bool(pair_row.get("signal_ok")):
        action = board_row.get("action") if board_row is not None else None
        if action in ADDABLE:
            return {"status": "ALERT", "reason": "all gates passed; " + str(action),
                    "binding_gate": None}
        reason = (board_row or {}).get("reason") or "risk board unavailable"
        return {"status": "NO_TRADE", "reason": reason, "binding_gate": "risk"}
    if unusual:
        fails = failed_gates(pair_row)
        return {"status": "NO_TRADE", "reason": ", ".join(fails) or "gates not cleared",
                "binding_gate": "signal"}
    return {"status": "QUIET", "reason": "|z| below {:.1f}".format(z_entry),
            "binding_gate": None}


def _verdict(perf: dict) -> dict:
    n = int(perf.get("n_trades") or 0)
    if n == 0:
        return {"headline": "No trades taken: every candidate failed a gate.",
                "edge_survives_costs": False}
    net = perf.get("mean_net_bp")
    gross = perf.get("mean_gross_bp")
    ok = net is not None and np.isfinite(net) and net > 0
    head = ("Net edge positive after costs ({:+.1f} bp per trade)".format(net) if ok else
            "No edge survives costs: {:+.1f} bp gross becomes {:+.1f} bp net per trade"
            .format(gross or 0.0, net or 0.0))
    return {"headline": head, "edge_survives_costs": bool(ok)}


# ---------------------------------------------------------------- build
def build(result, generated_at: datetime | None = None) -> dict:
    """PipelineResult -> snapshot dict."""
    cfg = result.config
    z_entry = float(cfg.z_entry)
    generated_at = generated_at or datetime.now(timezone.utc)
    asof = result.asof
    q, pairs, board = result.quotes, result.pairs, result.board

    # contracts on the latest session, front month per symbol
    contracts = []
    if q is not None and not q.empty:
        last = q[q["date"] == q["date"].max()]
        fm = normalize.front_month(last)
        liq = {}
        if result.rows is not None and not result.rows.empty:
            lr = result.rows[result.rows["date"] == result.rows["date"].max()]
            liq = lr.set_index("contract_id")[["tradable", "half_spread_bp"]].to_dict("index")
        for r in fm.to_dict("records"):
            sp = specs.get(r["symbol"])
            extra = liq.get(r["contract_id"], {})
            contracts.append(clean({
                "symbol": r["symbol"], "contract_id": r["contract_id"],
                "expiry_date": r.get("expiry_date"), "close": r.get("close"),
                "quote_base_g": sp.quote_base_g, "trading_unit_g": sp.trading_unit_g,
                "purity": sp.purity,
                "norm_close_10g": r.get("norm_close"),
                "inr_per_g_fine": (r.get("norm_close") or float("nan")) / 10.0,
                "volume": r.get("volume"), "open_interest": r.get("open_interest"),
                "sessions_to_expiry": r.get("sessions_to_expiry"),
                "tradable": extra.get("tradable"),
                "half_spread_bp": extra.get("half_spread_bp"),
            }))

    # latest pair decisions
    brd = {}
    if board is not None and not board.empty:
        brd = {r["pair"]: r for r in board.to_dict("records")}
    latest_pairs, counts = [], {"ALERT": 0, "NO_TRADE": 0, "QUIET": 0}
    if pairs is not None and not pairs.empty:
        lp = pairs[pairs["date"] == pairs["date"].max()]
        for r in lp.to_dict("records"):
            b = brd.get(r["pair"])
            d = decide(r, b, z_entry)
            counts[d["status"]] += 1
            latest_pairs.append(clean({
                "pair": r["pair"], "sym_a": r["sym_a"], "sym_b": r["sym_b"],
                "maturity_matched": r.get("maturity_matched"),
                "spread_bp": r.get("spread_bp"), "centre_bp": r.get("centre"),
                "scale_bp": r.get("scale"), "z": r.get("z"),
                "dislocation_bp": r.get("dislocation_bp"),
                "halflife": r.get("halflife"), "adf_t": r.get("adf_t"),
                "gates": {g: bool(r.get(g)) for g in GATE_LABELS},
                "signal_ok": r.get("signal_ok"), "direction": r.get("direction"),
                **d,
                "risk": None if b is None else {
                    k: b.get(k) for k in (
                        "action", "reason", "lots_a", "lots_b", "notional_exposure",
                        "marginal_margin", "initial_margin", "span_margin", "elm",
                        "delivery_margin", "floor_binds", "utilisation_if_added",
                        "buffer_state", "span_change_pct", "sessions_to_expiry",
                        "residual_frac")},
            }))
        order = {"ALERT": 0, "NO_TRADE": 1, "QUIET": 2}
        latest_pairs.sort(key=lambda p: (order[p["status"]], -abs(p["z"] or 0)))

    # pair history for charts
    history = {}
    if pairs is not None and not pairs.empty:
        dates = sorted(pairs["date"].unique())[-HISTORY_SESSIONS:]
        ph = pairs[pairs["date"].isin(set(dates))]
        for name, g in ph.groupby("pair"):
            history[name] = _records(g.sort_values("date"),
                                     ["date", "spread_bp", "centre", "scale", "z",
                                      "signal_ok"])

    # decision log: every unusual pair-session in the recent window
    log = []
    if pairs is not None and not pairs.empty:
        dates = sorted(pairs["date"].unique())[-DECISION_LOG_SESSIONS:]
        w = pairs[pairs["date"].isin(set(dates)) & (pairs["z"].abs() >= z_entry)]
        traded = set()
        if result.trades is not None and not result.trades.empty:
            # entry is lagged one session for execution; the signal date is the
            # session before entry in the trading calendar
            all_dates = sorted(pairs["date"].unique())
            idx = {d: i for i, d in enumerate(all_dates)}
            for t in result.trades.itertuples(index=False):
                i = idx.get(t.entry_date)
                if i is not None and i > 0:
                    traded.add((all_dates[i - 1], t.pair))
        for r in w.sort_values(["date", "pair"], ascending=[False, True]).to_dict("records"):
            if bool(r["signal_ok"]):
                # The risk board is only computed for the latest session, so history
                # can say a signal cleared its gates and whether the backtest took it
                # -- not that it was an alert a trader received.
                if (r["date"], r["pair"]) in traded:
                    status, reason = "TRADED", "gates passed; entered next session"
                else:
                    status, reason = "SIGNAL", "gates passed; not entered (book full, " \
                        "already open, or could not size)"
            else:
                status, reason = "NO_TRADE", ", ".join(failed_gates(r)) or "gates not cleared"
            log.append(clean({"date": r["date"], "pair": r["pair"], "z": r["z"],
                              "spread_bp": r["spread_bp"],
                              "dislocation_bp": r.get("dislocation_bp"),
                              "status": status, "reason": reason}))

    # gold level (exp of the fitted log level = INR per 10 g fine)
    gold = []
    cd = result.carry_daily
    if cd is not None and not cd.empty and "level" in cd.columns:
        g = cd.dropna(subset=["level"]).sort_values("date").tail(HISTORY_SESSIONS)
        gold = [clean({"date": d, "inr_per_10g_fine": float(np.exp(lv))})
                for d, lv in zip(g["date"], g["level"])]

    perf_keys = ("n_trades", "mean_gross_bp", "mean_net_bp", "median_net_bp",
                 "hit_rate_gross", "hit_rate_net", "gross_pnl", "total_cost", "net_pnl",
                 "cost_share_of_gross", "mean_sessions_held", "sharpe",
                 "peak_margin", "ann_return_on_margin", "max_drawdown_inr",
                 "exit_reasons", "tstat_caveat")
    perf = {k: result.performance.get(k) for k in perf_keys if k in result.performance}

    trades = []
    if result.trades is not None and not result.trades.empty:
        trades = _records(result.trades.sort_values("entry_date", ascending=False),
                          ["pair", "entry_date", "exit_date", "sessions_held",
                           "lots_a", "lots_b", "entry_z", "exit_z", "gross_bp",
                           "net_bp", "net_pnl", "exit_reason", "entry_margin"])

    rep = result.report
    n_sessions = int(q["date"].nunique()) if q is not None and not q.empty else 0
    snap = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "asof": asof, "generated_at": generated_at,
            "source": cfg.source, "null_world": cfg.null_world,
            "start": cfg.start, "end": cfg.end,
            "data_fingerprint": fingerprint(q),
            "run_id": fingerprint(q)[:12],
            "sessions": n_sessions, "z_entry": z_entry,
            "account_capital": cfg.account_capital,
            "span_parameters": result.span_source,
            "sample_data": cfg.source == "synthetic",
        },
        "summary": {**counts, "pairs": len(latest_pairs)},
        "contracts": contracts,
        "pairs": latest_pairs,
        "pair_history": history,
        "decision_log": log,
        "gold_level": gold,
        "data_quality": {
            "rows_in": rep.rows_in, "rows_out": rep.rows_out,
            "flags": dict(sorted(rep.counts.items(), key=lambda kv: -kv[1])),
            "date_echo_examples": sorted({tuple(map(str, e)) for e in
                                          (rep.details.get("date_echo_examples") or [])})[:10],
        },
        "backtest": {"performance": perf, "trades": trades,
                     "verdict": _verdict(result.performance)},
    }
    return clean(snap)


# ---------------------------------------------------------------- io
def save(snap: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f, indent=1, allow_nan=False)
    os.replace(tmp, path)         # atomic: the API never reads a half-written file
    return path


def load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
