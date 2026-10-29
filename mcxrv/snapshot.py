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

from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage4_simulate_deliver import eventstudy
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage1_intake import specs
from mcxrv.stage2_normalize_signal.z_score import SignalConfig

SCHEMA_VERSION = 2          # 2: adds the full-history "research" section
# Notional and half-spread behind the event study's cost line: the same assumption
# scripts/run_analysis.py prints, so the dashboard and the written report agree.
RESEARCH_NOTIONAL_INR = 500_000.0
RESEARCH_HALF_SPREAD_BP = 5.0
PREMIUM_EVERY_N_SESSIONS = 5
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


def _verdict(perf: dict, warming_up: str | None = None) -> dict:
    n = int(perf.get("n_trades") or 0)
    if n == 0 and warming_up:
        return {"headline": "Not enough history to backtest yet ({}).".format(warming_up),
                "edge_survives_costs": None}
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


# ---------------------------------------------------------------- monitoring detail
LIQ_AVG_SESSIONS = 20


def liquidity_level(symbol: str, avg_lots: float | None) -> str | None:
    """
    GREEN at >= 2x the symbol's minimum daily volume, YELLOW at >= 1x, RED below.
    The minimum is the same threshold the tradability gate uses (pair_match), so the
    colour and the gate cannot disagree.
    """
    from mcxrv.stage2_normalize_signal.pair_match import LiquidityConfig
    if avg_lots is None or not np.isfinite(avg_lots):
        return None
    floor = LiquidityConfig().min_volume_lots.get(symbol, 10.0)
    return "GREEN" if avg_lots >= 2 * floor else "YELLOW" if avg_lots >= floor else "RED"


def suggested_trade(p: dict, legs: dict) -> dict | None:
    """
    Which side of each leg the pair would trade, with lots and fine grams.

    direction is set by the engine only when every gate passes; for a pair that is
    unusual but blocked, the side follows the sign of z so the reader can see what
    WOULD have been traded. Spread too high (z > 0): sell a, buy b.
    """
    risk = p.get("risk") or {}
    z = p.get("z")
    d = p.get("direction") or (0 if z is None else (-1 if z > 0 else 1))
    if not d:
        return None
    a, b = p["sym_a"], p["sym_b"]
    buy, sell = (a, b) if d > 0 else (b, a)
    lots = {a: risk.get("lots_a"), b: risk.get("lots_b")}

    def leg(sym, side):
        n = lots.get(sym)
        fg = specs.get(sym).fine_g_per_lot
        return {"side": side, "symbol": sym, "expiry": (legs.get(sym) or {}).get("expiry_date"),
                "lots": n, "fine_g": None if n is None else n * fg}
    return {"legs": [leg(sell, "SELL"), leg(buy, "BUY")],
            "residual_frac": risk.get("residual_frac"),
            "margin": risk.get("marginal_margin")}


def archive_log(csv_dir: str, show: int = 30) -> dict:
    """
    Every downloaded bhavcopy file: the date it was requested for (from its name),
    the date written inside it, gold futures rows, SHA-256 of the bytes as stored,
    and LOADED / REJECTED-DATE. A file whose inside date differs from its name is
    MCX answering a different day -- the date echo the validator also flags per row.
    """
    import hashlib as _h
    from mcxrv.stage1_intake.validation import parse_response_date
    from mcxrv.sources.csv_dir import CSVDirectorySource
    src = CSVDirectorySource(csv_dir)
    files = src._files()
    out, rejected = [], 0
    for path in files:
        hint = src._date_hint(path)
        df = src._read_one(path)
        inside, gold = None, 0
        if not df.empty:
            m = src._map_columns(df)
            if m["Date"] is not None and len(m):
                inside = parse_response_date(m["Date"].iloc[0], requested=hint)[0]
            sym = m["Symbol"].astype(str).str.strip().str.upper() if m["Symbol"] is not None else None
            if sym is not None:
                ok = sym.isin(specs.ALL_SYMBOLS)
                if "Instrument" in m.columns and m["Instrument"].notna().any():
                    ok &= m["Instrument"].astype(str).str.strip().str.upper().isin(
                        ("FUTCOM", "FUTIDX", "", "NONE", "NAN"))
                gold = int(ok.sum())
        status = "LOADED" if inside is not None and (hint is None or inside == hint) \
            else "REJECTED_DATE" if inside is not None else "UNREADABLE"
        rejected += status != "LOADED"
        out.append({"file": os.path.basename(path), "requested": hint, "returned": inside,
                    "gold_rows": gold, "status": status, "path": path})
    out.sort(key=lambda r: (str(r["requested"] or ""), r["file"]), reverse=True)
    for r in out[:show]:
        with open(r["path"], "rb") as fh:
            r["sha256"] = _h.sha256(fh.read()).hexdigest()
        r["saved_at"] = datetime.fromtimestamp(os.path.getmtime(r["path"]), timezone.utc)
    rows = [{k: v for k, v in r.items() if k != "path"} for r in out[:show]]
    return clean({"files": len(out), "rejected": rejected, "rows": rows,
                  "source_dir": csv_dir})


def engine_parameters(result) -> list[dict]:
    """The settings the run actually used, each with what it does. Read-only."""
    from mcxrv.stage4_simulate_deliver.replay_driver import BacktestConfig
    from mcxrv.stage2_normalize_signal.half_life import CarryConfig
    from mcxrv.stage3_gates_decision.risk import RiskBoardConfig
    sc, bc, cc = SignalConfig(z_entry=result.config.z_entry), BacktestConfig(), CarryConfig()
    return [
        {"group": "Signal", "name": "Unusual if |z| ≥", "value": sc.z_entry, "unit": "",
         "what": "How far the spread must sit from its trailing centre, in robust standard deviations."},
        {"group": "Signal", "name": "Minimum dislocation", "value": sc.min_abs_bp, "unit": "bp",
         "what": "Gaps smaller than this are tick noise, whatever the z-score says."},
        {"group": "Signal", "name": "Trailing window", "value": sc.lookback, "unit": "sessions",
         "what": "History used for the centre, scale, stationarity test and half-life."},
        {"group": "Signal", "name": "Minimum history", "value": sc.min_history, "unit": "sessions",
         "what": "A pair is not scored at all before this much history exists."},
        {"group": "Signal", "name": "Maximum half-life", "value": sc.max_halflife_sessions, "unit": "sessions",
         "what": "Spreads that revert slower than this cannot converge inside a contract's life."},
        {"group": "Model", "name": "Premium window", "value": cc.window_days, "unit": "sessions",
         "what": "Trailing sessions used to estimate each coin's premium, refit every "
                 "{} sessions.".format(cc.refit_every)},
        {"group": "Trading", "name": "Exit when |z| ≤", "value": bc.exit_z, "unit": "",
         "what": "The spread has converged; close both legs."},
        {"group": "Trading", "name": "Stop when |z| ≥", "value": bc.stop_z, "unit": "",
         "what": "The spread kept widening; close rather than hope."},
        {"group": "Trading", "name": "Enter only with", "value": bc.min_sessions_to_expiry_entry,
         "unit": "sessions left", "what": "Leaves room to converge before the nearer leg expires."},
        {"group": "Trading", "name": "Force exit at", "value": bc.min_sessions_to_expiry_exit,
         "unit": "sessions left", "what": "Nothing is ever held into the tender period."},
        {"group": "Trading", "name": "Maximum holding", "value": bc.max_holding_sessions, "unit": "sessions",
         "what": "A trade still open after this long is closed."},
        {"group": "Capital", "name": "Account capital", "value": result.config.account_capital, "unit": "₹",
         "what": "Cash available for margin and daily mark-to-market."},
        {"group": "Capital", "name": "Notional per leg", "value": bc.target_notional_per_leg, "unit": "₹",
         "what": "Target size of each leg before rounding to whole lots."},
        {"group": "Risk limits", "name": "Drawdown kill switch", "value": 100 * RiskBoardConfig().max_drawdown_pct,
         "unit": "% of capital", "what": "Below the peak account value by this much, no new exposure until a person reviews it."},
        {"group": "Risk limits", "name": "Block new exposure at", "value": 100 * RiskBoardConfig().utilisation_block,
         "unit": "% of capital", "what": "Margin used if the trade were added; above this the board refuses."},
        {"group": "Risk limits", "name": "Warn at", "value": 100 * RiskBoardConfig().utilisation_warn,
         "unit": "% of capital", "what": "Margin used above this: add at reduced size."},
        {"group": "Risk limits", "name": "Margin regime block", "value": 100 * RiskBoardConfig().span_rise_block_pct,
         "unit": "% rise", "what": "SPAN requirement up this much over a month: the market is getting riskier, so no new exposure."},
    ]


# ---------------------------------------------------------------- anomaly review
ANOMALY_TOP = 30


def anomaly_review(result) -> dict:
    """
    The Isolation Forest review queue, for a human. It never gates data or trades:
    the forest is fit on the whole sample, which is fine for an audit and would be
    look-ahead if it filtered a backtest (see mcxrv/stage1_intake/anomaly.py).
    """
    from mcxrv.stage1_intake import anomaly as an
    rows = result.rows
    if not an.available():
        return {"available": False, "reason": "scikit-learn is not installed"}
    if rows is None or rows.empty:
        return {"available": False, "reason": "no rows to score"}
    det = an.detect(rows)
    if det.empty:
        return {"available": False, "reason": "too few rows for the detector"}
    flagged = det[det["is_anomaly"]]
    top = an.top_reasons(det, n=ANOMALY_TOP)
    # Every flag on the newest session, for the evening digest: the history-wide top
    # list is dominated by old crash days and would hide what happened today.
    latest_day = det["date"].max()
    latest = flagged[(flagged["date"] == latest_day) & ~flagged["rule_flagged"]].sort_values("score")
    return clean({
        "available": True,
        "rows_scored": int(len(det)),
        "flagged": int(len(flagged)),
        "flagged_new": int((~flagged["rule_flagged"]).sum()),
        "flagged_known": int(flagged["rule_flagged"].sum()),
        "by_symbol": _records(an.summary(det), ["symbol", "rows", "flagged", "flag_rate",
                                                "flagged_already_known", "flagged_new"]),
        "by_feature": (flagged[~flagged["rule_flagged"]].groupby("top_feature").size()
                       .sort_values(ascending=False).rename("n").reset_index()
                       .to_dict("records")),
        "top": _records(top, ["date", "symbol", "contract_id", "score", "top_feature",
                              "top_z", "log_ret", "peer_dev_bp", "ret_in_ticks",
                              "d_log_volume"]),
        "latest_session": latest_day,
        "latest": _records(latest, ["date", "symbol", "contract_id", "score",
                                    "top_feature", "top_z"]),
    })


# ---------------------------------------------------------------- research
def research(result) -> dict:
    """
    The evidence behind the verdict, over the whole loaded history: does a wide spread
    converge (event study), what does it cost to trade (sweep and breakeven), what is
    structural rather than signal (premiums), and how much data stands behind it.

    The monitoring views look at recent sessions; this section deliberately does not,
    because a finding about whether an edge exists needs every session available.
    """
    q, pairs = result.quotes, result.pairs
    perf = result.performance or {}
    if q is None or q.empty:
        return {}
    dates = sorted(q["date"].unique())
    out = {"window": {"start": dates[0], "end": dates[-1], "sessions": len(dates),
                      "years": round((dates[-1] - dates[0]).days / 365.25, 2)}}

    cov = []
    for sym in specs.ALL_SYMBOLS:
        g = q[q["symbol"] == sym]
        if g.empty:
            continue
        cov.append({"symbol": sym, "first": g["date"].min(), "last": g["date"].max(),
                    "sessions": int(g["date"].nunique()),
                    "contracts": int(g["contract_id"].nunique()),
                    "untradeable_rows": int(g["settle_untradeable"].sum())
                    if "settle_untradeable" in g.columns else None})
    out["coverage"] = cov

    ccfg = costs.CostConfig(half_spread_bp_override=result.config.half_spread_bp)
    rt = costs.round_trip_bp(RESEARCH_NOTIONAL_INR, RESEARCH_HALF_SPREAD_BP,
                             RESEARCH_HALF_SPREAD_BP, ccfg)["total_bp"]
    if pairs is not None and not pairs.empty:
        es = eventstudy.study(pairs)
        if not es.empty:
            hurdle = eventstudy.cost_hurdle_table(es, rt)
            out["event_study"] = {
                "round_trip_bp": rt, "half_spread_bp": RESEARCH_HALF_SPREAD_BP,
                "rows": _records(hurdle, ["z_band", "horizon", "n", "mean_conv_bp",
                                          "hit_rate", "t_newey_west", "boot_ci_lo",
                                          "boot_ci_hi", "net_conv_bp",
                                          "ci_clears_costs"])}

    gross = perf.get("mean_gross_bp")
    be = (costs.breakeven_half_spread_bp(gross, RESEARCH_NOTIONAL_INR, ccfg)
          if gross is not None and np.isfinite(gross) and gross > 0 else 0.0)
    out["costs"] = {
        "notional_per_leg_inr": RESEARCH_NOTIONAL_INR,
        "sweep": _records(costs.cost_summary_table(RESEARCH_NOTIONAL_INR, ccfg),
                          ["half_spread_bp_per_leg", "statutory_bp", "execution_bp",
                           "round_trip_bp"]),
        "gross_bp": gross, "net_bp": perf.get("mean_net_bp"),
        "breakeven_half_spread_bp": be,
    }

    prem = result.premiums
    if prem is not None and not prem.empty:
        keep = set(sorted(prem["date"].unique())[::PREMIUM_EVERY_N_SESSIONS]
                   + [prem["date"].max()])
        out["premiums"] = _records(prem[prem["date"].isin(keep)].sort_values("date"),
                                   ["date", "symbol", "premium_bp"])

    t = result.trades
    if t is not None and not t.empty:
        t = t.assign(year=pd.to_datetime(t["entry_date"]).dt.year)

        def _agg(g):
            return {"n": int(len(g)), "gross_bp": float(g["gross_bp"].mean()),
                    "net_bp": float(g["net_bp"].mean()),
                    "hit_rate_net": float((g["net_bp"] > 0).mean()),
                    "net_pnl": float(g["net_pnl"].sum())}
        out["by_year"] = [{"year": int(y), **_agg(g)} for y, g in t.groupby("year")]
        out["by_pair"] = sorted(({"pair": p, **_agg(g)} for p, g in t.groupby("pair")),
                                key=lambda r: -r["n"])
        out["exit_reasons"] = {str(k): int(v) for k, v in
                               t["exit_reason"].value_counts().items()}
    bd = result.bt_daily
    if bd is not None and not bd.empty and "equity" in bd.columns:
        out["equity"] = _records(bd, ["date", "equity"])
    return clean(out)


# ---------------------------------------------------------------- build
def build(result, generated_at: datetime | None = None, archive: dict | None = None) -> dict:
    """PipelineResult -> snapshot dict."""
    cfg = result.config
    z_entry = float(cfg.z_entry)
    generated_at = generated_at or datetime.now(timezone.utc)
    asof = result.asof
    q, pairs, board = result.quotes, result.pairs, result.board

    # contracts on the latest session, front month per symbol
    contracts = []
    liq, avg_vol = {}, {}
    if q is not None and not q.empty:
        last = q[q["date"] == q["date"].max()]
        fm = normalize.front_month(last)
        liq = {}
        if result.rows is not None and not result.rows.empty:
            lr = result.rows[result.rows["date"] == result.rows["date"].max()]
            liq = lr.set_index("contract_id")[["tradable", "half_spread_bp"]].to_dict("index")
        recent = set(sorted(q["date"].unique())[-LIQ_AVG_SESSIONS:])
        avg_vol = (q[q["date"].isin(recent)].groupby("contract_id")["volume"].mean()
                   .to_dict())
        for r in fm.to_dict("records"):
            sp = specs.get(r["symbol"])
            extra = liq.get(r["contract_id"], {})
            av = avg_vol.get(r["contract_id"])
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
                "fine_g_per_lot": sp.fine_g_per_lot,
                "avg_volume_20": av,
                "liquidity": liquidity_level(r["symbol"], av),
            }))
    legs = {c["symbol"]: c for c in contracts}

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
                        "action", "reason", "contract_a", "contract_b", "sessions_a",
                        "sessions_b", "lots_a", "lots_b", "notional_exposure",
                        "marginal_margin", "initial_margin", "span_margin", "elm",
                        "delivery_margin", "floor_binds", "utilisation_if_added",
                        "buffer_state", "span_change_pct", "sessions_to_expiry",
                        "residual_frac")},
            }))

    # Pairs the signal could not score yet (too little history) still appear, so a
    # fresh real-data run says "warming up" instead of looking like missing data.
    counts["WARMUP"] = 0
    if q is not None and not q.empty:
        last_syms = set(q.loc[q["date"] == q["date"].max(), "symbol"])
        syms = [x for x in specs.ALL_SYMBOLS if x in last_syms]
        have = {p["pair"] for p in latest_pairs}
        n_sessions = int(q["date"].nunique())
        need = SignalConfig().min_history + 10      # pair_spreads' own threshold
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                name = "{}/{}".format(a, b)
                if name in have:
                    continue
                counts["WARMUP"] += 1
                latest_pairs.append(clean({
                    "pair": name, "sym_a": a, "sym_b": b,
                    "maturity_matched": specs.is_maturity_matched(a, b),
                    "spread_bp": None, "centre_bp": None, "scale_bp": None, "z": None,
                    "dislocation_bp": None, "halflife": None, "adf_t": None,
                    "gates": {g: g != "gate_history" and None for g in GATE_LABELS},
                    "signal_ok": False, "direction": None,
                    "status": "WARMUP",
                    "reason": "warming up: {} of {} sessions of history".format(
                        n_sessions, need),
                    "binding_gate": "history", "risk": None,
                }))
    order = {"ALERT": 0, "NO_TRADE": 1, "QUIET": 2, "WARMUP": 3}
    latest_pairs.sort(key=lambda p: (order[p["status"]], -abs(p["z"] or 0)))

    # What a reader needs beside the numbers: which contracts, how far apart their
    # expiries are, what would be traded, and the exact message the bot would send.
    from mcxrv import notify as _notify
    meta_for_msg = {"asof": asof, "sample_data": cfg.source == "synthetic",
                    "run_id": fingerprint(q)[:12] if q is not None and not q.empty else ""}
    for p in latest_pairs:
        # Legs are the contracts the risk board priced (enough runway to enter), which
        # near expiry is the next month, not the front month on the Overview page.
        risk_now = p.get("risk") or {}
        pair_legs = {}
        for sym, cid_key, s_key in ((p["sym_a"], "contract_a", "sessions_a"),
                                    (p["sym_b"], "contract_b", "sessions_b")):
            cid = risk_now.get(cid_key)
            if cid:
                av = avg_vol.get(cid)
                lrow = liq.get(cid, {})
                pair_legs[sym] = {"symbol": sym, "contract_id": cid,
                                  "expiry_date": cid.split("|", 1)[1],
                                  "sessions_to_expiry": risk_now.get(s_key),
                                  "avg_volume_20": av, "liquidity": liquidity_level(sym, av),
                                  "tradable": lrow.get("tradable"), "front_month": False}
            else:
                front = legs.get(sym) or {"symbol": sym}
                pair_legs[sym] = {**{k: front.get(k) for k in (
                    "symbol", "contract_id", "expiry_date", "sessions_to_expiry",
                    "avg_volume_20", "liquidity", "tradable")}, "front_month": True}
        la, lb = pair_legs[p["sym_a"]], pair_legs[p["sym_b"]]
        p["legs"] = [clean(la), clean(lb)]
        try:
            p["expiry_gap_days"] = abs((pd.Timestamp(la["expiry_date"]) -
                                        pd.Timestamp(lb["expiry_date"])).days)
        except (TypeError, KeyError, ValueError):
            p["expiry_gap_days"] = None
        p["trade"] = (clean(suggested_trade(p, pair_legs))
                      if p["status"] in ("ALERT", "NO_TRADE") else None)
        p["telegram"] = (_notify.format_message({"kind": "new_alert", "pair_state": p}, meta_for_msg)
                         if p["status"] == "ALERT" else None)

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
            # The range actually loaded, not the range requested: a CSV directory
            # run asks for "everything" and should report what that turned out to be.
            "start": q["date"].min() if q is not None and not q.empty else cfg.start,
            "end": q["date"].max() if q is not None and not q.empty else cfg.end,
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
                     "verdict": _verdict(
                         result.performance,
                         warming_up=latest_pairs[0]["reason"].split(": ", 1)[1]
                         if latest_pairs and counts["WARMUP"] == len(latest_pairs)
                         else None)},
        "research": research(result),
        "archive": archive,
        "parameters": engine_parameters(result),
        "account": clean((result.notes or {}).get("account") or {}),
        "anomalies": anomaly_review(result),
    }
    snap["meta"]["decision_window_sessions"] = (
        min(DECISION_LOG_SESSIONS, int(pairs["date"].nunique()))
        if pairs is not None and not pairs.empty else 0)
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
