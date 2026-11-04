"""
Cost-aware gate experiment. Same data and engine as the live pipeline; only the
entry rule changes. The safety margin is chosen on trades ENTERED before 2025
(in-sample) and the chosen setting is then read off for 2025-2026 (out-of-sample).
"""
import json
import os
import sys
import time
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mcxrv import pipeline  # noqa: E402
from mcxrv.stage1_intake import validation as ingest  # noqa: E402
from mcxrv.stage2_normalize_signal import half_life as carry, normalize  # noqa: E402
from mcxrv.stage2_normalize_signal import pair_match as liquidity  # noqa: E402
from mcxrv.stage2_normalize_signal import z_score as sig_mod  # noqa: E402
from mcxrv.stage3_gates_decision import edge  # noqa: E402
from mcxrv.stage3_gates_decision import liquidity_cost as costs  # noqa: E402
from mcxrv.stage3_gates_decision import margin_capital as margin  # noqa: E402
from mcxrv.stage4_simulate_deliver import replay_driver as backtest  # noqa: E402

SPLIT = date(2025, 1, 1)
t0 = time.time()
cfg = pipeline.PipelineConfig(source="csv", start=None, end=date.today(),
                              csv_dir=os.path.join(ROOT, "data", "raw"),
                              margins_file=os.path.join(ROOT, "data", "reference", "published_margins.json"))
quotes, rep = pipeline.load_quotes(cfg)
quotes = normalize.add_normalized(quotes)
rows, daily, prem = carry.fit(quotes, carry.CarryConfig())
rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
scfg = sig_mod.SignalConfig(z_entry=cfg.z_entry)
pairs = sig_mod.pair_spreads(carry.symbol_residual(rows), scfg)
sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
msched = margin.volatility_schedule(rows)
from mcxrv import published_margins  # noqa: E402
msched = {**msched, **published_margins.schedule(published_margins.load(cfg.margins_file),
                                                 rows["date"].unique(), sp.elm_for, sp.additional_for)}
print("prepared in %.0fs; sessions %d" % (time.time() - t0, rows["date"].nunique()), flush=True)

# Expected gain: learned walk-forward, and the optimistic rule for comparison.
learned = edge.predict_walk_forward(pairs, edge.EdgeConfig())
learned["rule_gain_bp"] = edge.rule_gain(learned, exit_z=scfg.z_exit)
sig = learned[learned["signal_ok"]]
print("signals %d, predicted %d (first prediction %s)" % (
    len(sig), sig["exp_gain_bp"].notna().sum(),
    sig.loc[sig["exp_gain_bp"].notna(), "date"].min()), flush=True)

# How good is the prediction? Rank correlation with the real outcome, out-of-sample.
oc = edge.outcomes(learned, edge.EdgeConfig().horizon)
chk = oc[oc["signal_ok"] & oc["exp_gain_bp"].notna() & oc["outcome_bp"].notna()]
for name, part in (("in-sample", chk[pd.to_datetime(chk["date"]) < pd.Timestamp(SPLIT)]),
                   ("out-of-sample", chk[pd.to_datetime(chk["date"]) >= pd.Timestamp(SPLIT)])):
    if len(part) > 5:
        print("  %s: n=%d  corr(pred, actual)=%.2f  rule corr=%.2f  mean actual=%.1f  mean pred=%.1f" % (
            name, len(part), part["exp_gain_bp"].corr(part["outcome_bp"], method="spearman"),
            part["rule_gain_bp"].corr(part["outcome_bp"], method="spearman"),
            part["outcome_bp"].mean(), part["exp_gain_bp"].mean()), flush=True)

pairs_exec = sig_mod.lag_for_execution(learned, sessions=1)
base = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=sp, margin_schedule=msched,
                               account_capital=cfg.account_capital, emit_position_daily=False)
ccfg = costs.CostConfig()


def summarise(trades):
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    res = {}
    for name, part in (("in", t[t["entry_date"] < pd.Timestamp(SPLIT)]),
                       ("out", t[t["entry_date"] >= pd.Timestamp(SPLIT)]), ("all", t)):
        res[name] = {"n": int(len(part)),
                     "gross_bp": float(part["gross_bp"].mean()) if len(part) else None,
                     "net_bp": float(part["net_bp"].mean()) if len(part) else None,
                     "net_pnl": float(part["net_pnl"].sum()) if len(part) else 0.0,
                     "hit": float((part["net_bp"] > 0).mean()) if len(part) else None}
    return res


variants = [("baseline (today)", None, "exp_gain_bp")]
for m in (0.0, 5.0, 10.0):
    variants.append(("rule gain, margin %g" % m, m, "rule_gain_bp"))
    variants.append(("learned gain, margin %g" % m, m, "exp_gain_bp"))

results = {}
for name, m, col in variants:
    t1 = time.time()
    trades, daily_bt, diag, _ = backtest.run(rows, pairs_exec, replace(base, min_edge_bp=m, edge_col=col), ccfg)
    results[name] = {"summary": summarise(trades),
                     "skipped_for_cost": int(diag.get("skip_edge_below_cost", 0)) if isinstance(diag, dict) else None}
    s = results[name]["summary"]
    print("%-28s all n=%3d net %6.1f bp | 2022-24 n=%3d net %6.1f | 2025-26 n=%3d net %6.1f bp  pnl %9.0f | %.0fs" % (
        name, s["all"]["n"], s["all"]["net_bp"] or 0, s["in"]["n"], s["in"]["net_bp"] or 0,
        s["out"]["n"], s["out"]["net_bp"] or 0, s["out"]["net_pnl"], time.time() - t1), flush=True)

json.dump(results, open(os.path.join(ROOT, "data", "processed", "edge_experiment.json"), "w"), indent=1, default=str)
print("done in %.0fs" % (time.time() - t0))
