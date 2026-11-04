"""
Evaluate the cost hurdle (edge_gate) on real data.

  python scripts/experiments/cost_hurdle_eval.py

In-sample: trades entered 2022-2024. Out-of-sample: entered 2025-2026. The hurdle's
settings (haircut 0.7, 1.5x cost, 5 bp net) were fixed before this was run and are
not tuned here. Prints, for the gate off and on, under both brokerage schedules:
trades, mean gross / cost / net bp with a block-bootstrap 95% interval, net by
predicted-edge tercile (cut points from in-sample only), net by pair and by exit
reason, and a half-spread sensitivity sweep that is reported, never chosen from.
"""
import json
import os
import sys
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from mcxrv import pipeline, published_margins  # noqa: E402
from mcxrv.stage2_normalize_signal import half_life as carry, normalize  # noqa: E402
from mcxrv.stage2_normalize_signal import pair_match as liquidity  # noqa: E402
from mcxrv.stage2_normalize_signal import z_score as sig_mod  # noqa: E402
from mcxrv.stage3_gates_decision import liquidity_cost as costs  # noqa: E402
from mcxrv.stage3_gates_decision import margin_capital as margin  # noqa: E402
from mcxrv.stage4_simulate_deliver import replay_driver as backtest  # noqa: E402

SPLIT = pd.Timestamp(2025, 1, 1)
BLOCK, RESAMPLES, SEED = 5, 2000, 20261104


def block_ci(x: np.ndarray):
    """Mean and 95% interval from a moving-block bootstrap over trades in entry order.
    Blocks keep neighbouring (overlapping, correlated) trades together."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * BLOCK:
        return float(x.mean()) if n else None, None, None
    rng = np.random.default_rng(SEED)
    starts = np.arange(n - BLOCK + 1)
    k = int(np.ceil(n / BLOCK))
    means = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        idx = (rng.choice(starts, k)[:, None] + np.arange(BLOCK)).ravel()[:n]
        means[i] = x[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def fmt(m, lo, hi):
    if m is None:
        return "    n/a"
    return "{:+6.1f} [{:+6.1f}, {:+6.1f}]".format(m, lo, hi) if lo is not None else "{:+6.1f} [ too few ]".format(m)


def prepare():
    cfg = pipeline.PipelineConfig(source="csv", start=None, end=date.today(),
                                  csv_dir=os.path.join(ROOT, "data", "raw"),
                                  margins_file=os.path.join(ROOT, "data", "reference", "published_margins.json"))
    q, _ = pipeline.load_quotes(cfg)
    q = normalize.add_normalized(q)
    rows, daily, _ = carry.fit(q, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    scfg = sig_mod.SignalConfig(z_entry=cfg.z_entry)
    pairs = sig_mod.lag_for_execution(sig_mod.pair_spreads(carry.symbol_residual(rows), scfg), 1)
    sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
    ms = {**margin.volatility_schedule(rows),
          **published_margins.schedule(published_margins.load(cfg.margins_file),
                                       rows["date"].unique(), sp.elm_for, sp.additional_for)}
    base = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=sp, margin_schedule=ms,
                                   account_capital=cfg.account_capital, emit_position_daily=False)
    return rows, pairs, base


def run(rows, pairs, base, gate, mode, spread_mult=1.0):
    r = rows if spread_mult == 1.0 else rows.assign(half_spread_bp=rows["half_spread_bp"] * spread_mult)
    t, *_ = backtest.run(r, pairs, replace(base, cost_gate=gate), costs.CostConfig(brokerage_mode=mode))
    if len(t):
        t = t.assign(entry_date=pd.to_datetime(t["entry_date"]),
                     cost_bp=t["total_cost"] / t["notional_a"] * 1e4).sort_values("entry_date")
    return t


def summary(t):
    out = {}
    for name, part in (("in", t[t["entry_date"] < SPLIT] if len(t) else t),
                       ("out", t[t["entry_date"] >= SPLIT] if len(t) else t)):
        out[name] = {"n": int(len(part))}
        for col in ("gross_bp", "cost_bp", "net_bp"):
            m, lo, hi = block_ci(part[col].to_numpy()) if len(part) else (None, None, None)
            out[name][col] = {"mean": m, "lo": lo, "hi": hi}
        if len(part):
            out[name]["cost_split_bp"] = {k: float(part[k].mean()) for k in ("brokerage_bp", "statutory_bp", "execution_bp")}
    return out


def main():
    rows, pairs, base = prepare()
    print("sessions {}  ({} to {})".format(rows["date"].nunique(), rows["date"].min(), rows["date"].max()))
    results = {"settings": {"haircut": base.haircut, "hurdle_mult": base.hurdle_mult,
                            "min_net_bp": base.min_net_bp, "split": str(SPLIT.date()),
                            "block": BLOCK, "resamples": RESAMPLES}}
    trades = {}
    print("\n== Trades, mean bp per trade [95% block-bootstrap interval] ==")
    for mode in ("floor", "cap"):
        for gate in (False, True):
            key = "{} brokerage, gate {}".format(mode, "ON" if gate else "off")
            t = run(rows, pairs, base, gate, mode)
            trades[key] = t
            s = summary(t)
            results[key] = s
            print("\n" + key)
            for part, label in (("in", "2022-24"), ("out", "2025-26")):
                p = s[part]
                print("  {}  n={:3d}  gross {}  cost {}  net {}".format(
                    label, p["n"], fmt(*p["gross_bp"].values()), fmt(*p["cost_bp"].values()),
                    fmt(*p["net_bp"].values())))
                if p.get("cost_split_bp"):
                    c = p["cost_split_bp"]
                    print("            cost = brokerage {:.1f} + taxes/fees {:.1f} + bid/ask {:.1f} bp".format(
                        c["brokerage_bp"], c["statutory_bp"], c["execution_bp"]))

    print("\n== Does the predicted edge rank trades? Net bp by tercile of expected net (gate off) ==")
    results["terciles"] = {}
    for mode in ("floor", "cap"):
        t = trades["{} brokerage, gate off".format(mode)].dropna(subset=["exp_net_bp"])
        ins = t[t["entry_date"] < SPLIT]
        cuts = np.quantile(ins["exp_net_bp"], [1 / 3, 2 / 3]) if len(ins) >= 6 else None
        if cuts is None:
            continue
        t = t.assign(tercile=np.digitize(t["exp_net_bp"], cuts))
        print("  {} brokerage (cut points from 2022-24: {:.1f}, {:.1f} bp)".format(mode, *cuts))
        results["terciles"][mode] = {}
        for part, label in (("in", "2022-24"), ("out", "2025-26")):
            sub = t[(t["entry_date"] < SPLIT) if part == "in" else (t["entry_date"] >= SPLIT)]
            line = []
            for k, name in enumerate(("low", "mid", "high")):
                g = sub[sub["tercile"] == k]
                line.append("{} n={:2d} {:+6.1f}".format(name, len(g), g["net_bp"].mean() if len(g) else float("nan")))
                results["terciles"][mode].setdefault(part, {})[name] = {
                    "n": int(len(g)), "net_bp": float(g["net_bp"].mean()) if len(g) else None}
            print("    {}: {}".format(label, " | ".join(line)))

    for key in ("floor brokerage, gate off", "floor brokerage, gate ON"):
        t = trades[key]
        if not len(t):
            continue
        print("\n== {}: net bp by pair / by exit reason (all years) ==".format(key))
        by_pair = t.groupby("pair")["net_bp"].agg(["count", "mean"]).round(1)
        by_exit = t.groupby("exit_reason")["net_bp"].agg(["count", "mean"]).round(1)
        print(by_pair.to_string())
        print(by_exit.to_string())
        results[key]["by_pair"] = by_pair.reset_index().to_dict("records")
        results[key]["by_exit"] = by_exit.reset_index().to_dict("records")

    print("\n== Half-spread sensitivity (reported, not chosen from): net bp per trade, floor brokerage ==")
    results["sensitivity"] = []
    for m in (0.5, 1.0, 1.5, 2.0):
        row = {"half_spread_x": m}
        for gate in (False, True):
            s = summary(run(rows, pairs, base, gate, "floor", m))
            row["gate_on" if gate else "gate_off"] = {p: {"n": s[p]["n"], "net_bp": s[p]["net_bp"]["mean"]} for p in ("in", "out")}
        results["sensitivity"].append(row)
        f = lambda d: "{:+.1f} (n={})".format(d["net_bp"], d["n"]) if d["net_bp"] is not None else "n/a"
        print("  x{:.1f}: gate off  2022-24 {}  2025-26 {}   |  gate ON  2022-24 {}  2025-26 {}".format(
            m, f(row["gate_off"]["in"]), f(row["gate_off"]["out"]), f(row["gate_on"]["in"]), f(row["gate_on"]["out"])))

    out = os.path.join(ROOT, "data", "processed", "cost_hurdle_eval.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
