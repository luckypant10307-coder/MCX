import os, sys, time
from dataclasses import replace
from datetime import date
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mcxrv import pipeline, published_margins
from mcxrv.stage2_normalize_signal import half_life as carry, normalize
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage2_normalize_signal import z_score as sig_mod
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage4_simulate_deliver import replay_driver as backtest
SPLIT = pd.Timestamp(2025, 1, 1)
cfg = pipeline.PipelineConfig(source="csv", start=None, end=date.today(), csv_dir=os.path.join(ROOT, "data", "raw"),
                              margins_file=os.path.join(ROOT, "data", "reference", "published_margins.json"))
q, _ = pipeline.load_quotes(cfg); q = normalize.add_normalized(q)
rows, daily, _ = carry.fit(q, carry.CarryConfig()); rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
resid = carry.symbol_residual(rows)
sp = margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows))
ms = {**margin.volatility_schedule(rows), **published_margins.schedule(published_margins.load(cfg.margins_file), rows["date"].unique(), sp.elm_for, sp.additional_for)}
for ze in (2.0, 2.5, 3.0, 3.5):
    scfg = sig_mod.SignalConfig(z_entry=ze)
    pairs = sig_mod.pair_spreads(resid, scfg)
    bc = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=sp, margin_schedule=ms, account_capital=cfg.account_capital, emit_position_daily=False)
    tr, _, _, _ = backtest.run(rows, sig_mod.lag_for_execution(pairs, 1), bc, costs.CostConfig())
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    i, o = tr[tr.entry_date < SPLIT], tr[tr.entry_date >= SPLIT]
    f = lambda t: "n=%3d gross %6.1f net %6.1f hit %3.0f%% pnl %8.0f" % (len(t), t.gross_bp.mean(), t.net_bp.mean(), 100*(t.net_bp > 0).mean(), t.net_pnl.sum()) if len(t) else "n=0"
    print("z>=%.1f | 2022-24: %s | 2025-26: %s" % (ze, f(i), f(o)), flush=True)
