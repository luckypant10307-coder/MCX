"""
End-to-end pipeline: ingest -> normalise -> carry fit -> signal -> backtest ->
attribution, with a written verdict.

Usage:
  python scripts/run_analysis.py --source synthetic --start 2021-01-01 --end 2026-09-18
  python scripts/run_analysis.py --source csv --csv-dir data/raw
  python scripts/run_analysis.py --source synthetic --null      # zero-edge world

The --null flag regenerates the world with residual_bp=0, so there is genuinely
nothing to find. Any material profit in that run is a bug in this code, not a
discovery. Run it every time before believing a positive result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv import (attribution, backtest, carry, costs, eventstudy, ingest,
                   ledger, liquidity, margin, normalize, risk)
from mcxrv import signal as sig_mod
from mcxrv import specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "csv", "live", "db", "db-raw"])
    p.add_argument("--csv-dir", default="data/raw")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-09-18")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--null", action="store_true",
                   help="zero-edge synthetic world: the falsification test")
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--half-spread-bp", type=float, default=None,
                   help="override the per-row half-spread assumption")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--maturity-matched-only", action="store_true")
    p.add_argument("--span-params", default=None,
                   help="JSON of real MCX risk parameters; else assumed")
    p.add_argument("--margin-limit", type=float, default=None,
                   help="cap on SPAN+ELM blocked at once")
    p.add_argument("--account-capital", type=float, default=1_000_000.0,
                   help="cash actually put up; drives margin calls")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def load_raw(args):
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.source == "synthetic":
        from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource
        cfg = SyntheticConfig(start=start, end=end, seed=args.seed,
                              residual_bp=0.0 if args.null else 22.0)
        src = SyntheticSource(cfg)
        raw = src.fetch_range(start, end)
        return raw, src

    if args.source == "csv":
        from mcxrv.sources.csv_dir import CSVDirectorySource
        src = CSVDirectorySource(args.csv_dir)
        return src.load_all(start, end), None

    if args.source == "db-raw":
        # Replay the stored raw tape through the full validation path. Use when a
        # validation rule has changed and history must be rebuilt under the new one.
        from mcxrv.sources.db_source import DbSource
        return DbSource().fetch_range(start, end, symbols=specs.ALL_SYMBOLS), None

    from mcxrv.sources.mcx_live import MCXLiveSource
    src = MCXLiveSource()
    return src.fetch_range(start, end, symbols=specs.ALL_SYMBOLS), None


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- 1. ingest ----------------
    section("1. DATA INGESTION AND VALIDATION")
    if args.source == "db":
        # The database IS the validated store: quotes were validated on the way in,
        # so re-running ingest here would only repeat work. The data-quality ledger
        # is read back from the run log rather than recomputed.
        from mcxrv.db import repo as dbrepo
        from datetime import datetime as _dt
        with dbrepo.connect() as _c:
            df = dbrepo.read_quotes(
                _c,
                start=_dt.strptime(args.start, "%Y-%m-%d").date(),
                end=_dt.strptime(args.end, "%Y-%m-%d").date(),
                symbols=specs.ALL_SYMBOLS)
            # Restrict to the runs that actually produced the CURRENT quote rows.
            # quote.run_id is rewritten on upsert, so a full reload leaves only the
            # newest run attributed, while incremental daily loads each keep their
            # own. Summing dq_flag unrestricted would double-count a reloaded range.
            dq = dbrepo._fetch_df(
                _c, "SELECT issue, sum(rows) AS rows FROM dq_flag "
                    "WHERE run_id IN (SELECT DISTINCT run_id FROM quote "
                    "                 WHERE run_id IS NOT NULL) "
                    "GROUP BY issue ORDER BY 2 DESC")
        if df.empty:
            print("No stored quotes. Run scripts/db_load.py first.")
            return
        rep = ingest.DataQualityReport(rows_in=len(df), rows_out=len(df))
        for r in dq.itertuples(index=False):
            rep.counts[r.issue] = int(r.rows)
        rep.note("source", "postgres (already validated on ingest)")
        tc = None
        print("Read {:,} validated quotes from Postgres, {} .. {}"
              .format(len(df), df["date"].min(), df["date"].max()))
        print("\nData quality flags accumulated across all ingest runs:")
        for k, v in sorted(rep.counts.items(), key=lambda kv: -kv[1]):
            print("    {:<34s} {:>9,}".format(k, v))
    else:
        raw, src = load_raw(args)
        df, rep, tc = ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)
        print(rep.summary())
    if df.empty:
        print("\nNo usable rows. Stopping.")
        return

    # ---------------- 2. normalise ----------------
    df = normalize.add_normalized(df)
    section("2. NORMALISATION TO INR PER 10 g FINE GOLD")
    print("Conversion uses quote_base_g, NOT trading_unit_g. Per-symbol constants:")
    tbl = []
    for s in sorted(df["symbol"].unique()):
        sp = specs.get(s)
        tbl.append({
            "symbol": s, "trading_unit_g": sp.trading_unit_g,
            "quote_base_g": sp.quote_base_g, "purity": sp.purity,
            "fine_g_per_lot": round(sp.fine_g_per_lot, 3),
            "norm_multiplier": round(10.0 / sp.fine_g_per_quote_base, 6),
        })
    print(pd.DataFrame(tbl).to_string(index=False))
    last = df[df["date"] == df["date"].max()]
    fm = normalize.front_month(last)
    if not fm.empty:
        print("\nLatest session ({}), front-month normalised prices:".format(df["date"].max()))
        print(fm[["symbol", "expiry_date", "close", "norm_close", "volume",
                  "sessions_to_expiry"]].to_string(index=False))
        print("\nRaw closes are NOT comparable; norm_close is. Spread of norm_close "
              "across symbols: {:.0f} bp".format(
                  (fm["norm_close"].max() / fm["norm_close"].min() - 1) * 10000))

    # ---------------- 3. carry fit ----------------
    section("3. TERM STRUCTURE FIT (level, carry slope, structural premium)")
    rows, daily, prem = carry.fit(df, carry.CarryConfig())
    ok = daily[daily["level"].notna()]
    print("sessions fitted: {} of {}".format(len(ok), len(daily)))
    print("mean carry slope (annualised): {:.4f}".format(ok["slope"].mean()))
    print("fit rmse, median: {:.1f} bp".format(ok["fit_rmse"].median() * 10000))
    if not prem.empty:
        pv = (prem[prem["date"] == prem["date"].max()]
              .set_index("symbol")["premium_bp"].round(1))
        print("\nStructural premium in force on the last session (bp vs GOLDM 995 bar):")
        print(pv.to_string())
        print("\nThis is the fabrication / form premium. A parity spread that does")
        print("not remove it reads it as a permanent 'opportunity'.")

    # ---------------- 4. liquidity ----------------
    section("4. LIQUIDITY AND TRADABILITY")
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
    cov = liquidity.coverage_report(rows)
    print(cov.to_string(index=False))
    print("\nvolume is not depth; half-spread figures are ASSUMPTIONS, swept below.")

    # ---------------- 5. signal ----------------
    section("5. SIGNAL CONSTRUCTION (trailing only)")
    sym_resid = carry.symbol_residual(rows)
    nf = liquidity.symbol_noise_floor(rows)
    scfg = sig_mod.SignalConfig(z_entry=args.z_entry)
    scored = sig_mod.score_symbols(sym_resid, scfg, noise_floor=nf)
    pairs = sig_mod.pair_spreads(sym_resid, scfg,
                                 maturity_matched_only=args.maturity_matched_only)
    print("symbol-level dislocation, sd by symbol (bp):")
    print((sym_resid.groupby("symbol")["resid_bp"].std().round(2)).to_string())
    print("\nmeasurement noise floor, median by symbol (bp):")
    print(nf.groupby("symbol")["noise_bp"].median().round(2).to_string())
    print("\nSignal-to-noise matters more than the z-score: a dislocation smaller")
    print("than the noise floor is quantisation and fit error, not information.")

    if not pairs.empty:
        print("\npair diagnostics (last session):")
        lastp = pairs[pairs["date"] == pairs["date"].max()]
        cols = ["pair", "maturity_matched", "spread_bp", "z", "halflife", "adf_t",
                "signal_ok"]
        print(lastp[cols].round(3).to_string(index=False))
        print("\ngate pass rates across the whole sample:")
        print(sig_mod.gate_report(pairs).round(4).to_string(index=False))
        fire = float(pairs["signal_ok"].mean())
        print("\nsignal fires on {:.2%} of pair-days. Silence is the default.".format(fire))

    # ---------------- 5b. event study ----------------
    section("6. EVENT STUDY: does a wide spread actually converge?")
    ccfg = costs.CostConfig(half_spread_bp_override=args.half_spread_bp)
    rt = costs.round_trip_bp(500_000.0, 5.0, 5.0, ccfg)["total_bp"]
    es = pd.DataFrame()
    if not pairs.empty:
        es = eventstudy.study(pairs)
        if not es.empty:
            print("Signed forward convergence, bp of one leg's notional.")
            print("Every pair-day counts here, tradable or not: this is the")
            print("STATISTICAL question, kept separate from executability.\n")
            show = ["z_band", "horizon", "n", "mean_conv_bp", "hit_rate",
                    "t_naive", "t_newey_west", "t_nonoverlap", "boot_p"]
            print(es[show].round(3).to_string(index=False))
            print("\nt_naive vs t_newey_west shows the overlap correction. The")
            print("|z| 0-1 row is the control band: extreme bands must beat it.")

            print("\nNetted against a round-trip cost of {:.1f} bp "
                  "(5 bp half-spread assumption):".format(rt))
            print(eventstudy.cost_hurdle_table(es, rt).round(3).to_string(index=False))

            bp_tbl = eventstudy.by_pair(pairs, z_min=args.z_entry, horizon=10)
            if not bp_tbl.empty:
                print("\nper-pair at |z| >= {:.1f}, 10-session horizon:".format(args.z_entry))
                print(bp_tbl.to_string(index=False))
                print("\nGOLDTEN lists only from 2025: check years_history before")
                print("treating any GOLDTEN pair as evidence.")

    # ---------------- 6. backtest ----------------
    section("7. WALK-FORWARD BACKTEST (executability)")
    # Scan ranges estimated from observed volatility when MCX's published risk
    # parameters are not supplied. Labelled as assumptions throughout.
    sp = (margin.SpanParameters.from_mcx_file(args.span_params)
          if args.span_params else
          margin.SpanParameters(price_scan_range_pct=margin.derive_scan_range(rows)))
    msched = margin.volatility_schedule(rows)
    bcfg = backtest.BacktestConfig(exit_z=scfg.z_exit, span_params=sp,
                                   margin_schedule=msched,
                                   margin_limit=args.margin_limit,
                                   account_capital=args.account_capital)
    # A settlement is published after the close, so a signal derived from day t's
    # settlement can only be filled at t+1. Shifting here rather than inside the
    # engine keeps the assumption visible and testable.
    pairs_exec = sig_mod.lag_for_execution(pairs, sessions=1)
    print("signals shifted +1 session for execution "
          "(no same-bar fills): {} -> {} rows".format(len(pairs), len(pairs_exec)))
    trades, bt_daily, diag, pos_daily = backtest.run(rows, pairs_exec, bcfg, ccfg)
    perf = backtest.performance(trades, bt_daily, bcfg)

    print("entry/skip diagnostics:")
    print(diag.to_string(index=False) if not diag.empty else "  (none)")
    print()
    if trades.empty:
        print("NO TRADES TAKEN. Every candidate failed a gate. That is a result,")
        print("not a failure: see the gate table above for the binding constraint.")
    else:
        for k in ["n_trades", "gross_pnl", "total_cost", "net_pnl",
                  "cost_share_of_gross", "hit_rate_gross", "hit_rate_net",
                  "mean_gross_bp", "mean_net_bp", "median_net_bp",
                  "mean_sessions_held", "sharpe", "ann_return", "ann_vol",
                  "max_drawdown_inr", "mean_residual_frac", "max_residual_frac",
                  "illiquid_exit_rate"]:
            if k in perf:
                v = perf[k]
                print("  {:<24s} {}".format(k, round(v, 4) if isinstance(v, float) else v))
        print("  {:<24s} {}".format("exit_reasons", perf.get("exit_reasons")))
        print()
        print("  per-trade net bp is the honest unit: gross {:.1f} bp minus".format(
            perf["mean_gross_bp"]))
        print("  {:.1f} bp of cost leaves {:.1f} bp net.".format(
            perf["mean_gross_bp"] - perf["mean_net_bp"], perf["mean_net_bp"]))

    # ---------------- 7. costs ----------------
    section("8. COST SENSITIVITY (the unobservable parameter)")
    notional = bcfg.target_notional_per_leg
    print("Round-trip cost of a 2-leg spread at Rs {:,.0f} per leg.".format(notional))
    print("Four crossings: open A, open B, close A, close B.\n")
    print(costs.cost_summary_table(notional, ccfg).to_string(index=False))
    if not trades.empty:
        gross = perf["mean_gross_bp"]
        be = costs.breakeven_half_spread_bp(abs(gross), notional, ccfg)
        print("\nMean gross edge: {:.1f} bp per trade.".format(gross))
        if np.isfinite(be):
            print("Breakeven half-spread: {:.1f} bp per leg per crossing.".format(be))
            print("The strategy exists only if the book is tighter than that on BOTH")
            print("legs, on the days it wants to trade. Check that against reality.")
        else:
            print("Breakeven half-spread exceeds the search range: costs do not bind.")

    # ---------------- 8c. per-pair capital ledger ----------------
    lcfg = ledger.LedgerConfig(account_capital=args.account_capital)
    led = ledger.build_ledger(pos_daily, lcfg)
    pos_summary = ledger.position_summary(led, trades, lcfg)
    acct = ledger.account_ledger(bt_daily, lcfg)

    # ---------------- 8b. margin ----------------
    section("9. MARGIN AND CAPITAL AT RISK (SPAN)")
    print("The SPAN ALGORITHM is implemented in full. The SPAN PARAMETERS are")
    print("ASSUMPTIONS: MCX publishes scan ranges and spread credits in separate")
    print("risk files, and nothing in bhavcopy determines them.")
    print("parameter source: {}".format(sp.source))
    print()
    print("price scan range in force (per symbol):")
    for k, v in sorted(sp.price_scan_range_pct.items()):
        print("  {:<12s} {:.2%}".format(k, v))
    print("  inter-commodity spread credit : {:.0%}".format(sp.inter_spread_credit))
    print("  extreme loss margin (ELM)     : {:.2%}, min {:.2%}".format(
        sp.elm_pct, sp.elm_min_pct))
    print("  initial margin floor          : {:.1%} of contract value ({} basis)"
          .format(sp.im_floor_pct, sp.im_floor_basis))
    print("  delivery period margin        : {}-day {:.0%} VaR, floor {:.1%}".format(
        sp.delivery_var_days, 0.99, sp.delivery_margin_floor_pct))
    print("  scan ranges are RE-ESTIMATED every 21 sessions from trailing volatility,")
    print("  so required margin rises when the market is disturbed.")

    if perf.get("n_trades", 0):
        print()
        for k in ("peak_margin", "mean_entry_margin", "margin_pct_notional",
                  "return_on_peak_margin", "ann_return_on_margin"):
            if k in perf:
                v = perf[k]
                txt = ("{:.2%}".format(v) if ("pct" in k or "return" in k)
                       else "Rs {:,.0f}".format(v))
                print("  {:<26s} {}".format(k, txt))
        print()
        print("  Return on capital, against two denominators:")
        print("    on the hardcoded Rs {:,.0f} base : {:+.2%}".format(
            bcfg.capital_base, perf.get("ann_return", float("nan"))))
        if "ann_return_on_margin" in perf:
            print("    on actual margin blocked       : {:+.2%}".format(
                perf["ann_return_on_margin"]))
        print()
        print("  THE INITIAL MARGIN FLOOR IS DECISIVE, and it cuts BOTH ways.")
        print("  SPAN grants a large inter-commodity credit on a matched spread, but")
        print("  MCX takes the HIGHER of SPAN and {:.0%} of contract value. The credit"
              .format(sp.im_floor_pct))
        print("  is absorbed entirely: margin lands on the floor at {:.0%} of notional,"
              .format(perf.get("margin_pct_notional", sp.im_floor_pct)))
        print("  not the ~3% that SPAN alone would ask.")
        print()
        print("  So more capital is blocked than SPAN implies, which makes the loss a")
        print("  smaller percentage than pure SPAN would show -- but it also means the")
        print("  spread earns NO capital efficiency here. It ties up the same 6% an")
        print("  outright would, while carrying none of the directional upside.")

        # Example sized from real data so the delivery VaR and concentration
        # surcharge actually engage; an example without volatility or open interest
        # silently disables both and makes the floor look unconditionally dominant.
        _last = rows[rows["date"] == rows["date"].max()]
        def _pick(symv):
            g = _last[_last["symbol"] == symv]
            return g.iloc[0] if len(g) else None
        _gm, _gp = _pick("GOLDM"), _pick("GOLDPETAL")
        _sig = lambda r: (float(msched.get((r["date"], r["symbol"]), 0.04)) / 3.5
                          if r is not None else 0.011)
        ex = [margin.MarginPosition(
                  "GOLDM", "x", 1,
                  float(_gm["close"]) if _gm is not None else 35485.0, 20,
                  sigma_daily=_sig(_gm),
                  open_interest=float(_gm["open_interest"]) if _gm is not None else None),
              margin.MarginPosition(
                  "GOLDPETAL", "y", -99,
                  float(_gp["close"]) if _gp is not None else 3596.5, 20,
                  sigma_daily=_sig(_gp),
                  open_interest=float(_gp["open_interest"]) if _gp is not None else None)]
        print()
        print("  margin vs the inter-commodity credit -- note SPAN moves while the")
        print("  margin actually blocked does not, because the floor binds:")
        print(margin.sweep_credit(ex, sp).to_string(index=False))
        print()
        print("  margin vs the price scan range:")
        print(margin.sweep_scan_range(ex, sp).to_string(index=False))
        print()
        print("  STRESS: margin when risk conditions deteriorate.")
        print("  Margin calls arrive exactly when a position is already losing, so a")
        print("  number computed under calm conditions is the least useful one.")
        print(margin.stress(ex, sp).to_string(index=False))
    else:
        print()
        print("  No trades, so no margin was ever blocked.")

    section("10. PER-PAIR CAPITAL LEDGER AND MARGIN CALLS")
    if pos_summary.empty:
        print("No positions, so no ledger.")
    else:
        print("Account capital: Rs {:,.0f}. Indian clearing settles MTM in CASH"
              .format(args.account_capital))
        print("daily, so the binding constraint on HOLDING a position is free cash,")
        print("not notional and not even margin. A position that cannot fund its")
        print("pay-in is liquidated by the broker, whatever the signal still says.")
        print()
        cols = ["pair", "sessions_held", "notional_exposure", "initial_margin",
                "span_margin", "elm", "delivery_margin", "total_capital_blocked",
                "mtm_worst", "min_cash_buffer_required", "peak_margin_utilisation",
                "margin_call_sessions", "was_forced_out"]
        cols = [c for c in cols if c in pos_summary.columns]
        print(pos_summary[cols].round(3).to_string(index=False))

        rep_cap = ledger.capital_report(acct, pos_summary, lcfg)
        print()
        for k in ("peak_required_margin", "peak_margin_utilisation", "min_free_cash",
                  "margin_call_sessions", "sessions_below_target_buffer",
                  "max_cash_buffer_required", "forced_exits",
                  "mean_return_on_capital_blocked"):
            if k in rep_cap:
                v = rep_cap[k]
                txt = ("{:.2%}".format(v) if ("utilisation" in k or "return" in k)
                       else ("Rs {:,.0f}".format(v) if isinstance(v, float)
                             else str(v)))
                print("  {:<32s} {}".format(k, txt))

        ev = ledger.call_events(acct)
        if not ev.empty:
            print()
            print("  MARGIN CALL EVENTS:")
            print(ev.to_string(index=False))
        else:
            print()
            print("  No margin call at this capital level. That is a statement about")
            print("  the ACCOUNT, not the strategy -- see the minimum below.")

        mv = ledger.minimum_viable_capital(pos_summary, acct)
        if mv:
            print()
            print("  Smallest account that could have run this without a forced exit:")
            print("    peak requirement      Rs {:,.0f}".format(mv["peak_requirement"]))
            print("    worst drawdown        Rs {:,.0f}".format(
                mv["worst_position_drawdown"]))
            print("    x{:.1f} safety         Rs {:,.0f}".format(
                mv["safety_multiple"], mv["minimum_viable_capital"]))
            print()
            print("  Run again with --account-capital below that to see the strategy")
            print("  get liquidated: forced exits cross a stressed book and cost more")
            print("  than chosen ones, so the same signal returns materially less.")

    # ---------------- 8d. live risk board ----------------
    section("11. LIVE RISK BOARD (should this pair be added to TODAY?)")
    rb_cfg = risk.RiskBoardConfig(account_capital=args.account_capital)
    asof = max(pairs["date"]) if not pairs.empty else None
    book = risk.open_book_from_trades(rows, trades, asof) if asof is not None else []
    rbd = risk.board(rows, pairs, rb_cfg, span_params=sp,
                     margin_schedule=margin.volatility_schedule(rows),
                     bt_cfg=bcfg, open_book=book)
    if rbd.empty:
        print("No pair could be priced on the latest session.")
    else:
        print("Session {}. Account capital Rs {:,.0f}. The book already open carries"
              .format(asof, args.account_capital))
        print("Rs {:,.0f} of requirement, and each pair below is costed MARGINALLY"
              .format(float(rbd["book_required"].iloc[0])))
        print("against it -- book-with minus book-without -- because SPAN nets across")
        print("the whole portfolio. Requirement change is measured over {} sessions"
              .format(rb_cfg.lookback_sessions))
        print("from {}, holding lots FIXED, so it reports risk and not sizing."
              .format(rbd["lookback_from"].dropna().iloc[0]
                      if rbd["lookback_from"].notna().any() else "n/a"))
        print()
        cols = [c for c in ("pair", "notional_exposure", "marginal_margin",
                            "utilisation_if_added", "buffer_state",
                            "span_change_pct", "z", "signal_ok", "action", "reason")
                if c in rbd.columns]
        print(rbd[cols].round(4).to_string(index=False))
        print()
        print("as a monitor would show it:")
        print()
        for line in risk.alerts(rbd):
            print("  " + line.replace(chr(10), chr(10) + "  "))
            print()
        rs = risk.summary(rbd)
        print("  {} pairs: {} addable, {} reduced size, {} blocked, {} unavailable"
              .format(rs["pairs"], rs["addable"], rs["reduced"], rs["blocked"],
                      rs["unavailable"]))
        print()
        print("  The verdict exists to say NO. A signal clearing its statistical")
        print("  gates is necessary and not sufficient: the position must also be")
        print("  fundable, and fundable under a deterioration that has not happened")
        print("  yet. Where a pair is blocked the BINDING reason is named, so the")
        print("  refusal can be argued with rather than merely obeyed.")
        if (rbd["marginal_margin"] - rbd["initial_margin"]).abs().max() < 1.0                 and bool(rbd["floor_binds"].any()):
            print()
            print("  Note: marginal margin equals standalone margin here. While the")
            print("  {:.0%} initial-margin FLOOR binds, the requirement is a flat"
                  .format(sp.im_floor_pct))
            print("  percentage of GROSS notional, which is additive -- so adding an")
            print("  offsetting pair earns no portfolio credit at all. The netting")
            print("  benefit SPAN computes is real and entirely unrealised.")

    # ---------------- 8. attribution ----------------
    section("12. ATTRIBUTION (is this relative value, or gold, or roll-down?)")
    if trades.empty or bt_daily.empty:
        print("No trades to attribute.")
    else:
        gb = attribution.gold_beta(bt_daily, daily)
        if "error" in gb:
            print("gold beta: " + gb["error"])
        else:
            b = gb["params"]["beta_gold"]
            lo, hi = gb["ci95"]["beta_gold"]
            print("  beta vs gold  : {:+.4f}   95% CI [{:+.4f}, {:+.4f}]".format(b, lo, hi))
            print("  alpha (daily) : {:+.6f}  t = {:+.2f}".format(
                gb["params"]["alpha"], gb["tstat"]["alpha"]))
            print("  R^2 vs gold   : {:.4f}".format(gb["r2"]))
            print("  verdict       : {}".format(gb["interpretation"]))

        tf = attribution.two_factor(bt_daily, daily)
        if "error" not in tf:
            print("\n  two-factor (gold + carry slope):")
            for nm in ["alpha", "beta_gold", "beta_slope"]:
                print("    {:<11s} {:+.6f}  t = {:+.2f}".format(
                    nm, tf["params"][nm], tf["tstat"][nm]))

        dec = attribution.decompose_trades(trades, rows, daily)
        summ = attribution.attribution_summary(dec)
        print("\n  per-trade P&L decomposition (bp of one leg's notional):")
        print(summ.round(3).to_string(index=False))
        print("\n  'alpha' is the residual-convergence term -- the only component that")
        print("  is the intended strategy. If 'roll' dominates it, this is a calendar")
        print("  carry harvester, not a cross-contract relative-value trade.")

        bb = attribution.block_bootstrap_mean(bt_daily["ret"].dropna(), block=10)
        if "error" not in bb:
            print("\n  block bootstrap on daily returns (respects overlap):")
            print("    mean {:+.6f}  95% CI [{:+.6f}, {:+.6f}]  p = {:.3f}".format(
                bb["mean"], bb["ci95"][0], bb["ci95"][1], bb["p_two_sided_vs_zero"]))

    # ---------------- 9. verdict ----------------
    section("13. VERDICT")
    verdict = build_verdict(args, rep, perf, pairs, trades, bt_daily, daily, ccfg, bcfg)
    print(verdict)

    # ---------------- persist ----------------
    tag = "null" if args.null else args.source
    rows.to_parquet(os.path.join(args.out_dir, "rows_{}.parquet".format(tag))) \
        if _has_parquet() else rows.to_csv(
            os.path.join(args.out_dir, "rows_{}.csv".format(tag)), index=False)
    if not pairs.empty:
        pairs.to_csv(os.path.join(args.out_dir, "pair_signals_{}.csv".format(tag)),
                     index=False)
    if not trades.empty:
        trades.to_csv(os.path.join(args.out_dir, "trades_{}.csv".format(tag)),
                      index=False)
    if not bt_daily.empty:
        bt_daily.to_csv(os.path.join(args.out_dir, "bt_daily_{}.csv".format(tag)),
                        index=False)
    daily.to_csv(os.path.join(args.out_dir, "carry_daily_{}.csv".format(tag)),
                 index=False)
    with open(os.path.join(args.out_dir, "performance_{}.json".format(tag)), "w") as f:
        json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                   for k, v in perf.items()}, f, indent=2, default=str)
    print("\nArtefacts written to {}/".format(args.out_dir))


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def build_verdict(args, rep, perf, pairs, trades, bt_daily, carry_daily,
                  ccfg, bcfg) -> str:
    L = []
    if args.null:
        L.append("NULL TEST (residual_bp = 0): the data contains no relative value")
        L.append("by construction. Expected result: no edge.")
        L.append("")

    n = perf.get("n_trades", 0)
    if n == 0:
        L.append("No trades. The gates held. Report the binding gate, not a Sharpe.")
        return "\n".join(L)

    net = perf.get("mean_net_bp", np.nan)
    gross = perf.get("mean_gross_bp", np.nan)
    sharpe = perf.get("sharpe", np.nan)
    L.append("{} trades. Gross {:+.1f} bp/trade, net {:+.1f} bp/trade after costs."
             .format(n, gross, net))
    if np.isfinite(sharpe):
        L.append("Net Sharpe {:.2f} on a Rs {:,.0f} capital base."
                 .format(sharpe, bcfg.capital_base))

    bb = attribution.block_bootstrap_mean(bt_daily["ret"].dropna(), block=10)
    signif = None
    if "error" not in bb:
        lo, hi = bb["ci95"]
        signif = not (lo <= 0 <= hi)
        # State the DIRECTION. "Excludes zero" on its own reads as evidence of edge,
        # when a significantly negative mean is the opposite: a reliable loss.
        if not signif:
            L.append("Block-bootstrap 95% CI on daily return includes zero "
                     "[{:+.2e}, {:+.2e}]: indistinguishable from no edge."
                     .format(lo, hi))
        elif hi < 0:
            L.append("Block-bootstrap 95% CI on daily return is entirely NEGATIVE "
                     "[{:+.2e}, {:+.2e}]: the losses are systematic, not bad luck."
                     .format(lo, hi))
        else:
            L.append("Block-bootstrap 95% CI on daily return is entirely positive "
                     "[{:+.2e}, {:+.2e}].".format(lo, hi))

    gb = attribution.gold_beta(bt_daily, carry_daily)
    if "error" not in gb:
        lo, hi = gb["ci95"]["beta_gold"]
        neutral = lo <= 0 <= hi
        L.append("Gold beta {:+.3f}, {}.".format(
            gb["params"]["beta_gold"],
            "neutral" if neutral else "NOT neutral -- directional gold risk present"))

    L.append("")
    if net <= 0:
        L.append("CONCLUSION: no edge survives costs. The gross dislocation is real")
        L.append("and measurable, but friction on these contracts consumes it. This")
        L.append("is a valid analytical outcome and the most likely honest answer.")
    elif signif is False:
        L.append("CONCLUSION: net return is positive but not statistically")
        L.append("distinguishable from zero once overlap is respected. Not tradable")
        L.append("on this evidence.")
    else:
        be = costs.breakeven_half_spread_bp(abs(gross), bcfg.target_notional_per_leg, ccfg)
        L.append("CONCLUSION: a net edge survives under the stated cost assumptions.")
        L.append("It rests on executing inside {:.1f} bp half-spread per leg. That".format(be)
                 if np.isfinite(be) else "It rests on the stated cost assumptions.")
        L.append("assumption is unobservable in bhavcopy and must be validated")
        L.append("against a real order book before any capital is risked.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
