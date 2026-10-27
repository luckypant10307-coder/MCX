"""
Load bhavcopy into Postgres, then optionally fit the model and persist its output.

Usage:
  python scripts/db_load.py --source synthetic --start 2021-01-01 --end 2026-09-18
  python scripts/db_load.py --source csv --csv-dir data/raw --model
  python scripts/db_load.py --source live --since-last --model --backtest
  python scripts/db_load.py --model-only          # refit from stored quotes

Two properties worth relying on:

  Idempotent. Every derived table upserts on its natural key, so re-running a range
  converges instead of accumulating. Only the raw tape appends, because a duplicate
  in the upstream response is part of what happened and belongs in the record.

  Re-runnable without re-fetching. `--model-only` rebuilds every model output from
  stored quotes. Validation rules and model parameters will change; the raw tape and
  the quotes do not have to be fetched again when they do.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv.stage1_intake import anomaly
from mcxrv.stage4_simulate_deliver import evaluation as attribution
from mcxrv.stage4_simulate_deliver import replay_driver as backtest
from mcxrv.stage2_normalize_signal import half_life as carry
from mcxrv.stage3_gates_decision import liquidity_cost as costs
from mcxrv.stage4_simulate_deliver import eventstudy
from mcxrv.stage1_intake import validation as ingest
from mcxrv.stage4_simulate_deliver import position_ledger as ledger
from mcxrv.stage2_normalize_signal import pair_match as liquidity
from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage2_normalize_signal import normalize
from mcxrv.stage3_gates_decision import risk
from mcxrv.stage1_intake import specs
from mcxrv.stage2_normalize_signal import z_score as sig_mod
from mcxrv.db import DbConfig, repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "csv", "live", "db"])
    p.add_argument("--csv-dir", default="data/raw")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-09-18")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--since-last", action="store_true",
                   help="start the day after the newest stored session")
    p.add_argument("--model", action="store_true", help="fit and persist the model")
    p.add_argument("--model-only", action="store_true",
                   help="skip fetching; refit from stored quotes")
    p.add_argument("--backtest", action="store_true",
                   help="run and persist the walk-forward backtest")
    p.add_argument("--anomaly", action="store_true",
                   help="run the Isolation Forest data-quality review queue")
    p.add_argument("--contamination", type=float, default=0.01,
                   help="expected anomaly share; a prior, not a measurement")
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--account-capital", type=float, default=1_000_000.0,
                   help="cash actually put up; drives margin calls")
    return p.parse_args()


def fetch(args, start: date, end: date):
    if args.source == "synthetic":
        from mcxrv.sources.synthetic import SyntheticConfig, SyntheticSource
        src = SyntheticSource(SyntheticConfig(start=start, end=end, seed=args.seed))
        return src.fetch_range(start, end)
    if args.source == "csv":
        from mcxrv.sources.csv_dir import CSVDirectorySource
        return CSVDirectorySource(args.csv_dir).load_all(start, end)
    from mcxrv.sources.mcx_live import MCXLiveSource
    return MCXLiveSource().fetch_range(start, end, symbols=specs.ALL_SYMBOLS)


def main():
    args = parse_args()
    cfg = DbConfig.from_env()
    print("db: {}".format(cfg.safe_repr()))

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    with repo.connect(cfg) as conn:
        # ---------------- fetch and persist the tape ----------------
        if not args.model_only:
            if args.since_last:
                got = conn.execute("SELECT max(trade_date) FROM quote").fetchone()[0]
                if got:
                    start = got + timedelta(days=1)
                    print("since-last: newest stored session {} -> fetching from {}"
                          .format(got, start))
                    if start > end:
                        print("nothing new to fetch.")
                        args.model_only = True

        if not args.model_only:
            run_id = repo.start_run(conn, args.source, start, end)
            conn.commit()
            print("\nrun {}: fetching {} .. {} from {}"
                  .format(run_id, start, end, args.source))
            try:
                raw = fetch(args, start, end)
                n_raw = repo.write_raw(conn, raw, run_id, args.source)
                print("  raw rows stored      : {:,}".format(n_raw))

                tidy, rep, tc = ingest.build_dataset(raw, symbols=specs.ALL_SYMBOLS)
                if tidy.empty:
                    repo.finish_run(conn, run_id, n_raw, 0, "failed",
                                    "no rows survived validation")
                    conn.commit()
                    raise SystemExit("no usable rows after validation")
                tidy = normalize.add_normalized(tidy)

                n_c = repo.upsert_contracts(conn, tidy)
                n_q = repo.upsert_quotes(conn, tidy, run_id)
                repo.write_dq(conn, run_id, rep)
                repo.finish_run(conn, run_id, n_raw, n_q, "ok")
                conn.commit()
                print("  contracts upserted   : {:,}".format(n_c))
                print("  quotes upserted      : {:,}".format(n_q))
                print("\n  data quality flags:")
                for k, v in sorted(rep.counts.items(), key=lambda kv: -kv[1]):
                    if v:
                        print("    {:<34s} {:>8,}".format(k, v))
            except SystemExit:
                raise
            except Exception as exc:
                conn.rollback()
                with repo.connect(cfg) as c2:
                    repo.finish_run(c2, run_id, 0, 0, "failed", str(exc)[:400])
                    c2.commit()
                raise

        # ---------------- model ----------------
        if args.model or args.model_only or args.backtest:
            print("\nrefitting model from stored quotes ...")
            q = repo.read_quotes(conn, symbols=specs.ALL_SYMBOLS)
            if q.empty:
                raise SystemExit("no stored quotes; load data first")
            # norm_close is persisted, but recompute so a spec change takes effect
            q = normalize.add_normalized(q)
            print("  sessions {} .. {}  ({:,} rows, {} contracts)".format(
                q["date"].min(), q["date"].max(), len(q),
                q["contract_id"].nunique()))

            rows, daily, prem = carry.fit(q, carry.CarryConfig())
            rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())
            sym_resid = carry.symbol_residual(rows)
            nf = liquidity.symbol_noise_floor(rows)
            scfg = sig_mod.SignalConfig(z_entry=args.z_entry)
            pairs = sig_mod.pair_spreads(sym_resid, scfg)

            n = repo.write_model(conn, daily=daily, prem=prem, rows=rows,
                                 sym_resid=sym_resid, pairs=pairs)
            conn.commit()
            print("  persisted: " + ", ".join(
                "{} {:,}".format(k, v) for k, v in n.items()))

            if not pairs.empty:
                live = pairs[pairs["date"] == pairs["date"].max()]
                print("  signal fires on {:.2%} of pair-sessions; {} live today"
                      .format(float(pairs["signal_ok"].mean()),
                              int(live["signal_ok"].sum())))

        # ---------------- anomaly review queue ----------------
        if args.anomaly:
            print("\nscanning for data-quality anomalies ...")
            if not anomaly.available():
                print("  scikit-learn not installed; skipping (pip install scikit-learn)")
            else:
                det = anomaly.detect(
                    rows, anomaly.AnomalyConfig(contamination=args.contamination))
                if det.empty:
                    print("  not enough rows to fit")
                else:
                    n_an = repo.write_anomalies(conn, det)
                    conn.commit()
                    print("  scored {:,} rows, flagged {:,}".format(
                        n_an, int(det["is_anomaly"].sum())))
                    print()
                    print(anomaly.summary(det).to_string(index=False))
                    new_hits = det[det["is_anomaly"] & ~det["rule_flagged"]]
                    print()
                    print("\n  {} flagged rows that NO existing rule caught. That is "
                          "the queue worth\n  a human's time. Nothing here blocks the "
                          "model: these flags are advisory,\n  and filtering a "
                          "backtest on them would be look-ahead."
                          .format(len(new_hits)))
                    if len(new_hits):
                        print()
                        print(anomaly.top_reasons(det, 8).to_string(index=False))

        # ---------------- backtest ----------------
        if args.backtest:
            print("\nrunning walk-forward backtest ...")
            pairs_exec = sig_mod.lag_for_execution(pairs, sessions=1)
            ccfg = costs.CostConfig()
            sp = margin.SpanParameters(
                price_scan_range_pct=margin.derive_scan_range(rows))
            bcfg = backtest.BacktestConfig(
                exit_z=scfg.z_exit, span_params=sp,
                margin_schedule=margin.volatility_schedule(rows),
                account_capital=args.account_capital)
            trades, bt_daily, diag, pos_daily = backtest.run(rows, pairs_exec, bcfg, ccfg)
            mrid = conn.execute(
                "SELECT COALESCE(max(model_run_id), 0) + 1 FROM bt_daily").fetchone()[0]
            n = repo.write_backtest(conn, trades, bt_daily, model_run_id=mrid)
            lcfg = ledger.LedgerConfig(account_capital=args.account_capital)
            led = ledger.build_ledger(pos_daily, lcfg)
            summary = ledger.position_summary(led, trades, lcfg)
            n["bt_position_ledger"] = repo.write_position_ledger(
                conn, summary, model_run_id=mrid)

            # Live risk board for the latest session, priced against whatever the
            # backtest still had open then. Stored so a refusal can be audited.
            asof_rb = max(pairs["date"])
            rbd = risk.board(
                rows, pairs,
                risk.RiskBoardConfig(account_capital=args.account_capital),
                span_params=sp, margin_schedule=bcfg.margin_schedule, bt_cfg=bcfg,
                open_book=risk.open_book_from_trades(rows, trades, asof_rb))
            n["risk_board"] = repo.write_risk_board(
                conn, rbd, model_run_id=mrid,
                account_capital=args.account_capital)
            conn.commit()
            perf = backtest.performance(trades, bt_daily, bcfg)
            print("  model_run_id {}: ".format(mrid) + ", ".join(
                "{} {:,}".format(k, v) for k, v in n.items()))
            if perf.get("n_trades"):
                print("  {} trades, gross {:+.1f} bp, net {:+.1f} bp per trade"
                      .format(perf["n_trades"], perf["mean_gross_bp"],
                              perf["mean_net_bp"]))
            else:
                print("  no trades cleared every gate")
            acct = ledger.account_ledger(bt_daily, lcfg)
            if not acct.empty:
                print("  peak margin Rs {:,.0f}, utilisation {:.1%}, "
                      "margin-call sessions {}, forced exits {}".format(
                          float(acct["required_margin"].max()),
                          float(acct["margin_utilisation"].max()),
                          int(acct["margin_call"].sum()),
                          int(summary["was_forced_out"].sum())
                          if not summary.empty else 0))
            if not rbd.empty:
                blocked = int((rbd["action"] == "do not add exposure").sum())
                print("  risk board {}: {} of {} pairs refused new exposure".format(
                    asof_rb, blocked, len(rbd)))
                # Priced pairs only. The unavailable ones are a sizing fact, not a
                # capital one, and a monitor listing only those says nothing about
                # the account -- which is what these lines exist to show.
                priced = rbd[rbd["action"] != "unavailable"]
                for line in risk.alerts(priced):
                    print("    " + line.replace(chr(10), chr(10) + "    "))

        print("\nstore contents:")
        print(repo.table_counts(conn).to_string(index=False))


if __name__ == "__main__":
    main()
