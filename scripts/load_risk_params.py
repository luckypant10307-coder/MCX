"""
Load published MCX risk parameters, replacing the model's assumptions.

WHERE TO GET THE FILES
----------------------
Three MCXCCL publications, downloaded BY HAND into a directory (MCX refuses
programmatic requests and this project does not work around that):

  * MCX Daily SPAN Risk Parameter File  -- the PC-SPAN input, one per business date.
    Price scan ranges, spread charges, spot-month charges, inter-commodity credits.
  * MCXCCL Risk and Collaterals         -- margin rate files and circulars: initial
    margin floor, ELM, additional/ad-hoc margins, delivery-period margin.
  * MCXCCL Cross-Margin Benefit         -- which contracts net against each other
    and at what rate. For a cross-contract gold strategy, the decisive number.

Usage:
  python scripts/load_risk_params.py --sniff data/risk/span_20260918.txt
  python scripts/load_risk_params.py --dir data/risk
  python scripts/load_risk_params.py --dir data/risk --persist --json out/params.json

WHY THE DATE MATTERS
--------------------
Each file describes the risk regime AS OF its business date. Loading one file and
applying it across a whole backtest is look-ahead in the margin dimension, and it
flatters the result precisely when margin was rising. This script reports COVERAGE --
how many sessions actually have a published file -- so the gap between "we have real
parameters" and "we have real parameters for the days we tested" stays visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv.stage3_gates_decision import spanfile
from mcxrv.stage1_intake import specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/risk",
                   help="directory of manually downloaded MCX parameter files")
    p.add_argument("--file", action="append", default=[],
                   help="a single file; repeatable")
    p.add_argument("--sniff", default=None,
                   help="describe a file's structure and exit, interpreting nothing")
    p.add_argument("--persist", action="store_true",
                   help="write the parsed parameters to Postgres")
    p.add_argument("--json", default=None,
                   help="write a SpanParameters JSON for --span-params")
    p.add_argument("--lenient", action="store_true",
                   help="do not raise on SPAN records that fail validation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.sniff:
        info = spanfile.sniff(args.sniff)
        print(json.dumps(info, indent=2, default=str))
        print()
        print("Nothing above has been interpreted as a parameter. Confirm the layout,")
        print("then load it -- the parser refuses a record it cannot validate rather")
        print("than guessing, because a margin rate read from the wrong column is")
        print("worse than no rate at all.")
        return

    paths = list(args.file)
    if not paths:
        if not os.path.isdir(args.dir):
            raise SystemExit(
                "no files. Download the MCX SPAN risk parameter file, the MCXCCL "
                "risk-and-collaterals margin file and the cross-margin benefit file "
                "into {} and re-run. Nothing here fetches them."
                .format(os.path.abspath(args.dir)))
        files = spanfile.load_dir(args.dir)
    else:
        files = []
        for p in paths:
            files.append(spanfile.load(p) if not args.lenient
                         else spanfile.parse_span_risk_file(p, strict=False))
    if not files:
        raise SystemExit("no readable parameter files in {}".format(args.dir))

    print("loaded {} file(s)".format(len(files)))
    for f in sorted(files, key=lambda r: r.business_date):
        print("  {:<12s} {}  {:>2d} commodities, {:>2d} spread credits{}".format(
            f.kind, f.business_date, len(f.commodities), len(f.credits),
            "  [{} unrecognised records]".format(sum(f.unrecognised.values()))
            if f.unrecognised else ""))
        for note in f.notes:
            print("      note: {}".format(note))

    print()
    print("published parameters, by commodity:")
    print("  {:<12s} {:>10s} {:>9s} {:>9s} {:>9s}".format(
        "commodity", "scan", "ELM", "delivery", "additional"))
    for f in sorted(files, key=lambda r: r.business_date):
        for code in sorted(f.commodities):
            c = f.commodities[code]
            try:
                scan = "{:.3%}".format(c.scan_pct())
            except spanfile.SpanFileError:
                scan = "{:,.0f} INR".format(c.price_scan_range)
            print("  {:<12s} {:>10s} {:>9s} {:>9s} {:>9s}".format(
                code, scan,
                "{:.3%}".format(c.elm_pct) if c.elm_pct is not None else "-",
                "{:.3%}".format(c.delivery_pct) if c.delivery_pct is not None else "-",
                "{:.3%}".format(c.additional_pct)
                if c.additional_pct is not None else "-"))

    creds = [c for f in files for c in f.credits]
    print()
    if creds:
        print("cross-margin benefit (published):")
        for c in sorted(creds, key=lambda x: (x.leg_a, x.leg_b)):
            print("  {:<12s} / {:<12s}  {:>7.2%}   ratio {}:{}".format(
                c.leg_a, c.leg_b, c.credit_rate, c.ratio_a, c.ratio_b))
        named = {c.key() for c in creds}
        universe = set(specs.ALL_SYMBOLS)
        missing = [(a, b) for a in sorted(universe) for b in sorted(universe)
                   if a < b and (a, b) not in named]
        if missing:
            print()
            print("  NOT recognised, so credited at ZERO:")
            for a, b in missing:
                print("    {} / {}".format(a, b))
            print()
            print("  That is a result, not a gap. A spread MCX does not recognise")
            print("  carries the full scanning risk of both legs, and the strategy")
            print("  pays for two outright positions to hold one relative-value bet.")
    else:
        print("no cross-margin credits published in these files.")

    sp = spanfile.to_span_parameters(files)
    base = margin.SpanParameters()
    print()
    print("assumption -> published:")
    for fname in ("price_scan_range_pct", "elm_pct_by_symbol",
                  "additional_pct_by_symbol", "delivery_margin_floor_pct",
                  "intra_spread_charge_pct", "inter_spread_credit"):
        was, now = getattr(base, fname), getattr(sp, fname)
        if was != now:
            print("  {:<28s} {}  ->  {}".format(fname, was, now))
    if sp.inter_spread_credit_pairs:
        print("  {:<28s} {} published pair rates".format(
            "inter_spread_credit_pairs", len(sp.inter_spread_credit_pairs)))
    print()
    print("  source: {}".format(sp.source))
    print()
    print("  Unchanged fields are STILL ASSUMPTIONS. The initial-margin floor, the")
    print("  delivery VaR horizon and the concentration surcharge come from the")
    print("  contract specification and circulars, not from these files.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {
            "price_scan_range_pct": sp.price_scan_range_pct,
            "inter_spread_credit": sp.inter_spread_credit,
            "inter_spread_credit_pairs": {"{}|{}".format(*k): v for k, v
                                          in sp.inter_spread_credit_pairs.items()},
            "elm_pct": sp.elm_pct,
            "elm_pct_by_symbol": sp.elm_pct_by_symbol,
            "additional_margin_pct": sp.additional_margin_pct,
            "additional_pct_by_symbol": sp.additional_pct_by_symbol,
            "delivery_margin_floor_pct": sp.delivery_margin_floor_pct,
            "intra_spread_charge_pct": sp.intra_spread_charge_pct,
            "source": sp.source,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print()
        print("wrote {} -- pass it as --span-params".format(args.json))

    if args.persist:
        from mcxrv.db import DbConfig, repo
        cfg = DbConfig.from_env()
        with repo.connect(cfg) as conn:
            n = repo.write_risk_params(conn, files)
            conn.commit()
        print()
        print("persisted: " + ", ".join("{} {}".format(k, v) for k, v in n.items()))

    sched = spanfile.schedule_from_files(files)
    print()
    print("scan-range schedule covers {} (date, commodity) pairs across {} sessions."
          .format(len(sched), len({d for d, _ in sched})))
    print("Sessions with no file keep the ESTIMATED schedule and stay flagged as")
    print("estimated. Back-filling them from the newest file would be look-ahead in")
    print("the margin dimension -- and it would flatter the result exactly when")
    print("margin was rising, which is when the strategy could least afford it.")


if __name__ == "__main__":
    main()
