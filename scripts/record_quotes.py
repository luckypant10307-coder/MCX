"""
Record bid/ask quotes by hand and compare them with the cost model's assumed spread.

The bhavcopy has no bid or ask, so the model's half-spread is an assumption
(pair_match.model_half_spread_bp). This is the check against reality: read a quote
off your broker's screen, record it, and after a few weeks the report shows per
contract how the observed half-spread compares with the model's.

  python scripts/record_quotes.py add GOLDM 2026-12-04 --bid 141450 --ask 141452 --source "Zerodha Kite"
  python scripts/record_quotes.py add "GOLDPETAL|2026-11-30" --bid 14394 --ask 14395
  python scripts/record_quotes.py report

Rows go to data/reference/observed_spreads.csv:
  timestamp, contract_id, bid, ask, source
Prices are as quoted on screen (the contract's own quote basis). The timestamp
defaults to now in IST; pass --time "2026-11-06 14:30" for a quote noted earlier.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = os.path.join("data", "reference", "observed_spreads.csv")
FIELDS = ["timestamp", "contract_id", "bid", "ask", "source"]
IST = timezone(timedelta(hours=5, minutes=30))


def contract_id(symbol_or_id: str, expiry: str | None) -> str:
    """'GOLDM' + '2026-12-04' or 'GOLDM|2026-12-04' -> 'GOLDM|2026-12-04', checked."""
    from mcxrv.stage1_intake import specs
    if "|" in symbol_or_id:
        sym, exp = symbol_or_id.split("|", 1)
    else:
        sym, exp = symbol_or_id, expiry
    sym = sym.strip().upper()
    if sym not in specs.ALL_SYMBOLS:
        raise ValueError("unknown contract {!r}; expected one of {}".format(sym, ", ".join(specs.ALL_SYMBOLS)))
    if not exp:
        raise ValueError("give the expiry date, e.g. {} 2026-12-04".format(sym))
    datetime.strptime(exp.strip(), "%Y-%m-%d")
    return "{}|{}".format(sym, exp.strip())


def validate(bid: float, ask: float) -> None:
    if not (bid > 0 and ask > 0):
        raise ValueError("bid and ask must be positive prices")
    if ask < bid:
        raise ValueError("ask ({}) is below bid ({}): check the order".format(ask, bid))
    if (ask - bid) / ((ask + bid) / 2) > 0.02:
        raise ValueError("spread over 2% of price: a typo is more likely than a quote")


def add(path: str, cid: str, bid: float, ask: float, source: str, when: datetime) -> dict:
    validate(bid, ask)
    row = {"timestamp": when.astimezone(IST).isoformat(timespec="minutes"), "contract_id": cid,
           "bid": bid, "ask": ask, "source": source}
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def report(path: str, csv_dir: str = os.path.join("data", "raw")):
    """
    Per contract: observed half-spread (bp of mid) against the model's assumption
    for the same contract on the same session, or the latest session before it.
    Returns a DataFrame; prints it.
    """
    import pandas as pd
    from mcxrv import pipeline
    from mcxrv.stage2_normalize_signal import normalize
    from mcxrv.stage2_normalize_signal.pair_match import model_half_spread_bp

    if not os.path.exists(path):
        print("No quotes recorded yet. Add some with: python scripts/record_quotes.py add ...")
        return pd.DataFrame()
    obs = pd.read_csv(path)
    if obs.empty:
        print("No quotes recorded yet.")
        return obs
    obs["when"] = pd.to_datetime(obs["timestamp"], utc=True).dt.tz_convert(IST)
    obs["date"] = obs["when"].dt.date
    mid = (obs["bid"] + obs["ask"]) / 2
    obs["observed_bp"] = (obs["ask"] - obs["bid"]) / 2 / mid * 1e4

    start = min(obs["date"]) - timedelta(days=10)
    q, _ = pipeline.load_quotes(pipeline.PipelineConfig(
        source="csv", start=start, end=max(obs["date"]), csv_dir=csv_dir))
    if q is None or q.empty:
        obs["model_bp"] = float("nan")
    else:
        q = normalize.add_normalized(q)
        q["model_bp"] = model_half_spread_bp(q["volume"], q["tick_bp_of_norm"])
        q = q[["contract_id", "date", "model_bp"]].assign(day=pd.to_datetime(q["date"])).sort_values("day")
        obs = obs.assign(day=pd.to_datetime(obs["date"])).sort_values("day")
        # the quote's own session, or the latest one before it (never a later one)
        obs = pd.merge_asof(obs, q.drop(columns="date"), on="day", by="contract_id", direction="backward")

    out = (obs.groupby("contract_id")
           .agg(quotes=("observed_bp", "size"), observed_bp=("observed_bp", "median"),
                model_bp=("model_bp", "mean"), first=("date", "min"), last=("date", "max"))
           .reset_index())
    out["observed_over_model"] = out["observed_bp"] / out["model_bp"]
    pd.set_option("display.width", 140)
    print("Half-spread in bp of mid price: observed (median of your quotes) vs the model's assumption")
    print(out.round(3).to_string(index=False))
    few = out[out["quotes"] < 10]
    if len(few):
        print("\nFewer than 10 quotes for {} contract(s): treat their ratio as a first look, not a measurement."
              .format(len(few)))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="record one quote")
    a.add_argument("contract", help="symbol (GOLDM) or contract id (GOLDM|2026-12-04)")
    a.add_argument("expiry", nargs="?", help="YYYY-MM-DD, when the first argument is a symbol")
    a.add_argument("--bid", type=float, required=True)
    a.add_argument("--ask", type=float, required=True)
    a.add_argument("--source", default="manual", help="where the quote was read, e.g. 'Zerodha Kite'")
    a.add_argument("--time", default=None, help="'YYYY-MM-DD HH:MM' in IST; default now")
    a.add_argument("--file", default=PATH)
    r = sub.add_parser("report", help="compare recorded quotes with the model")
    r.add_argument("--file", default=PATH)
    args = p.parse_args(argv)

    if args.cmd == "add":
        try:
            cid = contract_id(args.contract, args.expiry)
            when = (datetime.strptime(args.time, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
                    if args.time else datetime.now(IST))
            row = add(args.file, cid, args.bid, args.ask, args.source, when)
        except ValueError as e:
            print("Not recorded: {}".format(e))
            return 1
        hs = (row["ask"] - row["bid"]) / 2 / ((row["ask"] + row["bid"]) / 2) * 1e4
        print("Recorded {} {}  bid {} ask {}  half-spread {:.2f} bp".format(
            row["timestamp"], cid, row["bid"], row["ask"], hs))
        return 0
    report(args.file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
