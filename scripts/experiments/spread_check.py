"""
Compare the cost model's assumed half-spread with spreads estimated from each
contract's own daily high, low and close (Corwin-Schultz; Abdi-Ranaldo).

  python scripts/experiments/spread_check.py

Groups contract-days by symbol, year, and whether the contract was the most traded
of its symbol that day (busy) or not (thin). Reports, per group: the model's mean
half-spread and the estimators' averaged half-spread, all in bp.

Read with tests/test_spread_estimate.py in mind: at gold's daily volatility these
estimators cannot resolve spreads of a few bp, so agreement on busy contracts says
little; a large gap on thin contracts is the signal worth acting on.
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from mcxrv import pipeline  # noqa: E402
from mcxrv.stage2_normalize_signal import half_life as carry, normalize  # noqa: E402
from mcxrv.stage2_normalize_signal import pair_match as liquidity  # noqa: E402
from mcxrv.stage2_normalize_signal import spread_estimate as se  # noqa: E402


def main():
    cfg = pipeline.PipelineConfig(source="csv", start=None, end=date.today(),
                                  csv_dir=os.path.join(ROOT, "data", "raw"))
    q, _ = pipeline.load_quotes(cfg)
    q = normalize.add_normalized(q)
    rows, daily, _ = carry.fit(q, carry.CarryConfig())
    rows = liquidity.annotate(rows, daily, liquidity.LiquidityConfig())

    cs = se.corwin_schultz(q)
    ar = se.abdi_ranaldo(q)
    m = (rows[["date", "symbol", "contract_id", "volume", "half_spread_bp", "tradable"]]
         .merge(cs, on=["contract_id", "date"], how="left")
         .merge(ar, on=["contract_id", "date"], how="left"))
    m["year"] = pd.to_datetime(m["date"]).dt.year
    busiest = m.groupby(["date", "symbol"])["volume"].transform("max")
    m["kind"] = np.where(m["volume"] == busiest, "busy", "thin")
    m = m[m["tradable"]]

    print("Half-spread, bp: cost model vs estimators (tradable contract-days only)\n")
    print("{:<11} {:<5} {:>4} {:>6} {:>7}   {:>6} {:>6} {:>6}".format(
        "symbol", "kind", "year", "days", "volume", "model", "CS", "AR"))
    for (sym, kind, yr), g in m.groupby(["symbol", "kind", "year"]):
        if len(g) < 40:
            continue
        print("{:<11} {:<5} {:>4} {:>6} {:>7.0f}   {:>6.1f} {:>6.1f} {:>6.1f}".format(
            sym, kind, yr, len(g), g["volume"].median(), g["half_spread_bp"].mean(),
            se.average(g["cs_half_spread_bp"]), g["ar_half_spread_bp"].mean()))


if __name__ == "__main__":
    main()
