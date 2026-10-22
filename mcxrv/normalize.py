"""
Put every contract on one basis: INR per 10 g of FINE gold.

The conversion uses `quote_base_g`, NOT `trading_unit_g`. That distinction is the
crux of this problem. GOLDM has a 100 g trading unit but is quoted per 10 g, so it
already shares GOLDTEN's quotation base and differs from it only in purity
(0.999 / 0.995 = 1.00402, i.e. GOLDM should quote ~40 bp BELOW GOLDTEN on the raw
number, mechanically, with no mispricing whatsoever).

Dividing GOLDM by its 100 g trading unit -- the intuitive but wrong move -- makes
it look 10x cheap and produces a spread series that is pure arithmetic error.
`fine_g_per_lot` exists separately and is used only for hedge ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import specs


SPEC_FIELDS = ("purity", "quote_base_g", "trading_unit_g", "fine_g_per_lot",
               "tick_size_inr")


def _listing_dates(df: pd.DataFrame) -> pd.Series:
    """
    The governing date per row: the earliest session on which its contract appears.

    A spec revision applies to contracts listed after it, not to contracts already
    trading, so the contract's listing date -- not the session date -- selects the
    version. Within the loaded window the first observed session is the best
    available proxy; when reading from the database the stored snapshot is used
    instead and this proxy is not consulted.
    """
    return df.groupby("contract_id")["date"].transform("min")


def add_normalized(df: pd.DataFrame, prefer_stored: bool = True) -> pd.DataFrame:
    """
    Attach the common-basis price and its logarithm.

    Adds:
      norm_close        INR per 10 g fine gold (the comparable price)
      log_norm          natural log of the above; the modelling variable
      fine_g_per_lot    grams of pure gold per lot (the hedging unit)
      lot_notional      rupee value of one lot at today's quote
      purity, quote_base_g, trading_unit_g   carried through for transparency
      spec_source       'stored' or 'registry' -- which specification was applied
      spec_mismatch     stored and registry specs disagree for this row

    Specification precedence, when `prefer_stored` is on:

      1. Columns carried on the frame (`spec_purity`, `spec_quote_base_g`, ...),
         which `db.repo.read_quotes` joins from the per-contract snapshot captured
         at ingest. This is what stops a future spec revision from retroactively
         rewriting historical prices.
      2. The in-code registry, selected by the contract's listing date.

    Where both exist and disagree, the stored value WINS and `spec_mismatch` is set.
    A mismatch is a fact worth surfacing -- either the registry was corrected or a
    revision was never recorded -- and silently preferring either one would hide it.
    """
    out = df.copy()

    # --- registry values, selected by listing date ---------------------------
    if specs.has_revisions():
        listed = _listing_dates(out)
        pairs = pd.Series(list(zip(out["symbol"], listed)), index=out.index)
        uniq = {p: specs.get(p[0], p[1]) for p in pairs.unique()}
        reg = {f: pairs.map(lambda p: getattr(uniq[p], f)).astype(float)
               for f in SPEC_FIELDS}
    else:
        # Fast path: one open-ended version per symbol, so the listing date is
        # irrelevant and a per-symbol lookup is enough.
        sp_map = {s: specs.get(s) for s in out["symbol"].unique()}
        reg = {f: out["symbol"].map(lambda s: getattr(sp_map[s], f)).astype(float)
               for f in SPEC_FIELDS}

    # --- stored snapshot, where the caller carried one ------------------------
    stored_cols = {f: "spec_" + f for f in SPEC_FIELDS}
    have_stored = prefer_stored and all(
        c in out.columns for c in ("spec_purity", "spec_quote_base_g"))

    if have_stored:
        mismatch = pd.Series(False, index=out.index)
        for f, col in stored_cols.items():
            if col not in out.columns:
                out[f] = reg[f]
                continue
            s = pd.to_numeric(out[col], errors="coerce")
            differs = s.notna() & (s - reg[f]).abs().gt(1e-9)
            mismatch |= differs
            out[f] = s.where(s.notna(), reg[f]).astype(float)
        out["spec_source"] = np.where(
            pd.to_numeric(out["spec_purity"], errors="coerce").notna(),
            "stored", "registry")
        out["spec_mismatch"] = mismatch
    else:
        for f in SPEC_FIELDS:
            out[f] = reg[f]
        out["spec_source"] = "registry"
        out["spec_mismatch"] = False

    fine_per_quote = out["quote_base_g"] * out["purity"]

    out["norm_close"] = out["close"].astype(float) * 10.0 / fine_per_quote
    out["log_norm"] = np.log(out["norm_close"])
    out["lot_notional"] = (out["close"].astype(float)
                           * out["trading_unit_g"] / out["quote_base_g"])

    # Tick size expressed on the common basis. This is the floor on measurable
    # spread precision: on GOLDPETAL a 0.5 tick is ~40 bp of the 1 g quote, so
    # apparent dislocations below that are quantisation, not information.
    out["tick_bp_of_norm"] = (
        out["tick_size_inr"] * 10.0 / fine_per_quote / out["norm_close"] * 10000.0
    )
    return out


def wide_normalized(df: pd.DataFrame, value: str = "norm_close") -> pd.DataFrame:
    """Pivot to date x contract_id. Useful for eyeballing and for charts."""
    return df.pivot_table(index="date", columns="contract_id", values=value)


def front_month(df: pd.DataFrame, min_sessions: int = 3) -> pd.DataFrame:
    """
    The nearest eligible contract per (date, symbol), by sessions to expiry.

    This is for DISPLAY and for choosing which instrument to trade on a given day.
    It is deliberately not stitched into a continuous price series: splicing at the
    roll injects an artificial jump that any mean-reversion signal misreads as an
    opportunity. Every row here keeps its own `contract_id`, so the identity of the
    instrument is never lost.
    """
    elig = df[df["sessions_to_expiry"] >= min_sessions]
    if elig.empty:
        return elig
    idx = elig.groupby(["date", "symbol"])["sessions_to_expiry"].idxmin()
    return df.loc[idx].sort_values(["date", "symbol"]).reset_index(drop=True)


def most_liquid(df: pd.DataFrame, min_sessions: int = 3) -> pd.DataFrame:
    """
    The highest-volume eligible contract per (date, symbol).

    Often but not always the front month. Preferred for execution assumptions,
    because volume is where a fill is even conceivable -- while remembering that
    volume still is not depth.
    """
    elig = df[(df["sessions_to_expiry"] >= min_sessions) & (~df["no_trade"])]
    if elig.empty:
        return elig
    idx = elig.groupby(["date", "symbol"])["volume"].idxmax()
    return df.loc[idx].sort_values(["date", "symbol"]).reset_index(drop=True)
