"""
Parse raw bhavcopy into a tidy, validated frame, and flag every data pathology
the problem statement warns about rather than silently absorbing it.

The governing principle: a data problem must become a COLUMN, not a correction.
Dropping a bad row hides it; repairing it invents data. Flagging it lets the
analysis decide, and lets the final report state how much data it stood on.

Pathologies handled here:

  date echo          MCX answers an unrecognised, holiday, malformed or future
                     date with the most recent available trading day. Silently
                     accepting it duplicates one session across many dates,
                     manufacturing zero-return days, deflating the volatility
                     estimate and inflating every Sharpe downstream.
  date format        Requests go out DD/MM/YYYY; responses come back MM/DD/YYYY.
                     Below day 13 both parse, so the corruption is partial and
                     silent. We parse strictly, then cross-check the echo.
  space padding      Symbol arrives as 'GOLDM     '. An untrimmed join drops legs.
  compact expiry     '04SEP2026'. Misparsing reassigns a row to another instrument.
  duplicates         Repeated (date, contract_id) rows.
  derived settlement On a no-trade day the exchange publishes a computed
                     settlement. It is smooth, mean-reverting and untradeable.
                     Backtesting fills against it is backtesting the exchange's
                     own model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import calendar as cal
from . import specs

TIDY_COLUMNS = [
    "date", "symbol", "expiry_date", "contract_id",
    "open", "high", "low", "close", "volume", "open_interest",
]


@dataclass
class DataQualityReport:
    """Auditable tally of what was wrong with the input."""
    rows_in: int = 0
    rows_out: int = 0
    counts: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)

    def note(self, key: str, value) -> None:
        self.details[key] = value

    def to_frame(self) -> pd.DataFrame:
        if not self.counts:
            return pd.DataFrame(columns=["issue", "rows"])
        return (pd.DataFrame(sorted(self.counts.items()), columns=["issue", "rows"])
                .sort_values("rows", ascending=False).reset_index(drop=True))

    def summary(self) -> str:
        lines = [
            "Data quality report",
            "  rows in : {}".format(self.rows_in),
            "  rows out: {}".format(self.rows_out),
        ]
        if self.counts:
            lines.append("  flags:")
            for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1]):
                lines.append("    {:<28s} {}".format(k, v))
        for k, v in self.details.items():
            lines.append("  {}: {}".format(k, v))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# date parsing
# ---------------------------------------------------------------------------

def parse_response_date(token, requested: date | None = None):
    """
    Parse a bhavcopy Date field, which MCX renders MM/DD/YYYY.

    Returns (parsed_date, flag) where flag is one of:
      'ok'              strict MM/DD/YYYY parse succeeded
      'swapped'         only DD/MM/YYYY parsed, or swapping matches the requested
                        date while MM/DD does not -- a format problem, not an echo
      'ambiguous_ok'    both orderings are valid and agree with the request
      'unparseable'     neither ordering yields a real date

    The `requested` cross-check is what separates a format bug from a date echo.
    Without it, the two are indistinguishable and get conflated.
    """
    s = str(token).strip()
    if not s or s.lower() in ("nan", "none"):
        return None, "unparseable"

    mdy = _try(s, "%m/%d/%Y")
    dmy = _try(s, "%d/%m/%Y")

    if mdy is None and dmy is None:
        # Text-month formats are unambiguous. "%d %b %Y" is the current MCX
        # bhavcopy download ("01 Oct 2026"); the others are older vintages.
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d%b%Y", "%d-%b-%Y", "%d %b %Y",
                    "%d %B %Y"):
            alt = _try(s, fmt)
            if alt is not None:
                return alt, "ok"
        return None, "unparseable"

    if mdy is not None and dmy is None:
        return mdy, "ok"
    if mdy is None and dmy is not None:
        # First field exceeded 12, so the stated MM/DD contract is violated.
        return dmy, "swapped"

    # Both parse (day <= 12). Prefer the documented MM/DD/YYYY, but if the request
    # is known and only the swap matches it, the field ordering is wrong.
    if requested is not None and mdy != requested and dmy == requested:
        return dmy, "swapped"
    if requested is not None and mdy == requested:
        return mdy, "ambiguous_ok"
    return mdy, "ok"


def _try(s: str, fmt: str):
    try:
        return datetime.strptime(s, fmt).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def tidy(raw: pd.DataFrame, symbols=None, strict: bool = False):
    """
    Turn raw bhavcopy rows into a validated tidy frame plus a quality report.

    Returns (df, report). `df` carries the TIDY_COLUMNS plus diagnostic flags:
      date_echo          the echoed Date disagreed with the date we requested
      date_flag          parse provenance, per `parse_response_date`
      expiry_window_bad  expiry fell outside the documented day-of-month window
      ohlc_bad           OHLC internally inconsistent
      no_trade           zero volume: the close is a derived settlement
      settle_untradeable no_trade, or OHLC collapsed onto the close

    Rows are dropped only when they cannot be identified at all -- unparseable
    date, unparseable expiry, unknown symbol, non-positive close. Everything else
    is retained and flagged, so downstream filters are explicit and visible.
    """
    rep = DataQualityReport(rows_in=len(raw))
    if raw.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS), rep

    df = raw.copy()

    # tombstoned fetch failures
    if "_source" in df.columns:
        errs = df["_source"].astype(str).eq("error")
        if errs.any():
            rep.bump("fetch_error_rows", int(errs.sum()))
            df = df[~errs].copy()

    # ---- symbol: trim the space padding before anything joins on it ----
    raw_sym = df["Symbol"].astype(str)
    df["symbol"] = raw_sym.str.strip().str.upper()
    padded = raw_sym.ne(df["symbol"])
    if padded.any():
        rep.bump("symbol_whitespace", int(padded.sum()))

    known = df["symbol"].isin(specs.SPECS)
    if symbols is not None:
        want = {s.strip().upper() for s in symbols}
        known &= df["symbol"].isin(want)
    rep.bump("symbol_unknown_or_filtered", int((~known).sum()))
    df = df[known].copy()
    if df.empty:
        rep.rows_out = 0
        return pd.DataFrame(columns=TIDY_COLUMNS), rep

    # ---- instrument: futures only ----
    # Current MCX downloads mix options on the same symbol (OPTFUT, with a strike
    # and CE/PE) into the file. Their prices are premiums, not futures prices, and
    # would collide with the futures rows on (date, contract) if let through.
    if "Instrument" in df.columns and df["Instrument"].notna().any():
        inst = df["Instrument"].astype(str).str.strip().str.upper()
        fut = inst.isin(("FUTCOM", "FUTIDX", "", "NONE", "NAN"))
        if (~fut).any():
            rep.bump("non_futures_rows_dropped", int((~fut).sum()))
            df = df[fut].copy()
        if df.empty:
            rep.rows_out = 0
            return pd.DataFrame(columns=TIDY_COLUMNS), rep

    # ---- date: parse, then cross-check the echo against what we asked for ----
    req = None
    if "_requested_date" in df.columns:
        req = pd.to_datetime(df["_requested_date"], errors="coerce").dt.date

    parsed, flags = [], []
    for i, tok in enumerate(df["Date"].to_numpy()):
        r = None if req is None else req.iloc[i]
        p, f = parse_response_date(tok, r)
        parsed.append(p)
        flags.append(f)
    df["date"] = parsed
    df["date_flag"] = flags

    bad_date = df["date"].isna()
    if bad_date.any():
        rep.bump("date_unparseable_dropped", int(bad_date.sum()))
        df = df[~bad_date].copy()

    for f in ("swapped", "ambiguous_ok"):
        n = int((df["date_flag"] == f).sum())
        if n:
            rep.bump("date_flag_" + f, n)

    if req is not None:
        req = req.loc[df.index]
        df["date_echo"] = df["date"].ne(req)
        n_echo = int(df["date_echo"].sum())
        if n_echo:
            rep.bump("date_echo_rows", n_echo)
            ex = df.loc[df["date_echo"], ["_requested_date", "Date"]].head(5)
            rep.note("date_echo_examples",
                     list(ex.itertuples(index=False, name=None)))
            # The echoed session is real data; it is the REQUESTED label that is
            # wrong. Keep the row under its true date and let dedup collapse the
            # repeats, rather than importing the same session many times over.
    else:
        df["date_echo"] = False

    # ---- expiry ----
    exp, exp_bad = [], []
    for tok in df["ExpiryDate"].to_numpy():
        try:
            exp.append(cal.parse_expiry(tok))
            exp_bad.append(False)
        except (ValueError, TypeError):
            exp.append(None)
            exp_bad.append(True)
    df["expiry_date"] = exp
    if any(exp_bad):
        rep.bump("expiry_unparseable_dropped", int(np.sum(exp_bad)))
    df = df[~pd.Series(exp_bad, index=df.index)].copy()
    if df.empty:
        rep.rows_out = 0
        return pd.DataFrame(columns=TIDY_COLUMNS), rep

    df["expiry_window_bad"] = [
        not cal.expiry_window_ok(s, e)
        for s, e in zip(df["symbol"], df["expiry_date"])
    ]
    n = int(df["expiry_window_bad"].sum())
    if n:
        rep.bump("expiry_outside_window", n)

    # expiry must not precede the session
    stale_exp = pd.Series(
        [e < d for d, e in zip(df["date"], df["expiry_date"])], index=df.index)
    if stale_exp.any():
        rep.bump("expiry_before_date_dropped", int(stale_exp.sum()))
        df = df[~stale_exp].copy()

    df["contract_id"] = [
        cal.contract_id(s, e) for s, e in zip(df["symbol"], df["expiry_date"])
    ]

    # ---- prices ----
    for src, dst in (("Open", "open"), ("High", "high"), ("Low", "low"),
                     ("Close", "close")):
        df[dst] = pd.to_numeric(df[src], errors="coerce")
    for src, dst in (("Volume", "volume"), ("OpenInterest", "open_interest")):
        df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0)

    bad_close = df["close"].isna() | (df["close"] <= 0)
    if bad_close.any():
        rep.bump("close_missing_or_nonpositive_dropped", int(bad_close.sum()))
        df = df[~bad_close].copy()

    df["ohlc_bad"] = (
        (df["low"] > df["high"])
        | (df["close"] > df["high"] + 1e-9)
        | (df["close"] < df["low"] - 1e-9)
        | (df["open"] > df["high"] + 1e-9)
        | (df["open"] < df["low"] - 1e-9)
    ).fillna(False)
    n = int(df["ohlc_bad"].sum())
    if n:
        rep.bump("ohlc_inconsistent", n)

    # ---- tradability of the published settlement ----
    df["no_trade"] = df["volume"] <= 0
    collapsed = (
        (df["high"] - df["low"]).abs().le(1e-9)
        & (df["open"] - df["close"]).abs().le(1e-9)
    ).fillna(False)
    df["settle_untradeable"] = df["no_trade"] | collapsed
    rep.bump("no_trade_rows", int(df["no_trade"].sum()))
    rep.bump("settle_untradeable_rows", int(df["settle_untradeable"].sum()))

    # ---- duplicates ----
    before = len(df)
    key = ["date", "contract_id"]
    dupe_mask = df.duplicated(subset=key, keep=False)
    n_dupe_rows = int(dupe_mask.sum())
    if n_dupe_rows:
        rep.bump("duplicate_rows_seen", n_dupe_rows)
        conflict = (df[dupe_mask].groupby(key)["close"].nunique() > 1)
        n_conflict = int(conflict.sum())
        if n_conflict:
            rep.bump("duplicate_key_price_conflict", n_conflict)
            rep.note("duplicate_conflict_note",
                     "same (date, contract) carried differing closes; kept the "
                     "highest-volume row")
    # Prefer the most-traded observation when a key repeats.
    df = (df.sort_values(key + ["volume"], ascending=[True, True, False])
            .drop_duplicates(subset=key, keep="first"))
    rep.bump("duplicate_rows_collapsed", before - len(df))

    out_cols = TIDY_COLUMNS + [
        "date_flag", "date_echo", "expiry_window_bad", "ohlc_bad",
        "no_trade", "settle_untradeable",
    ]
    out = (df[out_cols]
           .sort_values(["date", "symbol", "expiry_date"])
           .reset_index(drop=True))

    rep.rows_out = len(out)
    rep.note("date_range", "{} .. {}".format(out["date"].min(), out["date"].max()))
    rep.note("symbols", sorted(out["symbol"].unique()))
    rep.note("contracts", int(out["contract_id"].nunique()))
    rep.note("sessions", int(out["date"].nunique()))

    if strict and rep.counts.get("date_echo_rows"):
        raise ValueError(
            "date echo detected on {} rows; refusing to proceed under strict=True"
            .format(rep.counts["date_echo_rows"]))

    return out, rep


def build_dataset(raw: pd.DataFrame, symbols=None):
    """
    Full pipeline: tidy, derive the trading calendar from the data, attach
    lifecycle columns.

    The calendar is inferred from observed sessions rather than hardcoded, so it
    cannot desynchronise from the exchange's real holiday schedule.
    """
    df, rep = tidy(raw, symbols=symbols)
    if df.empty:
        return df, rep, cal.TradingCalendar(days=())
    tc = cal.TradingCalendar.from_dates(df["date"].unique())
    df = cal.lifecycle_frame(df, tc)
    rep.note("trading_sessions_inferred", len(tc.days))
    return df, rep, tc
