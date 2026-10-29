"""
Bhavcopy source abstraction.

Every source returns RAW bhavcopy rows -- MCX's own column names, MCX's own
string formats, space padding and all. Parsing, validation and normalisation
happen downstream in `ingest`. Keeping sources dumb means the synthetic source
exercises exactly the same parsing path as the live one, so a test against
synthetic data is a real test of production code.
"""
from __future__ import annotations

import abc
from datetime import date

import pandas as pd

# The columns the problem statement guarantees. Sources must emit at least these.
RAW_COLUMNS = [
    "Date", "Symbol", "ExpiryDate",
    "Open", "High", "Low", "Close",
    "Volume", "OpenInterest",
]


class BhavcopySource(abc.ABC):
    """A thing that can hand back one trading day of raw bhavcopy rows."""

    name: str = "abstract"

    @abc.abstractmethod
    def fetch_day(self, d: date) -> pd.DataFrame:
        """
        Return raw rows for a single requested date.

        Implementations MUST NOT correct, reindex or drop the Date column the
        upstream returned, even when it disagrees with `d`. That disagreement is
        the signal `ingest.validate_date_echo` exists to catch: MCX answers an
        unrecognised, holiday, malformed or future date with the most recent
        available trading day, and silently accepting it duplicates one day's
        prices across many dates.
        """

    def fetch_range(self, start: date, end: date, symbols=None) -> pd.DataFrame:
        """Concatenate day-by-day. Subclasses may override with a bulk path."""
        frames = []
        for d in pd.date_range(start, end, freq="D"):
            dd = d.date()
            try:
                df = self.fetch_day(dd)
            except Exception as exc:  # a bad day must not kill the run
                frames.append(_error_row(dd, exc))
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["_requested_date"] = dd.isoformat()
            df["_source"] = self.name
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS + ["_requested_date", "_source"])
        out = pd.concat(frames, ignore_index=True)
        if symbols is not None:
            want = {s.strip().upper() for s in symbols}
            out = out[out["Symbol"].astype(str).str.strip().str.upper().isin(want)]
        return out.reset_index(drop=True)


def _error_row(d: date, exc: Exception) -> pd.DataFrame:
    """A visible tombstone for a failed fetch, so gaps are auditable not invisible."""
    return pd.DataFrame([{
        **{c: None for c in RAW_COLUMNS},
        "_requested_date": d.isoformat(),
        "_source": "error",
        "_error": f"{type(exc).__name__}: {exc}",
    }])
