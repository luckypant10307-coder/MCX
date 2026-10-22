"""
Read bhavcopy files downloaded from MCX by hand.

This is the practical path where the live JSON route is blocked: MCX serves
bhavcopy CSV (and zipped CSV) from the browser with no API key. Drop the files
into a directory and point this at it.

Filenames are used ONLY as a hint for the requested-date cross-check. The
authoritative date is always the Date column inside the file, and the two are
compared so that a file saved under the wrong name -- a real hazard when the site
serves the last available session for a holiday request -- still gets caught by
`ingest.validate_date_echo` rather than silently mislabelling a session.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import date, datetime

import pandas as pd

from .base import RAW_COLUMNS, BhavcopySource

# Header spellings seen across MCX bhavcopy vintages.
_ALIASES = {
    "Date": ["Date", "TRADE DATE", "TradeDate", "DATE", "Trade Date"],
    "Symbol": ["Symbol", "SYMBOL", "Commodity", "COMMODITY", "Product"],
    "ExpiryDate": ["ExpiryDate", "EXPIRY DATE", "Expiry", "EXPIRY",
                   "Expiry Date", "ExpiryDt"],
    "Open": ["Open", "OPEN", "Open Price", "OPEN PRICE"],
    "High": ["High", "HIGH", "High Price", "HIGH PRICE"],
    "Low": ["Low", "LOW", "Low Price", "LOW PRICE"],
    "Close": ["Close", "CLOSE", "Close Price", "CLOSE PRICE",
              "Settlement Price", "SETTLEMENT PRICE", "Settle"],
    "Volume": ["Volume", "VOLUME", "Qty", "QTY", "Traded Qty", "VOLUME (LOTS)"],
    "OpenInterest": ["OpenInterest", "OPEN INTEREST", "OI", "Open Interest"],
}

_DATE_IN_NAME = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_DDMMYYYY_IN_NAME = re.compile(r"(\d{2})[-_]?(\d{2})[-_]?(20\d{2})")


class CSVDirectorySource(BhavcopySource):
    name = "csv_dir"

    def __init__(self, directory: str):
        self.directory = directory

    # ---- discovery ---------------------------------------------------------

    def _files(self):
        if not os.path.isdir(self.directory):
            return []
        out = []
        for fn in sorted(os.listdir(self.directory)):
            p = os.path.join(self.directory, fn)
            if not os.path.isfile(p):
                continue
            if fn.lower().endswith((".csv", ".txt", ".zip")):
                out.append(p)
        return out

    @staticmethod
    def _date_hint(path: str):
        """Best guess at the requested date from the filename. Advisory only."""
        base = os.path.basename(path)
        m = _DATE_IN_NAME.search(base)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        m = _DDMMYYYY_IN_NAME.search(base)
        if m:
            for a, b in ((1, 2), (2, 1)):
                try:
                    return date(int(m.group(3)), int(m.group(b)), int(m.group(a)))
                except ValueError:
                    continue
        return None

    # ---- reading ----------------------------------------------------------

    @staticmethod
    def _read_one(path: str) -> pd.DataFrame:
        frames = []
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for nm in z.namelist():
                    if nm.lower().endswith((".csv", ".txt")):
                        with z.open(nm) as fh:
                            frames.append(_read_csv_bytes(fh.read()))
        else:
            with open(path, "rb") as fh:
                frames.append(_read_csv_bytes(fh.read()))
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
        norm = {str(c).strip().upper(): c for c in df.columns}
        out = {}
        for canon, aliases in _ALIASES.items():
            found = None
            for a in aliases:
                key = a.strip().upper()
                if key in norm:
                    found = norm[key]
                    break
            out[canon] = df[found] if found is not None else None
        return pd.DataFrame(out)

    def load_all(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """
        Read every file in the directory into one raw frame.

        Each row carries `_requested_date` from its filename hint where available,
        so the echo check still has something to compare against. Where no hint
        exists the check is skipped for those rows and the report says so.
        """
        frames = []
        for p in self._files():
            df = self._read_one(p)
            if df.empty:
                continue
            mapped = self._map_columns(df)
            if mapped["Symbol"] is None or mapped["Close"] is None:
                continue
            hint = self._date_hint(p)
            mapped["_requested_date"] = hint.isoformat() if hint else None
            mapped["_source"] = self.name
            mapped["_file"] = os.path.basename(p)
            frames.append(mapped)

        if not frames:
            return pd.DataFrame(columns=RAW_COLUMNS + ["_requested_date", "_source"])

        out = pd.concat(frames, ignore_index=True)
        if start or end:
            parsed = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False)
            if start:
                out = out[parsed.dt.date >= start]
            if end:
                out = out[parsed.dt.date <= end]
        return out.reset_index(drop=True)

    def fetch_day(self, d: date) -> pd.DataFrame:
        """Rows whose in-file Date matches d. Filenames are not trusted for this."""
        allrows = self.load_all()
        if allrows.empty:
            return allrows
        parsed = pd.to_datetime(allrows["Date"], errors="coerce")
        return allrows[parsed.dt.date == d].reset_index(drop=True)


def _read_csv_bytes(b: bytes):
    """Read CSV bytes, tolerating encodings and a few preamble lines."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        for skip in (0, 1, 2, 3, 4):
            try:
                df = pd.read_csv(io.BytesIO(b), encoding=enc, skiprows=skip)
            except Exception:
                continue
            cols = {str(c).strip().upper() for c in df.columns}
            if any(k in cols for k in ("SYMBOL", "COMMODITY", "PRODUCT")):
                return df
    return None
