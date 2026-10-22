"""
Live MCX bhavcopy client.

Request format is DD/MM/YYYY. Response Date comes back MM/DD/YYYY. The two
disagree, and the asymmetry is load-bearing -- see `ingest.validate_date_echo`.

NOTE ON ACCESS: www.mcxindia.com sits behind Akamai bot protection that rejects
non-browser TLS clients regardless of headers. This client is correct and will
work from a context MCX serves (a browser-driven fetch, an allow-listed address,
or an upstream mirror). Where it is blocked, use CSVDirectorySource with files
downloaded from the site by hand -- that is the supported path and needs no key.
This module deliberately does not attempt to defeat the bot protection.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import requests

from .base import RAW_COLUMNS, BhavcopySource

BHAVCOPY_PAGE = "https://www.mcxindia.com/market-data/bhavcopy"
ENDPOINT = "https://www.mcxindia.com/backpage.aspx/GetDateWiseBhavCopy"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# MCX field names -> canonical raw names. The live payload has drifted between
# releases, so map defensively by alias rather than positionally.
_FIELD_ALIASES = {
    "Date": ["Date", "TradeDate", "BhavDate"],
    "Symbol": ["Symbol", "SYMBOL", "Commodity"],
    "ExpiryDate": ["ExpiryDate", "Expiry", "ExpiryDt"],
    "Open": ["Open", "OpenPrice"],
    "High": ["High", "HighPrice"],
    "Low": ["Low", "LowPrice"],
    "Close": ["Close", "ClosePrice", "SettlementPrice", "Settlement"],
    "Volume": ["Volume", "Vol", "TradedQty"],
    "OpenInterest": ["OpenInterest", "OI", "OpenInt"],
}


class MCXLiveSource(BhavcopySource):
    name = "mcx_live"

    def __init__(self, timeout: int = 40, session: requests.Session | None = None):
        self.timeout = timeout
        self.session = session or self._make_session()
        self._warmed = False

    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": _UA,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": BHAVCOPY_PAGE,
            "Origin": "https://www.mcxindia.com",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/json; charset=UTF-8",
        })
        return s

    def _warm(self) -> None:
        """Pick up whatever cookies the page sets before hitting the JSON route."""
        if self._warmed:
            return
        try:
            self.session.get(BHAVCOPY_PAGE, timeout=self.timeout)
        except requests.RequestException:
            pass
        self._warmed = True

    def fetch_day(self, d: date) -> pd.DataFrame:
        self._warm()
        payload = {"Date": d.strftime("%d/%m/%Y")}   # REQUEST is DD/MM/YYYY
        r = self.session.post(ENDPOINT, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code == 403:
            raise PermissionError(
                "MCX returned 403 (Akamai bot protection). Use CSVDirectorySource "
                "with manually downloaded bhavcopy files, or run from a context "
                "MCX serves."
            )
        r.raise_for_status()
        rows = self._unwrap(r.text)
        if not rows:
            return pd.DataFrame(columns=RAW_COLUMNS)
        return self._to_raw_frame(rows)

    @staticmethod
    def _unwrap(text: str) -> list:
        """ASP.NET wraps the array in {"d": ...}, sometimes as a JSON string."""
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("d", obj.get("Data", obj))
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            obj = obj.get("Data", obj.get("Table", []))
        return obj if isinstance(obj, list) else []

    @staticmethod
    def _to_raw_frame(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        out = {}
        for canon, aliases in _FIELD_ALIASES.items():
            for a in aliases:
                if a in df.columns:
                    out[canon] = df[a]
                    break
            else:
                out[canon] = None
        res = pd.DataFrame(out)
        # Keep the raw payload alongside, so an unmapped field stays recoverable.
        res["_raw_fields"] = [json.dumps(r, default=str) for r in rows]
        return res
