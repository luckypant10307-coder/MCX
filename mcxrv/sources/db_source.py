"""
Replay the stored raw tape as a BhavcopySource.

This exists so the database can stand in for a live fetch anywhere a source is
accepted, and so a stored range goes through the IDENTICAL parsing and validation
path it went through on the way in. That round trip is what proves the raw tape is
genuinely sufficient to rebuild everything downstream of it -- if replaying raw did
not reproduce the stored quotes, the tape would be missing something.

For ordinary analysis, prefer `repo.read_quotes`: the quotes are already validated,
so replaying raw only repeats work. Use this when a validation rule has CHANGED and
history needs rebuilding under the new rule.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..db import DbConfig, repo
from .base import RAW_COLUMNS, BhavcopySource


class DbSource(BhavcopySource):
    name = "db"

    def __init__(self, cfg: DbConfig | None = None):
        self.cfg = cfg or DbConfig.from_env()

    def fetch_day(self, d: date) -> pd.DataFrame:
        with repo.connect(self.cfg) as conn:
            df = repo.read_raw(conn, start=d, end=d)
        return df if not df.empty else pd.DataFrame(columns=RAW_COLUMNS)

    def fetch_range(self, start: date, end: date, symbols=None) -> pd.DataFrame:
        """One query for the whole range rather than a day at a time."""
        with repo.connect(self.cfg) as conn:
            return repo.read_raw(conn, start=start, end=end, symbols=symbols)
