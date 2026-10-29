"""
Contract lifecycle: expiry parsing, tender windows, days-to-expiry, tradability.

Every intended entry and exit must sit inside the relevant contract calendar.
This module is what makes that checkable rather than assumed.

Design note: the trading-day calendar is DERIVED from observed data, never
hardcoded. MCX holidays shift year to year, and a hardcoded list silently
desynchronises. If a date is absent from every symbol's data, it was not a
trading day.
"""
from __future__ import annotations

import calendar as _pycal
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import specs

# MCX renders ExpiryDate in a compact, case-inconsistent form: 04SEP2026, 5Sep2026,
# 30SEP2026. Accept all of them; reject anything else loudly.
_EXPIRY_RE = re.compile(r"^\s*(\d{1,2})([A-Za-z]{3})(\d{4})\s*$")
_MONTHS = {m.upper(): i for i, m in enumerate(_pycal.month_abbr) if m}


def parse_expiry(token: str) -> date:
    """
    Parse MCX's compact expiry format, e.g. '04SEP2026' -> date(2026, 9, 4).

    Raises ValueError on anything unparseable. Never silently coerces, because a
    misparsed expiry reassigns a row to the wrong instrument, which is worse than
    dropping it.
    """
    if isinstance(token, (date, datetime)):
        return token.date() if isinstance(token, datetime) else token
    m = _EXPIRY_RE.match(str(token))
    if not m:
        raise ValueError(f"unparseable ExpiryDate {token!r}; expected e.g. '04SEP2026'")
    dd, mon, yyyy = m.group(1), m.group(2).upper(), m.group(3)
    if mon not in _MONTHS:
        raise ValueError(f"unknown month abbreviation {mon!r} in ExpiryDate {token!r}")
    return date(int(yyyy), _MONTHS[mon], int(dd))


def format_expiry(d: date) -> str:
    """Render a date back into MCX's compact form (for round-trip tests)."""
    return f"{d.day:02d}{_pycal.month_abbr[d.month].upper()}{d.year}"


def contract_id(symbol: str, expiry: date) -> str:
    """
    Stable per-instrument key. Tracking by this -- never by 'near month' -- is what
    prevents the artificial price jump you get from splicing contracts at a roll.
    """
    return f"{symbol.strip().upper()}|{expiry.isoformat()}"


def expiry_window_ok(symbol: str, expiry: date) -> bool:
    """
    Check a parsed expiry falls inside the symbol's documented day-of-month window.

    A GOLDM expiry on the 29th, or a GOLDPETAL expiry on the 4th, means either the
    spec changed or the row is corrupt. Either way it must surface, not slide by.
    """
    sp = specs.get(symbol)
    last_dom = _pycal.monthrange(expiry.year, expiry.month)[1]
    hi = min(sp.expiry_day_hi, last_dom)
    lo = min(sp.expiry_day_lo, hi)
    if lo <= expiry.day <= hi:
        return True
    # An expiry that would fall on a weekend moves to the nearest business day,
    # which can sit just outside the nominal window: 26 Feb 2027 is the last
    # business day of a February whose 27th and 28th are a weekend. Accept it only
    # when EVERY day of the window is a weekend, so a genuinely misplaced expiry
    # still surfaces. (Exchange holidays are not modelled here.)
    window = [date(expiry.year, expiry.month, d) for d in range(lo, hi + 1)]
    if all(d.weekday() >= 5 for d in window):
        gap = (window[0] - expiry).days if expiry.day < lo else (expiry - window[-1]).days
        return 0 < gap <= 3 and expiry.weekday() < 5
    return False


@dataclass(frozen=True)
class TradingCalendar:
    """Trading days inferred from the union of dates observed in the data."""
    days: tuple[date, ...]

    @classmethod
    def from_dates(cls, dates) -> "TradingCalendar":
        uniq = sorted({_as_date(d) for d in dates})
        return cls(days=tuple(uniq))

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([pd.Timestamp(d) for d in self.days])

    def is_trading_day(self, d) -> bool:
        return _as_date(d) in set(self.days)

    def sessions_between(self, a, b) -> int:
        """Count of observed trading sessions in (a, b]. Negative if b < a."""
        a, b = _as_date(a), _as_date(b)
        sign = 1 if b >= a else -1
        lo, hi = (a, b) if b >= a else (b, a)
        n = sum(1 for d in self.days if lo < d <= hi)
        return sign * n

    @property
    def density(self) -> float:
        """Observed trading sessions per calendar day, for extrapolation."""
        if len(self.days) < 30:
            return 252.0 / 365.0
        span = (self.days[-1] - self.days[0]).days
        return len(self.days) / span if span > 0 else 252.0 / 365.0

    def sessions_remaining(self, d, expiry):
        """
        Sessions from `d` to `expiry`, extrapolating past the end of the data.

        Returns (count, estimated). Necessary because the calendar is inferred from
        observed sessions, so at the right-hand edge of the dataset every live
        contract would otherwise report zero sessions to expiry -- "we have no more
        data" silently becomes "this contract is expiring", which zeroes tau,
        collapses the maturity spread the curve fit needs, and marks the most recent
        session untradeable. That is precisely the session a trader is looking at.

        Beyond the last observed day the remaining sessions are estimated from the
        observed session density, and the flag says so, so a consumer can distinguish
        a counted number from an inferred one.
        """
        d, expiry = _as_date(d), _as_date(expiry)
        if expiry <= d:
            return 0, False
        last = self.days[-1] if self.days else d
        if expiry <= last:
            return self.sessions_between(d, expiry), False
        counted = self.sessions_between(d, last)
        extra_cal = (expiry - last).days
        return counted + int(round(extra_cal * self.density)), True

    def shift(self, d, n: int) -> date | None:
        """The trading day n sessions after d (n may be negative)."""
        d = _as_date(d)
        arr = list(self.days)
        # position of the last trading day <= d
        pos = np.searchsorted(np.array(arr, dtype=object), d, side="right") - 1
        tgt = pos + n
        if tgt < 0 or tgt >= len(arr):
            return None
        return arr[tgt]


def _as_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return pd.Timestamp(d).date()


def lifecycle_frame(df: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
    """
    Attach lifecycle columns to a tidy bhavcopy frame.

    Adds:
      days_to_expiry      calendar days, expiry - date
      sessions_to_expiry  observed trading sessions remaining (the honest clock)
      T_years             sessions_to_expiry / 252, the carry model's time variable
      in_tender           within the tender window; execution assumptions break here
      is_expiry_day       final session
      contract_age_days   days since this instrument first appeared in the data
    """
    out = df.copy()
    d = pd.to_datetime(out["date"]).dt.date
    e = pd.to_datetime(out["expiry_date"]).dt.date

    out["days_to_expiry"] = [(ed - dd).days for dd, ed in zip(d, e)]
    rem = [cal.sessions_remaining(dd, ed) for dd, ed in zip(d, e)]
    out["sessions_to_expiry"] = [max(n, 0) for n, _est in rem]
    # True where the count was extrapolated past the end of the observed calendar,
    # which happens for every live contract on the most recent sessions.
    out["sessions_estimated"] = [bool(est) for _n, est in rem]
    out["T_years"] = out["sessions_to_expiry"] / 252.0

    tender = out["symbol"].map(lambda s: specs.get(s).tender_days)
    out["in_tender"] = out["sessions_to_expiry"] <= tender
    out["is_expiry_day"] = out["days_to_expiry"] <= 0

    first_seen = out.groupby("contract_id")["date"].transform("min")
    out["contract_age_days"] = (
        pd.to_datetime(out["date"]) - pd.to_datetime(first_seen)
    ).dt.days

    return out


def tradable_mask(df: pd.DataFrame, min_sessions: int = 3) -> pd.Series:
    """
    Sessions on which we are willing to assume an entry or exit is placeable.

    Excludes the tender window and the last `min_sessions` sessions: liquidity
    collapses and settlement stops resembling an executable price. A backtest that
    ignores this books fills that could not have happened.
    """
    return (
        (~df["in_tender"])
        & (df["sessions_to_expiry"] >= min_sessions)
        & (~df["is_expiry_day"])
    )
