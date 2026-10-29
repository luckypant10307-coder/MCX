"""
Ground-truth bhavcopy generator.

This is not a stand-in for real data; it is a measuring instrument. Because the
true decomposition is known by construction, it establishes three things that
real data cannot establish on its own:

  1. The validators actually catch the traps. We inject date echoes, MM/DD vs
     DD/MM confusion, space padding, duplicates and derived settlements, then
     assert that `ingest` flags them.
  2. The signal engine recovers a dislocation that is really there. Inject a
     known relative-value residual of known half-life; check the z-score finds it.
  3. The backtest reports NO edge when there is none. Set residual_bp=0 and the
     world becomes a pure level-plus-carry process with zero relative value. Any
     strategy that still shows profit is reading its own look-ahead or a costing
     bug. That null test is the single most valuable check in the repo.

Generating process, in log space, for contract i of symbol s at time t:

    ln P_norm[i,t] = L_t  +  b_t * tau[i,t]  +  c_s  +  e[s,t]  +  eps[i,t]

    L_t        log spot, INR per 10 g FINE gold. Random walk.
    b_t        annualised carry (contango) rate. Slow OU around a positive mean,
               so tau carries the term structure and roll-down is mechanical.
    tau[i,t]   time to expiry in years. Drives roll-down: as tau goes to 0 the
               carry premium decays deterministically. A strategy long a far leg
               and short a near leg harvests that without it being an edge.
    c_s        structural log premium of symbol s over a bar contract. Represents
               fabrication cost embedded in a minted deliverable: a 1 g petal
               costs far more per gram to make than a 100 g bar. PERSISTENT and
               economically justified, not a mispricing.
    e[s,t]     the only genuine relative-value term. Mean-reverting OU, per
               symbol family, zero mean. This is what a signal should find.
    eps[i,t]   per-contract idiosyncratic noise.

Quotes are then de-normalised through each contract's own quote base and purity
and rounded to its tick size, so purity offsets and tick quantisation appear in
the output exactly as they do in real bhavcopy.
"""
from __future__ import annotations

import calendar as _pycal
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..stage1_intake import specs
from .base import RAW_COLUMNS, BhavcopySource

# Structural log premium over a bar contract, in fine-gold terms, in basis
# points. Magnitudes are assumptions chosen to be plausible for making charges
# on minted product; they are parameters, not claims about the real market.
DEFAULT_STRUCTURAL_PREMIUM_BP = {
    "GOLDM": 0.0,        # reference: 995 bar
    "GOLDTEN": 8.0,      # 10 g bar, slight fabrication
    "GOLDGUINEA": 35.0,  # 8 g minted coin
    "GOLDPETAL": 110.0,  # 1 g minted coin, highest making charge per gram
}

# Relative liquidity tiers: expected front-month lots traded per day.
DEFAULT_LIQUIDITY = {
    "GOLDM": 9000.0,
    "GOLDTEN": 600.0,
    "GOLDGUINEA": 180.0,
    "GOLDPETAL": 90.0,
}


@dataclass
class SyntheticConfig:
    start: date = date(2021, 1, 1)
    end: date = date(2026, 9, 18)
    seed: int = 7

    spot_start: float = 48000.0      # INR per 10 g fine at t0
    spot_vol: float = 0.14           # annualised log vol of gold
    spot_drift: float = 0.06         # annualised log drift

    carry_mean: float = 0.060        # annualised contango
    carry_vol: float = 0.020
    carry_kappa: float = 3.0         # slow: curve shape persists for months

    # The relative-value term. residual_bp = 0.0 makes the world edge-free.
    residual_bp: float = 22.0        # stationary sd of e[s,t], basis points
    residual_halflife_days: float = 12.0
    idio_bp: float = 4.0             # per-contract noise

    structural_premium_bp: dict = field(
        default_factory=lambda: dict(DEFAULT_STRUCTURAL_PREMIUM_BP))
    liquidity: dict = field(default_factory=lambda: dict(DEFAULT_LIQUIDITY))

    months_listed_ahead: int = 6     # how early a contract starts trading
    holidays_per_year: int = 14

    # --- pathology injection: emit the data problems the brief warns about ---
    inject_date_echo: bool = True    # holiday request returns last session rows
    inject_space_padding: bool = True
    inject_duplicate_rows: float = 0.002
    derived_settlement_threshold: int = 1   # volume below this: no genuine trade


class SyntheticSource(BhavcopySource):
    """Deterministic given a seed. Retains ground truth for verification."""

    name = "synthetic"

    def __init__(self, config: SyntheticConfig | None = None):
        self.cfg = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build()

    # ---------------- world construction -----------------------------------

    def _build(self) -> None:
        cfg = self.cfg
        self.holidays = self._make_holidays()
        self.sessions = self._make_sessions()
        n = len(self.sessions)
        dt = 1.0 / 252.0

        # log spot level: random walk
        shocks = self.rng.normal(0, cfg.spot_vol * np.sqrt(dt), n)
        drift = (cfg.spot_drift - 0.5 * cfg.spot_vol ** 2) * dt
        self.L = np.log(cfg.spot_start) + np.cumsum(drift + shocks)

        # carry rate: slow OU around carry_mean
        self.b = self._ou(n, dt, cfg.carry_mean, cfg.carry_vol, cfg.carry_kappa,
                          x0=cfg.carry_mean)

        # relative-value residual: one OU path per symbol
        kappa_e = np.log(2.0) / (cfg.residual_halflife_days / 252.0)
        sd_e = cfg.residual_bp / 10000.0
        self.e = {}
        for sym in specs.ALL_SYMBOLS:
            if sd_e <= 0 or sym == "GOLDM":
                # GOLDM is the numeraire: its residual is pinned to zero so the
                # other families are identified relative to it.
                self.e[sym] = np.zeros(n)
            else:
                sigma = sd_e * np.sqrt(2.0 * kappa_e)
                self.e[sym] = self._ou(n, dt, 0.0, sigma, kappa_e, x0=0.0)

        self.session_pos = {d: i for i, d in enumerate(self.sessions)}
        self.contracts = self._make_contracts()
        self._truth_rows = []

    def _ou(self, n, dt, mean, sigma, kappa, x0):
        """Exact-discretisation Ornstein-Uhlenbeck path."""
        a = np.exp(-kappa * dt)
        sd = sigma * np.sqrt((1.0 - a ** 2) / (2.0 * kappa)) if kappa > 0 else sigma
        x = np.empty(n)
        x[0] = x0
        z = self.rng.normal(0, 1, n)
        for t in range(1, n):
            x[t] = mean + a * (x[t - 1] - mean) + sd * z[t]
        return x

    def _make_holidays(self) -> set:
        """Plausible weekday holidays, deterministic under the seed."""
        hol = set()
        for yr in range(self.cfg.start.year, self.cfg.end.year + 1):
            jan1 = date(yr, 1, 1)
            weekdays = []
            for k in range(366):
                d = jan1 + timedelta(days=k)
                if d.year != yr:
                    break
                if d.weekday() < 5:
                    weekdays.append(d)
            size = min(self.cfg.holidays_per_year, len(weekdays))
            pick = self.rng.choice(len(weekdays), size=size, replace=False)
            hol.update(weekdays[int(i)] for i in pick)
        return hol

    def _make_sessions(self) -> list:
        out = []
        d = self.cfg.start
        while d <= self.cfg.end:
            if d.weekday() < 5 and d not in self.holidays:
                out.append(d)
            d += timedelta(days=1)
        return out

    def _prev_session_on_or_before(self, d: date):
        limit = self.cfg.start - timedelta(days=400)
        while d >= limit:
            if d.weekday() < 5 and d not in self.holidays:
                return d
            d -= timedelta(days=1)
        return None

    def _expiry_for(self, symbol: str, year: int, month: int):
        """
        GOLDM expires on the 5th, or the prior trading day. The 999 trio expire on
        the last day of the month, or the prior trading day. Both land inside the
        windows the contract specs document.
        """
        sp = specs.get(symbol)
        last_dom = _pycal.monthrange(year, month)[1]
        if sp.expiry_day_hi <= 5:
            target = date(year, month, 5)
        else:
            target = date(year, month, last_dom)
        e = self._prev_session_on_or_before(target)
        if e is None or e.month != month or e.year != year:
            return None
        hi = min(sp.expiry_day_hi, last_dom)
        return e if sp.expiry_day_lo <= e.day <= hi else None

    def _make_contracts(self) -> list:
        """One record per (symbol, expiry), with its listing date."""
        cfg = self.cfg
        out = []
        horizon = date(cfg.end.year, cfg.end.month, 1) + timedelta(days=400)
        for sym in specs.ALL_SYMBOLS:
            sp = specs.get(sym)
            y, m = cfg.start.year, cfg.start.month
            while date(y, m, 1) <= horizon:
                exp = self._expiry_for(sym, y, m)
                if exp is not None and exp >= cfg.start:
                    listed = exp - timedelta(days=30 * cfg.months_listed_ahead)
                    if sp.listed_from is not None:
                        listed = max(listed, sp.listed_from)
                    out.append({"symbol": sym, "expiry": exp, "listed": listed})
                m += 1
                if m > 12:
                    m, y = 1, y + 1
        return out

    # ---------------- per-day generation -----------------------------------

    def _live_contracts(self, d: date) -> list:
        return [c for c in self.contracts if c["listed"] <= d <= c["expiry"]]

    def fetch_day(self, d: date) -> pd.DataFrame:
        """
        Raw rows for a requested date, in MCX's own shape and formats.

        If `d` is not a trading day and inject_date_echo is on, this returns the
        most recent trading day rows carrying THAT session Date field, exactly as
        MCX does. The caller cannot tell without comparing the echoed Date to the
        date it asked for. That comparison is `ingest.validate_date_echo`.
        """
        cfg = self.cfg
        served = d
        if d not in self.session_pos:
            if not cfg.inject_date_echo:
                return pd.DataFrame(columns=RAW_COLUMNS)
            prev = self._prev_session_on_or_before(d)
            if prev is None or prev not in self.session_pos:
                return pd.DataFrame(columns=RAW_COLUMNS)
            served = prev

        t = self.session_pos[served]
        rows, truth = [], []

        by_sym = {}
        for c in self._live_contracts(served):
            by_sym.setdefault(c["symbol"], []).append(c)
        for lst in by_sym.values():
            lst.sort(key=lambda c: c["expiry"])

        for sym, lst in by_sym.items():
            sp = specs.get(sym)
            c_s = cfg.structural_premium_bp.get(sym, 0.0) / 10000.0
            for rank, c in enumerate(lst):
                exp = c["expiry"]
                tau = max((exp - served).days, 0) / 365.0
                eps = float(self.rng.normal(0, cfg.idio_bp / 10000.0))
                e_st = float(self.e[sym][t])

                log_norm = self.L[t] + self.b[t] * tau + c_s + e_st + eps
                norm = float(np.exp(log_norm))
                quote_true = sp.from_inr_per_10g_fine(norm)

                # liquidity: concentrated in the front month, tiered by symbol
                base = cfg.liquidity.get(sym, 100.0)
                decay = 0.30 ** rank
                vol = float(self.rng.gamma(2.0, base * decay / 2.0))
                vol = vol * (0.35 + 0.65 * np.exp(-tau * 2.0))
                volume = int(max(0, round(vol)))
                oi = int(max(0, round(volume * self.rng.uniform(2.5, 9.0)
                                      + 5 * decay * base)))

                derived = volume < cfg.derived_settlement_threshold
                if derived:
                    # No genuine trade. The exchange publishes a DERIVED
                    # settlement: it tracks level and carry but does NOT reflect
                    # today's relative-value dislocation. Filling here is fiction.
                    log_smooth = self.L[t] + self.b[t] * tau + c_s
                    quote_true = sp.from_inr_per_10g_fine(float(np.exp(log_smooth)))
                    volume = 0

                close = _round_tick(quote_true, sp.tick_size_inr)
                rng_frac = abs(float(self.rng.normal(0, 0.004))) + 0.0008
                hi = _round_tick(close * (1 + rng_frac), sp.tick_size_inr)
                lo = _round_tick(close * (1 - rng_frac), sp.tick_size_inr)
                op = _round_tick(lo + (hi - lo) * float(self.rng.uniform(0.15, 0.85)),
                                 sp.tick_size_inr)
                if derived:
                    op = hi = lo = close   # no trades: OHLC collapses onto settlement

                sym_out = "{:<10s}".format(sym) if cfg.inject_space_padding else sym
                mon = _pycal.month_abbr[exp.month].upper()
                rows.append({
                    "Date": served.strftime("%m/%d/%Y"),   # RESPONSE is MM/DD/YYYY
                    "Symbol": sym_out,
                    "ExpiryDate": "{:02d}{}{}".format(exp.day, mon, exp.year),
                    "Open": op, "High": hi, "Low": lo, "Close": close,
                    "Volume": volume, "OpenInterest": oi,
                })
                truth.append({
                    "date": served, "symbol": sym, "expiry_date": exp,
                    "contract_id": "{}|{}".format(sym, exp.isoformat()),
                    "true_L": float(self.L[t]), "true_b": float(self.b[t]),
                    "true_c_s": c_s, "true_residual": e_st, "true_idio": eps,
                    "true_tau": tau, "true_norm": norm, "is_derived": derived,
                })

        self._truth_rows.extend(truth)

        df = pd.DataFrame(rows, columns=RAW_COLUMNS)
        if cfg.inject_duplicate_rows > 0 and len(df):
            k = int(self.rng.binomial(len(df), cfg.inject_duplicate_rows))
            if k > 0:
                idx = self.rng.choice(len(df), size=k, replace=False)
                df = pd.concat([df, df.iloc[idx]], ignore_index=True)
        return df

    # ---------------- ground truth ------------------------------------------

    def truth(self) -> pd.DataFrame:
        """
        The true decomposition for every row generated so far.

        Join on (date, contract_id) to score a model: correlate an estimated
        residual against `true_residual`, or confirm an estimated structural
        premium recovers `true_c_s`.
        """
        if not self._truth_rows:
            return pd.DataFrame()
        t = pd.DataFrame(self._truth_rows)
        return t.drop_duplicates(subset=["date", "contract_id"]).reset_index(drop=True)

    def session_list(self) -> list:
        return list(self.sessions)


def _round_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return float(round(round(x / tick) * tick, 4))
