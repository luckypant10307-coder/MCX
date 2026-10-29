"""
SPAN initial margin, and capital actually at risk.

WHAT IS REAL HERE AND WHAT IS ASSUMED
--------------------------------------
The SPAN *algorithm* below is implemented faithfully: the 16-scenario risk array,
the scanning risk maximum, intra-commodity (calendar) spread charges, inter-commodity
spread credits, the spot-month charge, and Initial Margin = SPAN + ELM.

The SPAN *parameters* are NOT derivable from bhavcopy. MCX publishes price scan
ranges, spread charges and inter-commodity credits daily in separate risk-parameter
files, set by its own risk policy. Nothing in Open/High/Low/Close/Volume/OpenInterest
determines them.

So every parameter here is an ASSUMPTION, in exactly the way the bid-ask spread is,
and it gets the same treatment: a documented default, a data-driven fallback, a
sensitivity sweep, and a loader (`SpanParameters.from_mcx_file`) for when real files
are available. A margin number presented as fact when it was assumed is worse than
no margin number at all.

WHY IT MATTERS FOR THIS STRATEGY
--------------------------------
Sharpe is scale-invariant, so it does not move when the capital denominator changes.
Return on capital does, and it is the only denominator that allows comparison with
anything else.

The direction is counter-intuitive and worth stating: a spread is margin-EFFICIENT
because SPAN grants large inter-commodity credits, so the same rupee loss sits on a
much smaller capital base. Accounting for margin properly makes a losing spread
strategy look considerably WORSE, not better.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import specs

# ---------------------------------------------------------------------------
# the 16 SPAN scenarios
# ---------------------------------------------------------------------------
#
# (price_fraction_of_scan_range, volatility_direction, weight)
#
# Scenarios 1-14 sweep the price in thirds of the scan range against volatility up
# and down. Scenarios 15-16 are the "extreme" moves at twice the scan range, carried
# at a fractional weight because a move that large is not expected to be covered in
# full by initial margin.
#
# Volatility direction is retained for completeness and is inert for futures: a
# future has no vega, so its P&L is linear in price alone. It matters the moment
# options are added to the portfolio, which is why the structure is kept.
SPAN_SCENARIOS: tuple = (
    (0.0, +1, 1.0), (0.0, -1, 1.0),
    (+1 / 3, +1, 1.0), (+1 / 3, -1, 1.0),
    (-1 / 3, +1, 1.0), (-1 / 3, -1, 1.0),
    (+2 / 3, +1, 1.0), (+2 / 3, -1, 1.0),
    (-2 / 3, +1, 1.0), (-2 / 3, -1, 1.0),
    (+1.0, +1, 1.0), (+1.0, -1, 1.0),
    (-1.0, +1, 1.0), (-1.0, -1, 1.0),
    (+2.0, 0, 0.35), (-2.0, 0, 0.35),
)


@dataclass
class SpanParameters:
    """
    Risk parameters. Defaults are plausible for MCX gold; none is a measurement.

    `price_scan_range_pct` is the parameter that dominates the result. MCX sets it
    from its own volatility model and revises it as markets move; the default here is
    a static stand-in, and `derive_scan_range` estimates it from observed volatility
    when no published value is available.
    """
    # scanning
    price_scan_range_pct: dict = field(default_factory=lambda: {
        "GOLDM": 0.055, "GOLDTEN": 0.055, "GOLDGUINEA": 0.055, "GOLDPETAL": 0.055,
    })
    vol_scan_range_pct: float = 0.0        # inert for futures; kept for options

    # intra-commodity (calendar) spread charge, INR per spread of one lot-equivalent
    intra_spread_charge_pct: float = 0.005  # of one leg's notional

    # inter-commodity spread credit: the fraction of combined scanning risk waived
    # when offsetting positions in recognised related commodities are held. THE
    # dominant parameter for a cross-contract spread strategy.
    inter_spread_credit: float = 0.70

    # Published cross-margin files state a credit PER RECOGNISED PAIR, not one rate
    # for the whole complex. When present this maps frozenset-like sorted tuples of
    # two symbols to their rate and takes precedence; `inter_spread_credit` remains
    # the fallback for pairs no file names. Keeping both lets a report say which
    # legs were credited on published authority and which on an assumption.
    inter_spread_credit_pairs: dict = field(default_factory=dict)

    # which symbols MCX recognises as one spreadable complex. All four are the same
    # underlying metal, so a credit is plausible -- but recognition is MCX's decision,
    # not a fact derivable from the data, hence `recognised_groups`.
    recognised_groups: tuple = (("GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"),)

    # --- delivery period -----------------------------------------------------
    # MCX charges a delivery-period margin computed as a 5-day 99% VaR, subject to a
    # floor. That is a different object from a flat percentage: it rises with realised
    # volatility, so the charge is largest exactly when the market is most disturbed.
    delivery_var_days: int = 5
    delivery_var_z: float = 2.326        # one-sided 99%
    delivery_margin_floor_pct: float = 0.03
    spot_month_sessions: int = 5         # sessions before expiry the charge begins

    # --- extreme loss margin -------------------------------------------------
    elm_pct: float = 0.0125
    elm_min_pct: float = 0.01            # specification minimum; ELM never falls below
    # Published files state ELM and additional margin PER COMMODITY. Folding them
    # into one scalar would apply one contract's rate to every leg -- and since these
    # contracts carry different rates, that is wrong in both directions at once.
    elm_pct_by_symbol: dict = field(default_factory=dict)
    additional_pct_by_symbol: dict = field(default_factory=dict)

    # --- initial margin floor ------------------------------------------------
    # MCX gold specifies initial margin as the HIGHER of a percentage of contract
    # value and the SPAN requirement. The floor is decisive for a spread: SPAN with a
    # large inter-commodity credit can fall well under it, at which point the credit
    # stops mattering and the floor sets the capital blocked.
    im_floor_pct: float = 0.06
    # Basis for the floor. MCX states it against contract value; whether a recognised
    # spread is floored on GROSS notional or on the NET delta is exchange policy and
    # is not derivable from the data, so both are supported and the choice is stated.
    im_floor_basis: str = "gross"        # "gross" | "net_delta" | "none"

    # --- concentration -------------------------------------------------------
    # Additional margin once a position is large relative to the contract's open
    # interest. Exchanges impose this because an outsized holder cannot exit at the
    # prices the risk model assumes.
    concentration_threshold: float = 0.05   # share of open interest
    concentration_surcharge_pct: float = 0.02

    # additional / ad-hoc margin the exchange may impose
    additional_margin_pct: float = 0.0

    source: str = "assumed defaults (not MCX published parameters)"

    # ---- loaders ----------------------------------------------------------

    def __post_init__(self):
        """
        Normalise the per-pair credit keys.

        `credit_for` looks up a sorted TUPLE of two symbols, but JSON has no tuple
        keys, so a parameter set round-tripped through `--span-params` arrives with
        "GOLDM|GOLDTEN" strings instead. Left unnormalised every lookup misses, and
        because a populated mapping suppresses the scalar fallback, EVERY spread
        would silently be credited at zero -- published rates loaded and then
        discarded, with no error anywhere. Normalising here means the lookup cannot
        depend on which door the parameters came through.
        """
        if not self.inter_spread_credit_pairs:
            return
        fixed = {}
        for k, v in self.inter_spread_credit_pairs.items():
            if isinstance(k, str):
                parts = [p for p in re.split(r"[|/,]", k) if p.strip()]
                if len(parts) != 2:
                    raise ValueError(
                        "inter_spread_credit_pairs key {!r} does not name two "
                        "commodities".format(k))
                key = tuple(sorted(p.strip().upper() for p in parts))
            else:
                key = tuple(sorted(str(p).strip().upper() for p in k))
            fixed[key] = float(v)
        object.__setattr__(self, "inter_spread_credit_pairs", fixed) \
            if getattr(type(self), "__dataclass_params__").frozen \
            else setattr(self, "inter_spread_credit_pairs", fixed)

    def elm_for(self, symbol: str) -> float:
        """ELM for one commodity, never below the specification minimum."""
        rate = self.elm_pct_by_symbol.get(symbol.strip().upper(), self.elm_pct)
        return max(float(rate), float(self.elm_min_pct))

    def additional_for(self, symbol: str) -> float:
        return float(self.additional_pct_by_symbol.get(
            symbol.strip().upper(), self.additional_margin_pct))

    def credit_for(self, sym_a: str, sym_b: str) -> float:
        """
        The credit rate for one recognised leg pair.

        A published per-pair rate wins over the scalar assumption. An unrecognised
        pair earns nothing -- silently falling back to the generous default would
        turn "MCX does not recognise this spread" into "MCX credits it at 70%",
        which is the most expensive possible mistake for this strategy.
        """
        if not self.inter_spread_credit_pairs:
            return float(self.inter_spread_credit)
        key = tuple(sorted((sym_a.strip().upper(), sym_b.strip().upper())))
        return float(self.inter_spread_credit_pairs.get(key, 0.0))

    @classmethod
    def from_mcx_file(cls, path: str) -> "SpanParameters":
        """
        Load real parameters from a JSON extract of an MCX risk-parameter file.

        Expected keys mirror the field names. Anything absent keeps its default, and
        `source` records the file so a report can state which numbers were published
        and which were assumed.
        """
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        kw = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        kw["source"] = "MCX risk parameters: {}".format(os.path.basename(path))
        return cls(**kw)

    def scan_pct(self, symbol: str) -> float:
        return float(self.price_scan_range_pct.get(symbol.strip().upper(), 0.055))


def derive_scan_range(rows: pd.DataFrame, k: float = 3.5,
                      floor_pct: float = 0.03, halflife: int = 60) -> dict:
    """
    Estimate a price scan range per symbol from observed volatility.

    A scan range is meant to cover a severe one-day move, so `k` standard deviations
    of the EWMA daily return is the natural construction; k=3.5 corresponds to roughly
    a 99.95% one-day coverage under normality, which is the order MCX targets.

    This is a DEFENSIBLE STAND-IN, not a reconstruction of MCX's number. MCX sets the
    scan range by its own risk policy, applies floors and discretionary add-ons, and
    revises it on its own schedule. Use `SpanParameters.from_mcx_file` when the real
    values are available; use this to keep the model honest when they are not.
    """
    out = {}
    for sym, g in rows.sort_values("date").groupby("symbol"):
        # one representative series per symbol: the most-traded contract each session
        idx = g.groupby("date")["volume"].idxmax()
        s = g.loc[idx].sort_values("date")
        r = np.log(s["norm_close"].astype(float)).diff().dropna()
        if len(r) < 30:
            out[sym] = floor_pct
            continue
        ew = r.ewm(halflife=halflife, min_periods=20).std().iloc[-1]
        out[sym] = float(max(k * float(ew), floor_pct))
    return out


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarginPosition:
    """One leg. `lots` is signed: positive long, negative short."""
    symbol: str
    contract_id: str
    lots: int
    price: float                  # the contract's own quote
    sessions_to_expiry: int = 99
    # Trailing daily volatility of this contract, for the delivery-period VaR and
    # for a volatility-responsive scan range. None falls back to the static policy.
    sigma_daily: float | None = None
    open_interest: float | None = None    # for the concentration surcharge

    @property
    def multiplier(self) -> float:
        sp = specs.get(self.symbol)
        return sp.trading_unit_g / sp.quote_base_g

    @property
    def signed_value(self) -> float:
        """Signed rupee exposure. This is the delta a linear future carries."""
        return self.lots * self.multiplier * self.price

    @property
    def notional(self) -> float:
        return abs(self.signed_value)


# ---------------------------------------------------------------------------
# the SPAN computation
# ---------------------------------------------------------------------------

def scanning_risk(positions, params: SpanParameters) -> float:
    """
    Worst weighted loss across the 16 scenarios, for one combined commodity.

    Offsetting positions within the commodity net here, because the scan applies a
    single price move to the whole group -- which is precisely why SPAN then adds an
    intra-commodity spread charge: perfect correlation across expiries is assumed by
    the scan and is not true in the market.
    """
    if not positions:
        return 0.0
    worst = 0.0
    for frac, _vol_dir, weight in SPAN_SCENARIOS:
        pnl = 0.0
        for p in positions:
            move = params.scan_pct(p.symbol) * frac
            pnl += p.signed_value * move
        loss = max(-pnl, 0.0) * weight
        worst = max(worst, loss)
    return worst


def intra_commodity_charge(positions, params: SpanParameters) -> float:
    """
    Charge for calendar spreads inside one commodity.

    The scan nets long and short deltas across expiries as though they moved
    identically. They do not, so SPAN charges for each spread formed. The number of
    spreads is the matched portion of long against short exposure.
    """
    longs = sum(p.signed_value for p in positions if p.signed_value > 0)
    shorts = -sum(p.signed_value for p in positions if p.signed_value < 0)
    matched = min(longs, shorts)
    return matched * params.intra_spread_charge_pct


def delivery_var_pct(sigma_daily: float | None, params: SpanParameters) -> float:
    """
    Delivery-period margin rate as a 5-day 99% VaR, subject to the floor.

        rate = max(floor, z(99%) * sigma_daily * sqrt(5))

    This is the shape MCX's gold specification uses, and it matters that it is a VaR
    rather than a fixed percentage: it RISES with realised volatility, so the charge
    is heaviest exactly when the market is most disturbed and a position is hardest
    to exit. A flat percentage would understate the requirement in precisely the
    conditions that make delivery risky.
    """
    floor = params.delivery_margin_floor_pct
    if sigma_daily is None or not np.isfinite(sigma_daily) or sigma_daily <= 0:
        return floor
    var = params.delivery_var_z * float(sigma_daily) * np.sqrt(params.delivery_var_days)
    return float(max(floor, var))


def delivery_charge(positions, params: SpanParameters) -> float:
    """Delivery-period margin on every position inside the tender window."""
    total = 0.0
    for p in positions:
        if p.sessions_to_expiry <= params.spot_month_sessions:
            total += p.notional * delivery_var_pct(p.sigma_daily, params)
    return total


# Retained under the old name so existing callers keep working; the flat charge is
# now the VaR floor rather than a separate policy.
def spot_month_charge(positions, params: SpanParameters) -> float:
    return delivery_charge(positions, params)


def concentration_charge(positions, params: SpanParameters) -> float:
    """
    Surcharge once a holding is large relative to the contract's open interest.

    An outsized position cannot be liquidated at the prices the risk model assumes,
    so the exchange demands more. This binds hardest on exactly the thin contracts
    this strategy wants to trade: a few hundred GOLDPETAL lots is a large share of
    that book, while the same rupee amount of GOLDM is negligible.
    """
    total = 0.0
    for p in positions:
        oi = p.open_interest
        if oi is None or not np.isfinite(oi) or oi <= 0:
            continue
        share = abs(p.lots) / float(oi)
        if share > params.concentration_threshold:
            excess = (share - params.concentration_threshold) / max(
                params.concentration_threshold, 1e-9)
            total += p.notional * params.concentration_surcharge_pct * min(excess, 5.0)
    return total


def _groups(positions, params: SpanParameters) -> dict:
    """Split a portfolio into SPAN combined commodities, keyed by symbol."""
    out: dict = {}
    for p in positions:
        out.setdefault(p.symbol.strip().upper(), []).append(p)
    return out


def inter_commodity_credit(by_symbol: dict, scan_by_symbol: dict,
                           params: SpanParameters) -> float:
    """
    Credit for offsetting positions across recognised related commodities.

    SPAN forms inter-commodity spreads from the delta remaining after the intra-
    commodity scan, and waives a fraction of the scanning risk those matched deltas
    attract. A long GOLDM against a short GOLDPETAL is the same metal twice, so
    almost all of the combined scanning risk is spurious.

    The matched fraction is computed on signed exposure and the credit applied
    proportionally to each leg's scanning risk. This is the delta-matching step of
    real SPAN; the credit RATE is exchange policy and is an assumption here.
    """
    total = 0.0
    for group in params.recognised_groups:
        members = [s for s in group if s in by_symbol]
        if len(members) < 2:
            continue
        longs = {s: sum(p.signed_value for p in by_symbol[s] if p.signed_value > 0)
                 for s in members}
        shorts = {s: -sum(p.signed_value for p in by_symbol[s] if p.signed_value < 0)
                  for s in members}

        # Scanning risk per rupee of exposure, so a matched chunk can be costed
        # against whichever legs actually formed it.
        rate = {}
        for s in members:
            gross_s = longs[s] + shorts[s]
            rate[s] = (scan_by_symbol.get(s, 0.0) / gross_s) if gross_s > 0 else 0.0

        rem_long = {s: v for s, v in longs.items() if v > 0}
        rem_short = {s: v for s, v in shorts.items() if v > 0}

        # Pair-by-pair matching, because a published cross-margin file grants a rate
        # per recognised pair and an unrecognised pair earns nothing.
        #
        # The order is ALPHABETICAL by symbol. That is deterministic and is not
        # chosen to maximise the credit -- a matching order picked for its result
        # would be assuming the answer this model exists to test. It is NOT the
        # exchange's spread priority: real SPAN forms spreads in the priority its
        # file states, `SpreadCredit.priority` is parsed but not yet carried into
        # `SpanParameters`, and where published rates differ between pairs the order
        # changes the credit. Pinning that needs a real file to read the priorities
        # from; until then this is a documented approximation, not the rule.
        pairs = sorted(((a, b) for a in rem_long for b in rem_short if a != b),
                       key=lambda ab: (ab[0], ab[1]))
        for a, b in pairs:
            avail = min(rem_long.get(a, 0.0), rem_short.get(b, 0.0))
            if avail <= 0:
                continue
            cr = params.credit_for(a, b)
            if cr <= 0:
                continue
            rem_long[a] -= avail
            rem_short[b] -= avail
            total += avail * (rate[a] + rate[b]) * cr
    return total


def span_margin(positions, params: SpanParameters | None = None) -> dict:
    """
    Full SPAN + ELM breakdown for a portfolio.

    Returns an itemised dict so a report can show WHERE the margin comes from rather
    than a single opaque number:

        scanning_risk      worst-case scenario loss, netted within each commodity
        intra_spread       calendar spread charge
        spot_month         delivery-window add-on
        inter_credit       credit for offsetting related commodities (subtracted)
        span               scanning + intra + spot - credit, floored at zero
        elm                Extreme Loss Margin on notional
        additional         any ad-hoc exchange margin
        initial_margin     SPAN + ELM + additional: the capital actually blocked
        notional           gross two-leg notional, for the margin-to-notional ratio
    """
    params = params or SpanParameters()
    positions = [p for p in positions if p.lots != 0]
    if not positions:
        return {"scanning_risk": 0.0, "intra_spread": 0.0, "spot_month": 0.0,
                "inter_credit": 0.0, "span": 0.0, "elm": 0.0, "additional": 0.0,
                "initial_margin": 0.0, "notional": 0.0, "margin_pct_notional": 0.0}

    by_symbol = _groups(positions, params)
    scan_by_symbol = {s: scanning_risk(ps, params) for s, ps in by_symbol.items()}
    scan_total = sum(scan_by_symbol.values())
    intra = sum(intra_commodity_charge(ps, params) for ps in by_symbol.values())
    spot = delivery_charge(positions, params)
    conc = concentration_charge(positions, params)
    credit = inter_commodity_credit(by_symbol, scan_by_symbol, params)

    span = max(scan_total + intra + spot + conc - credit, 0.0)
    notional = sum(p.notional for p in positions)
    net_delta = abs(sum(p.signed_value for p in positions))

    # ELM and additional margin are charged PER LEG at that commodity's own rate;
    # these contracts do not share one. ELM never falls below the specification
    # minimum.
    elm = sum(p.notional * params.elm_for(p.symbol) for p in positions)
    additional = sum(p.notional * params.additional_for(p.symbol) for p in positions)
    computed = span + elm + additional

    # MCX states initial margin as the HIGHER of a percentage of contract value and
    # the SPAN requirement. For a spread this is often decisive: SPAN with a large
    # inter-commodity credit can fall far below the floor, and at that point the
    # credit stops mattering entirely -- the floor sets the capital blocked.
    if params.im_floor_basis == "gross":
        floor = notional * params.im_floor_pct
    elif params.im_floor_basis == "net_delta":
        floor = net_delta * params.im_floor_pct
    else:
        floor = 0.0
    im = max(computed, floor)

    return {
        "scanning_risk": scan_total, "intra_spread": intra, "spot_month": spot,
        "delivery": spot, "concentration": conc,
        "inter_credit": credit, "span": span, "elm": elm, "additional": additional,
        "computed_margin": computed, "im_floor": floor,
        "floor_binds": bool(floor > computed),
        "initial_margin": im, "notional": notional, "net_delta": net_delta,
        "margin_pct_notional": im / notional if notional > 0 else 0.0,
    }


def initial_margin(positions, params: SpanParameters | None = None) -> float:
    """Just the number, for callers that do not need the breakdown."""
    return span_margin(positions, params)["initial_margin"]


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def sweep_credit(positions, params: SpanParameters | None = None,
                 credits=(0.0, 0.3, 0.5, 0.7, 0.85)) -> pd.DataFrame:
    """
    Margin against the inter-commodity credit rate.

    The single most consequential unknown for a spread strategy: it decides whether
    the position blocks two legs' worth of margin or a fraction of one.
    """
    from dataclasses import replace as _replace
    base = params or SpanParameters()
    rows = []
    for c in credits:
        m = span_margin(positions, _replace(base, inter_spread_credit=c))
        rows.append({"inter_spread_credit": c,
                     "span": round(m["span"], 0),
                     "initial_margin": round(m["initial_margin"], 0),
                     "pct_of_notional": round(m["margin_pct_notional"], 4)})
    return pd.DataFrame(rows)


def sweep_scan_range(positions, params: SpanParameters | None = None,
                     scans=(0.03, 0.045, 0.055, 0.07, 0.09)) -> pd.DataFrame:
    """Margin against the price scan range, the other dominant parameter."""
    from dataclasses import replace as _replace
    base = params or SpanParameters()
    rows = []
    for s in scans:
        p = _replace(base, price_scan_range_pct={k: s for k in base.price_scan_range_pct})
        m = span_margin(positions, p)
        rows.append({"price_scan_range": s,
                     "scanning_risk": round(m["scanning_risk"], 0),
                     "initial_margin": round(m["initial_margin"], 0),
                     "pct_of_notional": round(m["margin_pct_notional"], 4)})
    return pd.DataFrame(rows)


def volatility_schedule(rows: pd.DataFrame, k: float = 3.5, floor_pct: float = 0.03,
                        halflife: int = 60, revision_sessions: int = 21) -> dict:
    """
    Per-session, per-symbol scan range from TRAILING volatility.

    Margin is not static. When gold volatility rises the exchange widens the scan
    range and every open position suddenly requires more cash -- which is exactly when
    a losing spread is least able to supply it. A single scan range for the whole
    sample hides that entirely and understates peak capital.

    Strictly trailing, and stepped on a revision cycle rather than daily, because an
    exchange revises parameters periodically rather than continuously. Returns
    {(date, symbol): scan_pct} for the backtest to look up as it walks forward.

    Volatility here is of the NORMALISED price, so it is comparable across contracts
    of different sizes -- a 1 g petal and a 100 g bar move by the same percentage.
    """
    out: dict = {}
    for sym, g in rows.sort_values("date").groupby("symbol"):
        idx = g.groupby("date")["volume"].idxmax()
        s = g.loc[idx].sort_values("date")
        r = np.log(s["norm_close"].astype(float)).diff()
        ew = r.ewm(halflife=halflife, min_periods=20).std()
        dates = list(s["date"])
        current = floor_pct
        for i, d in enumerate(dates):
            if i % revision_sessions == 0:
                v = ew.iloc[i] if i < len(ew) else np.nan
                if np.isfinite(v) and v > 0:
                    current = float(max(k * float(v), floor_pct))
            out[(d, sym)] = current
    return out


def params_for_session(base: SpanParameters, asof, schedule: dict | None,
                       fallback: dict | None = None) -> SpanParameters:
    """
    Base parameters with that session's scan ranges substituted in.

    `schedule` is the preferred source -- published files, when they exist for this
    session. `fallback` is consulted only where the schedule has no entry, and is
    meant to be the ESTIMATED volatility schedule.

    The distinction is a look-ahead control, not tidiness. Published parameter files
    cover the dates they were issued for and no others. Without a fallback, an
    uncovered session silently keeps whatever is in `base` -- and when `base` was
    folded from a set of published files, that is the NEWEST file's parameters
    applied retroactively to a session that never saw them. That flatters the result
    exactly when margin was rising, which is when the strategy could least afford it.
    `scan_source` records which of the two supplied each session so the distinction
    survives into the report.
    """
    from dataclasses import replace as _replace
    if not schedule and not fallback:
        return base

    published = {sym: v for (d, sym), v in (schedule or {}).items() if d == asof}
    estimated = {sym: v for (d, sym), v in (fallback or {}).items() if d == asof}
    if not published and not estimated:
        return base

    merged = dict(base.price_scan_range_pct)
    merged.update(estimated)
    merged.update(published)          # published wins where both exist
    src = base.source
    if published and estimated:
        src += " [scan: published for {}, estimated for {}]".format(
            ",".join(sorted(published)), ",".join(sorted(set(estimated) - set(published))))
    elif published:
        src += " [scan: published]"
    elif estimated:
        src += " [scan: ESTIMATED -- no published file for this session]"
    return _replace(base, price_scan_range_pct=merged, source=src)


def stress(positions, params: SpanParameters | None = None) -> pd.DataFrame:
    """
    What happens to required margin when risk conditions deteriorate.

    A margin number computed under calm conditions is the least useful one, because
    margin calls arrive precisely when the position is already losing. Each row is a
    plausible adverse move in exchange risk policy or market state.

    The correlation-breakdown row deserves particular attention for a spread. The
    inter-commodity credit exists only because the legs are assumed to move together.
    They are the same metal, so the assumption is sound most of the time -- but an
    exchange can withdraw the credit, and when it does the margin on a "hedged"
    position can multiply without any position change at all.
    """
    from dataclasses import replace as _replace
    base = params or SpanParameters()
    baseline = span_margin(positions, base)["initial_margin"]

    scen = [
        ("baseline", base),
        ("volatility doubles (scan range x2)",
         _replace(base, price_scan_range_pct={k: 2 * v for k, v
                                              in base.price_scan_range_pct.items()})),
        ("exchange raises scan range 50%",
         _replace(base, price_scan_range_pct={k: 1.5 * v for k, v
                                              in base.price_scan_range_pct.items()})),
        ("spread credit halved (correlation weakens)",
         _replace(base, inter_spread_credit=base.inter_spread_credit * 0.5)),
        ("spread credit withdrawn entirely",
         _replace(base, inter_spread_credit=0.0)),
        ("ad-hoc margin imposed (+3%)",
         _replace(base, additional_margin_pct=base.additional_margin_pct + 0.03)),
    ]
    rows = []
    for label, p in scen:
        m = span_margin(positions, p)
        rows.append({
            "scenario": label,
            "initial_margin": round(m["initial_margin"], 0),
            "pct_of_notional": round(m["margin_pct_notional"], 4),
            "floor_binds": m["floor_binds"],
            "vs_baseline": round(m["initial_margin"] / baseline, 3)
            if baseline > 0 else np.nan,
        })

    # delivery period is a state change, not a parameter change
    from dataclasses import replace as _r
    in_delivery = [
        MarginPosition(p.symbol, p.contract_id, p.lots, p.price,
                       sessions_to_expiry=1, sigma_daily=p.sigma_daily,
                       open_interest=p.open_interest)
        for p in positions]
    m = span_margin(in_delivery, base)
    rows.append({
        "scenario": "position held into the delivery window",
        "initial_margin": round(m["initial_margin"], 0),
        "pct_of_notional": round(m["margin_pct_notional"], 4),
        "floor_binds": m["floor_binds"],
        "vs_baseline": round(m["initial_margin"] / baseline, 3)
        if baseline > 0 else np.nan,
    })
    return pd.DataFrame(rows)


def capital_efficiency(net_pnl: float, peak_margin: float,
                       sessions: int) -> dict:
    """
    Return on capital actually blocked, rather than on an arbitrary denominator.

    Reported alongside Sharpe, which is scale-invariant and therefore says nothing
    about capital efficiency in either direction.
    """
    if peak_margin <= 0:
        return {"return_on_margin": np.nan, "note": "no margin ever blocked"}
    roc = net_pnl / peak_margin
    years = max(sessions, 1) / 252.0
    return {
        "peak_margin": peak_margin,
        "net_pnl": net_pnl,
        "return_on_margin": roc,
        "annualised_return_on_margin": roc / years if years > 0 else np.nan,
    }
