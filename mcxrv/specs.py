"""
Contract specifications for MCX gold futures.

The single most important distinction encoded here is between:

  trading_unit_g  -- how much metal one lot delivers  (drives HEDGE RATIOS)
  quote_base_g    -- how much metal the quoted price refers to (drives PRICE NORMALISATION)

These are NOT the same. GOLDM has a 100 g trading unit but is quoted per 10 g.
Confusing the two is the classic error in this problem: it makes GOLDM look 10x
mispriced against GOLDTEN, when in fact the two share a quotation base and differ
only in purity.

Bhavcopy `Close` is always the QUOTE, i.e. rupees per `quote_base_g` grams of
`purity`-fine metal. It is never the contract notional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    trading_unit_g: float      # metal delivered per lot
    quote_base_g: float        # metal the quote refers to
    purity: float              # fineness, e.g. 0.995
    expiry_day_lo: int         # expiry window, day-of-month (inclusive)
    expiry_day_hi: int
    deliverable: str           # physical form of the deliverable
    listed_from: date | None   # first date the contract family is available
    tender_days: int           # business days before expiry when tender period opens
    tick_size_inr: float       # minimum price increment, in quote terms
    notes: str = ""

    # ---- derived -----------------------------------------------------------

    @property
    def fine_g_per_lot(self) -> float:
        """Grams of PURE gold delivered per lot. This is the hedging unit."""
        return self.trading_unit_g * self.purity

    @property
    def fine_g_per_quote_base(self) -> float:
        """Grams of PURE gold the quoted price refers to."""
        return self.quote_base_g * self.purity

    def to_inr_per_10g_fine(self, quote: float) -> float:
        """
        Convert a raw bhavcopy quote to the common basis: INR per 10 g of FINE gold.

        GOLDM  @ 12000 -> 12000 * 10 / (10 * 0.995) = 12060.30
        GOLDTEN@ 12100 -> 12100 * 10 / (10 * 0.999) = 12112.11
        GOLDG  @  9700 ->  9700 * 10 / ( 8 * 0.999) = 12137.14
        GOLDP  @  1215 ->  1215 * 10 / ( 1 * 0.999) = 12162.16
        """
        return quote * 10.0 / self.fine_g_per_quote_base

    def from_inr_per_10g_fine(self, norm: float) -> float:
        """Inverse of `to_inr_per_10g_fine` -- back to a tradable quote."""
        return norm * self.fine_g_per_quote_base / 10.0

    def lot_notional(self, quote: float) -> float:
        """Rupee value of one lot at a given quote."""
        return quote * self.trading_unit_g / self.quote_base_g


# Expiry windows and purities per the MCX contract specifications.
# GOLDM expires 3rd-5th; the 999-purity trio all expire 27th-31st. That mismatch
# is why GOLDM is never maturity-matched to the others and always needs a carry
# bridge, while GOLDTEN/GOLDGUINEA/GOLDPETAL form a clean same-expiry triangle.
SPECS: dict[str, ContractSpec] = {
    "GOLDM": ContractSpec(
        symbol="GOLDM",
        trading_unit_g=100.0,
        quote_base_g=10.0,
        purity=0.995,
        expiry_day_lo=3,
        expiry_day_hi=5,
        deliverable="bar",
        listed_from=None,
        tender_days=3,
        tick_size_inr=1.0,
        notes="Gold Mini. 995 fineness bar. Expiry window differs from the 999 trio "
              "by ~23 days -- every GOLDM cross-spread contains a carry term.",
    ),
    "GOLDTEN": ContractSpec(
        symbol="GOLDTEN",
        trading_unit_g=10.0,
        quote_base_g=10.0,
        purity=0.999,
        expiry_day_lo=27,
        expiry_day_hi=31,
        deliverable="bar",
        listed_from=date(2025, 1, 1),
        tender_days=3,
        tick_size_inr=1.0,
        notes="Listed 2025 onward -- short history, too few independent "
              "observations for strong significance claims.",
    ),
    "GOLDGUINEA": ContractSpec(
        symbol="GOLDGUINEA",
        trading_unit_g=8.0,
        quote_base_g=8.0,
        purity=0.999,
        expiry_day_lo=27,
        expiry_day_hi=31,
        deliverable="coin",
        listed_from=None,
        tender_days=3,
        tick_size_inr=1.0,
        notes="Deliverable is a minted 8 g coin. Fabrication cost is embedded in "
              "the deliverable, so a PERSISTENT premium over bar contracts is "
              "economically justified, not a mispricing.",
    ),
    "GOLDPETAL": ContractSpec(
        symbol="GOLDPETAL",
        trading_unit_g=1.0,
        quote_base_g=1.0,
        purity=0.999,
        expiry_day_lo=27,
        expiry_day_hi=31,
        deliverable="coin",
        listed_from=None,
        tender_days=3,
        tick_size_inr=0.5,
        notes="1 g lot. Highest fabrication premium per gram and the thinnest "
              "book. Settlement prices on far months may be derived, not traded.",
    ),
}

# Symbols that share an expiry window, hence are directly maturity-matched.
MATURITY_MATCHED_GROUPS: list[list[str]] = [
    ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"],
]

ALL_SYMBOLS = list(SPECS)
BAR_SYMBOLS = [s for s, sp in SPECS.items() if sp.deliverable == "bar"]
COIN_SYMBOLS = [s for s, sp in SPECS.items() if sp.deliverable == "coin"]


# ---------------------------------------------------------------------------
# effective dating
# ---------------------------------------------------------------------------
#
# Contract specifications get revised. When an exchange changes a purity, a lot size
# or a quotation base, the revision applies to CONTRACTS LISTED AFTER IT -- not
# retroactively to contracts already trading. Two expiries of the same symbol can
# therefore be governed by different specs on the same session.
#
# That is why the governing date is the contract's LISTING date, not the session
# date, and why the per-contract snapshot stored in the database is the correct
# grain rather than a convenience.
#
# Without this, a single static spec per symbol would silently apply today's values
# to all of history the moment anything was revised -- rewriting every past price
# with no error raised. Today every symbol has exactly one open-ended version, so
# nothing changes numerically; the machinery exists so that a future revision is a
# data change rather than a silent corruption.


@dataclass(frozen=True)
class SpecVersion:
    """A contract specification and the window of listing dates it governs."""
    spec: ContractSpec
    valid_from: date | None = None   # None: open at the start of history
    valid_to: date | None = None     # None: still in force

    def covers(self, d: date | None) -> bool:
        if d is None:
            return self.valid_to is None      # unspecified date means "current"
        if self.valid_from is not None and d < self.valid_from:
            return False
        if self.valid_to is not None and d >= self.valid_to:
            return False
        return True


# One open-ended version per symbol today. `add_version` extends this.
SPEC_HISTORY: dict[str, list[SpecVersion]] = {
    sym: [SpecVersion(spec=sp)] for sym, sp in SPECS.items()
}


def _key(symbol: str) -> str:
    key = (symbol or "").strip().upper()
    if key not in SPEC_HISTORY:
        raise KeyError(f"unknown gold symbol {symbol!r} (normalised to {key!r})")
    return key


def get(symbol: str, on: date | None = None) -> ContractSpec:
    """
    Look up the spec governing `symbol`, tolerating MCX's space padding.

    `on` is the contract's LISTING date -- the date that decides which revision
    applies. Omitting it returns the current specification, which is correct for
    anything forward-looking and is what every existing caller wants.

    Raises if a date falls in no version's window, rather than silently falling back
    to the current spec: a gap in the spec history is a data problem to fix, not
    something to paper over with today's numbers.
    """
    key = _key(symbol)
    versions = SPEC_HISTORY[key]
    if len(versions) == 1 and on is None:
        return versions[0].spec
    for v in versions:
        if v.covers(on):
            return v.spec
    if on is None:
        return versions[-1].spec
    raise KeyError(
        f"no {key} specification covers listing date {on}; "
        f"known windows: {[(v.valid_from, v.valid_to) for v in versions]}")


def versions(symbol: str) -> list[SpecVersion]:
    """Every specification version for a symbol, oldest first."""
    return list(SPEC_HISTORY[_key(symbol)])


def add_version(symbol: str, spec: ContractSpec, valid_from: date) -> None:
    """
    Record a specification revision effective for contracts listed from `valid_from`.

    Closes the currently open version at that date so the windows stay contiguous and
    non-overlapping. `SPECS[symbol]` is repointed at the newest version, so code that
    asks for "the current spec" keeps working.
    """
    key = _key(symbol)
    hist = SPEC_HISTORY[key]
    for i, v in enumerate(hist):
        if v.valid_to is None:
            hist[i] = SpecVersion(spec=v.spec, valid_from=v.valid_from,
                                  valid_to=valid_from)
    hist.append(SpecVersion(spec=spec, valid_from=valid_from, valid_to=None))
    hist.sort(key=lambda v: (v.valid_from or date.min))
    SPECS[key] = hist[-1].spec


def has_revisions(symbol: str | None = None) -> bool:
    """True when any symbol (or the named one) carries more than one version."""
    if symbol is not None:
        return len(SPEC_HISTORY[_key(symbol)]) > 1
    return any(len(v) > 1 for v in SPEC_HISTORY.values())


def is_maturity_matched(sym_a: str, sym_b: str) -> bool:
    a, b = sym_a.strip().upper(), sym_b.strip().upper()
    return any(a in g and b in g for g in MATURITY_MATCHED_GROUPS)


def hedge_ratio_fine(sym_long: str, sym_short: str) -> float:
    """
    Lots of `sym_short` per lot of `sym_long` for FINE-GRAM neutrality.

    GOLDM  99.500 g fine/lot
    GOLDTEN 9.990 g fine/lot  -> ratio 9.9600
    GOLDGUINEA 7.992 g        -> ratio 12.4500
    GOLDPETAL  0.999 g        -> ratio 99.5996

    These are not integers, so a real position carries a small unhedged gold
    residual. `backtest` reports it rather than assuming it away.
    """
    return get(sym_long).fine_g_per_lot / get(sym_short).fine_g_per_lot
