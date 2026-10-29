"""
Load PUBLISHED MCX risk parameters, replacing the assumptions in `margin.py`.

WHERE THESE NUMBERS COME FROM
-----------------------------
Three MCXCCL publications between them determine every parameter the margin model
currently assumes:

  1. **MCX Daily SPAN Risk Parameter File** -- the PC-SPAN input, one per business
     date (and per intraday snapshot). Carries the price scan range, intra-commodity
     spread charges, spot/delivery month charges and the inter-commodity spread
     credits. This is the authoritative source for the scanning parameters.

  2. **MCXCCL Risk and Collaterals** -- the margin circulars and rate files: initial
     margin percentage floors, extreme loss margin, additional and ad-hoc margins,
     delivery-period margin, concentration margins.

  3. **MCXCCL Cross-Margin Benefit** -- which commodities are recognised as one
     spreadable complex and at what credit rate. For a cross-contract gold strategy
     this is THE dominant parameter: everything about whether a spread is capital
     efficient turns on it.

NOTHING HERE FETCHES
--------------------
MCX sits behind bot protection that refuses programmatic requests, and working around
it is not something this project does. The files are downloaded by hand into a
directory and read from there -- the same arrangement as `sources/csv_dir.py` for
bhavcopy. `sniff()` exists so an unfamiliar file can be inspected before it is
trusted.

THE PARSER REFUSES RATHER THAN GUESSES
--------------------------------------
A margin parameter read out of the wrong column is worse than no parameter at all: it
produces a specific, confident, wrong number, and every figure downstream inherits it
silently. So every reader here validates what it extracted -- ranges, dates, commodity
codes -- and raises `SpanFileError` naming the offending record instead of falling
back to a default. A parameter is either published and parsed, or assumed and
labelled; it is never quietly somewhere in between.

EFFECTIVE DATING IS A LOOK-AHEAD CONTROL, NOT BOOKKEEPING
---------------------------------------------------------
A SPAN file is stamped with a business date and describes the risk regime AS OF that
date. Applying today's file across a historical backtest is look-ahead in the margin
dimension -- the same defect this project polices in the signal, arriving through a
different door, and it flatters the result precisely when margin was rising. Files are
therefore stored per business date and looked up for the session being priced;
`params_for_session` falls back to the estimated schedule, flagged, when no file
covers a session -- `params_for_session(base, asof, published, fallback=estimated)`,
where leaving `fallback` out means an uncovered session keeps `base` and the guard
does NOT apply.
"""
from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from dataclasses import dataclass, field, replace
from datetime import date, datetime

from . import margin as margin_mod
from . import specs


class SpanFileError(ValueError):
    """Raised when a file cannot be read with confidence. Never swallowed."""


# ---------------------------------------------------------------------------
# what a published file yields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommodityRisk:
    """Published risk parameters for one combined commodity, on one business date."""
    commodity: str
    business_date: date
    price_scan_range: float          # in `scan_basis` units
    scan_basis: str                  # "pct" | "inr_per_unit"
    intra_spread_charge: float = 0.0
    spot_month_charge: float = 0.0
    elm_pct: float | None = None
    additional_pct: float | None = None
    delivery_pct: float | None = None
    contract_value: float | None = None   # needed to express an INR scan as a pct

    def scan_pct(self) -> float:
        """
        The scan range as a fraction of contract value.

        SPAN files state the scan range in currency per contract, not as a
        percentage. Converting needs the contract value, and without it the number
        cannot be compared across contracts of different sizes -- so this raises
        rather than returning a plausible-looking ratio computed from nothing.
        """
        if self.scan_basis == "pct":
            return float(self.price_scan_range)
        if not self.contract_value:
            raise SpanFileError(
                "{} scan range is stated in currency ({}) but no contract value was "
                "parsed, so it cannot be expressed as a percentage. Supply "
                "contract_value, or load a file that states the basis."
                .format(self.commodity, self.price_scan_range))
        return float(self.price_scan_range) / float(self.contract_value)


@dataclass(frozen=True)
class SpreadCredit:
    """One inter-commodity spread definition and the credit it earns."""
    business_date: date
    leg_a: str
    leg_b: str
    credit_rate: float               # 0..1, the fraction of scanning risk waived
    ratio_a: int = 1
    ratio_b: int = 1
    priority: int = 1

    def key(self) -> tuple:
        return tuple(sorted((self.leg_a, self.leg_b)))


@dataclass
class RiskFile:
    """Everything one published file contributed, with its provenance."""
    business_date: date
    exchange: str
    source_path: str
    kind: str                        # "span" | "risk_collateral" | "cross_margin"
    commodities: dict = field(default_factory=dict)      # code -> CommodityRisk
    credits: list = field(default_factory=list)          # SpreadCredit
    unrecognised: dict = field(default_factory=dict)     # record type -> count
    notes: list = field(default_factory=list)

    def label(self) -> str:
        return "{} {} ({})".format(self.exchange, self.business_date,
                                   os.path.basename(self.source_path))


# ---------------------------------------------------------------------------
# reading the container
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    """Text of a file, transparently unwrapping the single-member zips MCX ships."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise SpanFileError("{}: zip archive is empty".format(path))
            if len(names) > 1:
                # Several snapshots in one archive is normal; which one is meant is
                # not something to guess, because they differ exactly in the numbers
                # being loaded.
                raise SpanFileError(
                    "{}: archive holds {} members ({}). Extract the one you mean and "
                    "pass it directly, so the record says which snapshot was used."
                    .format(path, len(names), ", ".join(sorted(names)[:6])))
            with zf.open(names[0]) as fh:
                return fh.read().decode("utf-8", errors="replace")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def sniff(path: str, sample: int = 3) -> dict:
    """
    Describe a file without interpreting it.

    The point is to make an unfamiliar layout cheap to inspect: record types and how
    many of each, or CSV headers, plus a few raw lines. Nothing here is trusted or
    stored -- it exists so the layout can be confirmed before any number is believed.
    """
    text = _read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out = {"path": path, "lines": len(lines), "samples": lines[:sample]}
    if not lines:
        out["format"] = "empty"
        return out

    # MCX circular exports carry title rows above the header, so line 1 is not the
    # header and often has no delimiters at all. Find the first line that does.
    delim = _delimiter(text)
    hdr_i = next((i for i, ln in enumerate(lines[:25]) if ln.count(delim) >= 3), None)
    if hdr_i is not None and not _looks_positional(lines):
        rdr = csv.reader(io.StringIO("\n".join(lines[hdr_i:])), delimiter=delim)
        header = next(rdr, [])
        out["format"] = "csv"
        out["delimiter"] = "tab" if delim == "\t" else "comma"
        out["header_line"] = hdr_i
        out["title_rows"] = lines[:hdr_i]
        out["header"] = [h.strip() for h in header]
        out["rows"] = sum(1 for _ in rdr)
        return out

    counts: dict = {}
    for ln in lines:
        counts[ln[:1]] = counts.get(ln[:1], 0) + 1
    out["format"] = "positional / record-typed"
    out["record_types"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    out["by_type"] = {t: next(ln for ln in lines if ln[:1] == t)
                      for t in list(out["record_types"])[:6]}
    return out


def _delimiter(text: str) -> str:
    return "\t" if text.count("\t") > text.count(",") else ","


def _looks_positional(lines) -> bool:
    """A SPAN file's records start with a single-character type code."""
    typed = sum(1 for ln in lines[:50] if ln[:1].isalnum() and ln[1:2] in " 0123456789")
    return typed >= max(5, len(lines[:50]) // 2)


def _find_date_in_text(text: str) -> date | None:
    """
    A business date out of a title row.

    MCX circular exports state it as free text ("Applicable Margins as on
    18-09-2026") rather than as a column, and the date is the one thing a parameter
    file may not be loaded without.
    """
    for m in re.finditer(r"\d{1,4}[-/ ][A-Za-z0-9]{2,3}[-/ ]\d{2,4}", text):
        d = _parse_date(m.group(0))
        if d:
            return d
    return None


# ---------------------------------------------------------------------------
# the MCXCCL CSV publications
# ---------------------------------------------------------------------------

# Header synonyms. MCX renames columns between circulars, so matching is by
# normalised substring rather than exact equality -- but an unmatched REQUIRED column
# is an error, never a default.
_ALIASES = {
    "commodity": ("commodity", "symbol", "product", "instrument", "scrip"),
    "business_date": ("date", "business date", "trade date", "as on", "asondate"),
    "scan_pct": ("price scan range", "scan range", "pricescanrange", "span %",
                 "span margin %", "initial margin %", "im %", "margin %"),
    "elm_pct": ("extreme loss", "elm", "elm %"),
    "additional_pct": ("additional margin", "adhoc margin", "ad-hoc margin",
                       "special margin"),
    "delivery_pct": ("delivery margin", "delivery period margin", "tender margin"),
    "spot_charge": ("spot month", "near month charge", "tender period charge"),
    "intra_charge": ("intra commodity", "calendar spread", "spread charge"),
    "leg_a": ("commodity 1", "leg 1", "commodity a", "first commodity", "symbol 1"),
    "leg_b": ("commodity 2", "leg 2", "commodity b", "second commodity", "symbol 2"),
    "credit_rate": ("spread credit", "credit rate", "benefit", "cross margin benefit",
                    "credit %", "benefit %"),
    "ratio_a": ("ratio 1", "ratio a"),
    "ratio_b": ("ratio 2", "ratio b"),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).strip().lower()).strip()


def _match_columns(header) -> dict:
    """
    Map our field names onto this file's columns.

    Every (field, alias) pair competes at once, longest alias first, because
    specificity has to beat field order. A cross-margin file's "Commodity 1" column
    matches both the generic `commodity` alias and the specific `leg_a` one; letting
    the generic alias claim it first -- as iterating field by field does -- silently
    turns a two-leg spread definition into a one-commodity margin row and loses the
    pairing entirely.
    """
    normed = [(i, _norm(h)) for i, h in enumerate(header)]
    candidates = sorted(
        ((alias, fieldname) for fieldname, aliases in _ALIASES.items()
         for alias in aliases),
        key=lambda av: (-len(av[0]), av[0]))

    found: dict = {}
    taken: set = set()
    for alias, fieldname in candidates:
        if fieldname in found:
            continue
        for i, h in normed:
            if i in taken:
                continue
            if alias in h:
                found[fieldname] = i
                taken.add(i)
                break
    return found


def _is_percent_header(header_cell):
    """
    What this column's NAME declares about its units.

    True for a declared percentage, False for a declared fraction, and None when the
    header says nothing -- which is a genuinely different state from "not a
    percentage" and must stay distinguishable, because only the unknown case may fall
    back to guessing from magnitude.
    """
    h = _norm(header_cell) + str(header_cell)
    if "%" in str(header_cell) or "pct" in h or "percent" in h:
        return True
    if "fraction" in h or "ratio of" in h:
        return False
    return None


def _pct(raw, percent_column: bool | None = None) -> float | None:
    """
    A rate column, returned as a fraction.

    The units have to come from the column HEADER, not from the value. A cell reading
    "0.50" in a column headed "Additional Margin (%)" is 0.50 %, i.e. 0.005 -- while
    the same cell in an unlabelled column is most likely 50 %. Guessing from
    magnitude alone gets that wrong by a factor of a hundred, silently, on exactly
    the small ad-hoc margins that are easiest to overlook.

    So: when the header declares a percentage, always divide by 100. Only when it
    declares nothing does magnitude decide, and that fallback is documented rather
    than hidden. Anything landing outside 0..1 raises instead of being coerced.
    """
    if raw is None:
        return None
    txt = str(raw).strip().replace("%", "").replace(",", "")
    if not txt or txt.lower() in ("na", "n/a", "-", "nil"):
        return None
    try:
        v = float(txt)
    except ValueError:
        raise SpanFileError("not a numeric rate: {!r}".format(raw))

    if percent_column:
        frac = v / 100.0
    elif percent_column is False:
        frac = v
    else:
        frac = v / 100.0 if v > 1.0 else v

    if not (0.0 <= frac <= 1.0):
        raise SpanFileError(
            "rate {!r} reads as {:.4f}, outside 0..1 -- the column is probably not a "
            "rate, or the units are not what this parser assumes".format(raw, frac))
    return frac


def _parse_date(raw) -> date | None:
    txt = str(raw).strip()
    if not txt:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y",
                "%Y%m%d", "%m/%d/%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(txt, fmt).date()
        except ValueError:
            continue
    return None


def parse_margin_csv(path: str, business_date: date | None = None,
                     kind: str = "risk_collateral") -> RiskFile:
    """
    Read an MCXCCL Risk-and-Collaterals style margin rate file (CSV or TSV).

    One row per commodity, columns matched by name. Rows whose commodity is not a
    symbol this project models are kept in `notes` rather than dropped silently: a
    file that turns out to contain no gold at all should say so loudly.
    """
    text = _read_text(path)
    dialect_delim = "\t" if text.count("\t") > text.count(",") else ","
    rdr = csv.reader(io.StringIO(text), delimiter=dialect_delim)
    rows = [r for r in rdr if any(str(c).strip() for c in r)]
    if not rows:
        raise SpanFileError("{}: no rows".format(path))

    # The header is not always the first line -- circular files carry title rows.
    hdr_i, cols = None, {}
    for i, r in enumerate(rows[:20]):
        got = _match_columns(r)
        if "commodity" in got and len(got) >= 2:
            hdr_i, cols = i, got
            break
    if hdr_i is None:
        raise SpanFileError(
            "{}: no header row matched. Columns seen: {!r}. Run sniff() and extend "
            "_ALIASES with this file's names rather than guessing positions."
            .format(path, rows[0][:12]))

    bdate = business_date
    if bdate is None and "business_date" in cols:
        # EVERY date in the column, not the first one. A file holding several
        # business dates was silently collapsing to the first date while the
        # commodity dict kept the LAST row -- so the newest rates were stored under
        # the oldest date, which is the exact look-ahead this module exists to stop.
        di = cols["business_date"]
        found = {d for d in (_parse_date(r[di]) for r in rows[hdr_i + 1:]
                             if len(r) > di) if d}
        if len(found) > 1:
            raise SpanFileError(
                "{}: holds {} business dates ({} .. {}). One RiskFile describes one "
                "date, so this is refused rather than collapsed. Use "
                "parse_margin_history() to get one file per date, or pass "
                "business_date= to select one."
                .format(path, len(found), min(found), max(found)))
        bdate = next(iter(found), None)
    if bdate is None:
        bdate = _find_date_in_text(" ".join(" ".join(r) for r in rows[:hdr_i]))
    if bdate is None:
        bdate = _date_from_name(path)
    if bdate is None:
        raise SpanFileError(
            "{}: no business date in the file or its name. A margin parameter "
            "without a date cannot be applied to a session without risking "
            "look-ahead, so it is refused.".format(path))

    out = RiskFile(business_date=bdate, exchange="MCX", source_path=path, kind=kind)
    known = set(specs.ALL_SYMBOLS)
    seen_other = set()

    header_row = rows[hdr_i]
    pct_col = {name: _is_percent_header(header_row[i])
               for name, i in cols.items() if i < len(header_row)}

    def cell(r, name):
        i = cols.get(name)
        return r[i] if i is not None and i < len(r) else None

    def rate(r, name):
        return _pct(cell(r, name), pct_col.get(name))

    if "scan_pct" not in cols:
        raise SpanFileError(
            "{}: matched a header but no price-scan / margin-rate column among {!r}. "
            "Without it there is nothing to scan with, and the ELM and delivery "
            "columns that ARE present would be stored against no scan range. Extend "
            "_ALIASES['scan_pct'] with this file's column name."
            .format(path, [h for h in header_row][:12]))

    di = cols.get("business_date")
    for r in rows[hdr_i + 1:]:
        # When a date column exists, only rows for the selected date belong to this
        # RiskFile -- otherwise another date's rates end up under this one's label.
        if di is not None and len(r) > di:
            rd = _parse_date(r[di])
            if rd is not None and rd != bdate:
                continue
        code = str(cell(r, "commodity") or "").strip().upper()
        if not code:
            continue
        if code not in known:
            seen_other.add(code)
            continue
        scan = rate(r, "scan_pct")
        if scan is None:
            continue
        out.commodities[code] = CommodityRisk(
            commodity=code, business_date=bdate,
            price_scan_range=scan, scan_basis="pct",
            intra_spread_charge=rate(r, "intra_charge") or 0.0,
            spot_month_charge=rate(r, "spot_charge") or 0.0,
            elm_pct=rate(r, "elm_pct"),
            additional_pct=rate(r, "additional_pct"),
            delivery_pct=rate(r, "delivery_pct"))

    if not out.commodities:
        raise SpanFileError(
            "{}: parsed cleanly but contained none of {}. Other commodities present: "
            "{}. This is the right file format and the wrong file."
            .format(path, sorted(known), sorted(seen_other)[:10] or "none"))
    if seen_other:
        out.notes.append("{} other commodities in the file, not loaded"
                         .format(len(seen_other)))
    return out


def parse_margin_history(path: str) -> list:
    """
    One RiskFile per business date, for a margin file covering several days.

    A parameter set belongs to exactly one date. Splitting rather than collapsing is
    what lets a backtest look up the regime that was actually in force on the session
    it is pricing.
    """
    text = _read_text(path)
    delim = _delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)
            if any(str(c).strip() for c in r)]
    hdr_i, cols = None, {}
    for i, r in enumerate(rows[:20]):
        got = _match_columns(r)
        if "commodity" in got and len(got) >= 2:
            hdr_i, cols = i, got
            break
    if hdr_i is None or "business_date" not in cols:
        return [parse_margin_csv(path)]
    di = cols["business_date"]
    dates = sorted({d for d in (_parse_date(r[di]) for r in rows[hdr_i + 1:]
                                if len(r) > di) if d})
    return [parse_margin_csv(path, business_date=d) for d in dates]


def parse_cross_margin_csv(path: str,
                           business_date: date | None = None) -> RiskFile:
    """
    Read an MCXCCL Cross-Margin Benefit file: which pairs net, and by how much.

    For this strategy it is the single most consequential published number. Every
    claim about whether a gold spread is capital efficient rests on the credit rate,
    and until it is read from here the model is sweeping an assumption.
    """
    text = _read_text(path)
    delim = "\t" if text.count("\t") > text.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)
            if any(str(c).strip() for c in r)]
    if not rows:
        raise SpanFileError("{}: no rows".format(path))

    hdr_i, cols = None, {}
    for i, r in enumerate(rows[:20]):
        got = _match_columns(r)
        if "leg_a" in got and "leg_b" in got and "credit_rate" in got:
            hdr_i, cols = i, got
            break
    if hdr_i is None:
        raise SpanFileError(
            "{}: no cross-margin header matched (need two commodity legs and a "
            "credit rate). Columns seen: {!r}".format(path, rows[0][:12]))

    bdate = business_date or _date_from_name(path)
    if bdate is None and "business_date" in cols:
        for r in rows[hdr_i + 1:]:
            if len(r) > cols["business_date"]:
                bdate = _parse_date(r[cols["business_date"]])
                if bdate:
                    break
    if bdate is None:
        bdate = _find_date_in_text(" ".join(" ".join(r) for r in rows[:hdr_i]))
    if bdate is None:
        raise SpanFileError("{}: no business date in the file or its name".format(path))

    out = RiskFile(business_date=bdate, exchange="MCX", source_path=path,
                   kind="cross_margin")
    known = set(specs.ALL_SYMBOLS)

    def cell(r, name, default=None):
        i = cols.get(name)
        return r[i] if i is not None and i < len(r) else default

    for r in rows[hdr_i + 1:]:
        a = str(cell(r, "leg_a") or "").strip().upper()
        b = str(cell(r, "leg_b") or "").strip().upper()
        if not a or not b:
            continue
        if a not in known or b not in known:
            continue
        rate = _pct(cell(r, "credit_rate"),
                    _is_percent_header(rows[hdr_i][cols["credit_rate"]])
                    if cols["credit_rate"] < len(rows[hdr_i]) else None)
        if rate is None:
            continue
        out.credits.append(SpreadCredit(
            business_date=bdate, leg_a=a, leg_b=b, credit_rate=rate,
            ratio_a=_int(cell(r, "ratio_a"), 1), ratio_b=_int(cell(r, "ratio_b"), 1)))

    if not out.credits:
        raise SpanFileError(
            "{}: no spread recognised between {}. If MCX grants no cross-margin "
            "benefit across these contracts that is a FINDING, not a parse failure -- "
            "record it explicitly rather than loading an empty file."
            .format(path, sorted(known)))
    return out


def _int(raw, default: int) -> int:
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default


def _date_from_name(path: str) -> date | None:
    """MCX names its daily files with the business date; use it as a fallback."""
    base = os.path.basename(path)
    for pat, fmt in ((r"(20\d{2}[-_]?\d{2}[-_]?\d{2})", "%Y%m%d"),
                     (r"(\d{2}[-_]?\d{2}[-_]?20\d{2})", "%d%m%Y")):
        m = re.search(pat, base)
        if m:
            try:
                return datetime.strptime(re.sub(r"[-_]", "", m.group(1)), fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# the PC-SPAN risk parameter file
# ---------------------------------------------------------------------------

# Record types in the SPAN layout that carry what this model needs. The exact field
# positions differ between SPAN versions and between exchanges' profiles, so they are
# declared here in one place rather than scattered through the parser -- and the
# parser VERIFIES what it extracted instead of assuming the declaration is right.
# A price scan range is a severe one-day move, not a doubling. Anything above this
# is not a rate at all -- it is a tier number, a contract size or a count that landed
# in the field this parser was looking at.
MAX_PLAUSIBLE_SCAN = 0.50

SPAN_RECORDS = {
    "B": "exchange / business date header",
    "C": "combined commodity definition",
    "P": "price scan range and charges",
    "S": "inter-commodity spread definition",
    "T": "tier definition",
    "D": "delta",
}


def _plausible_rate(nums, scan, nth: int) -> float:
    """
    The nth rate-looking field AFTER the one taken as the scan range.

    Positional reads are the weak point of this parser, so a field is only accepted
    when it is in the range a rate can occupy. Anything else contributes nothing
    rather than a confident wrong number.
    """
    after = []
    seen_scan = False
    for n in nums:
        if not seen_scan and n == scan:
            seen_scan = True
            continue
        if seen_scan and 0.0 <= n <= MAX_PLAUSIBLE_SCAN:
            after.append(n)
    return after[nth - 1] if len(after) >= nth else 0.0


def parse_span_risk_file(path: str, strict: bool = True) -> RiskFile:
    """
    Read the daily PC-SPAN risk parameter file.

    The SPAN layout is positional and versioned, and the field offsets in a given
    exchange's profile are not derivable from the data. Rather than hard-code offsets
    that may be wrong for this file, the parser splits each record on whitespace,
    identifies the commodity code against the symbols this project models, and takes
    the numeric fields that validate as rates or currency amounts -- then checks the
    result for plausibility.

    With `strict` (the default) anything that fails that check raises, naming the
    record. An MCX file whose profile differs will therefore fail loudly on the first
    run and can be pinned with `sniff()`, instead of quietly producing margins that
    are wrong by a factor.
    """
    text = _read_text(path)
    lines = [ln.rstrip("\n\r") for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SpanFileError("{}: no records".format(path))

    bdate, exch = None, "MCX"
    commodities: dict = {}
    credits: list = []
    unrecognised: dict = {}
    notes_out: list = []
    known = set(specs.ALL_SYMBOLS)

    for ln in lines:
        rt = ln[:1].upper()
        fields = ln[1:].split()
        if rt == "B":
            for f in fields:
                d = _parse_date(f)
                if d:
                    bdate = d
                    break
            if fields and fields[0].isalpha():
                exch = fields[0].upper()
            continue
        if rt not in SPAN_RECORDS:
            unrecognised[rt] = unrecognised.get(rt, 0) + 1
            continue

        codes = [f.upper() for f in fields if f.upper() in known]
        nums = []
        for f in fields:
            try:
                nums.append(float(f.replace(",", "")))
            except ValueError:
                continue

        if rt in ("C", "P") and len(codes) == 1 and nums:
            code = codes[0]
            # Take the first field that is PLAUSIBLE as a scan range, not simply the
            # first numeric one. SPAN records carry tier numbers, contract sizes and
            # counts before the rates, and blindly taking nums[0] reads "1" as a
            # 100 % scan range -- a number that is absurd, produces absurd margins,
            # and raises nothing. A fractional scan range above MAX_PLAUSIBLE_SCAN is
            # not a scan range this parser will believe.
            scan = next((n for n in nums if 0.0 < n <= MAX_PLAUSIBLE_SCAN), None)
            if scan is None:
                msg = ("{}: record for {} has no field that reads as a scan range "
                       "(0 < x <= {:.0%}): {!r}. The field layout of this SPAN "
                       "profile is not what this parser assumes -- run sniff() and "
                       "pin it before trusting any number from this file."
                       .format(path, code, MAX_PLAUSIBLE_SCAN, ln[:120]))
                if strict:
                    raise SpanFileError(msg)
                notes_out.append(msg)
                continue
            basis = "pct"
            prev = commodities.get(code)
            commodities[code] = CommodityRisk(
                commodity=code, business_date=bdate or date.min,
                price_scan_range=scan, scan_basis=basis,
                intra_spread_charge=_plausible_rate(nums, scan, 1) or (
                    prev.intra_spread_charge if prev else 0.0),
                spot_month_charge=_plausible_rate(nums, scan, 2) or (
                    prev.spot_month_charge if prev else 0.0))
        elif rt == "S" and len(codes) >= 2 and nums:
            rate = next((n for n in nums if 0.0 < n <= 100.0), None)
            if rate is None:
                if strict:
                    raise SpanFileError(
                        "{}: spread record names {} but no field reads as a credit "
                        "rate: {!r}".format(path, codes[:2], ln[:120]))
                continue
            credits.append(SpreadCredit(
                business_date=bdate or date.min, leg_a=codes[0], leg_b=codes[1],
                credit_rate=rate / 100.0 if rate > 1.0 else rate))

    if bdate is None:
        bdate = _date_from_name(path)
    if bdate is None:
        raise SpanFileError(
            "{}: no business date found. A SPAN file without its date cannot be "
            "applied to a session without risking look-ahead.".format(path))

    out = RiskFile(business_date=bdate, exchange=exch, source_path=path, kind="span",
                   commodities={k: replace(v, business_date=bdate)
                                for k, v in commodities.items()},
                   credits=[replace(c, business_date=bdate) for c in credits],
                   unrecognised=unrecognised, notes=list(notes_out))

    if not out.commodities:
        msg = ("{}: parsed {} records but recognised no scan range for any of {}. "
               "Run spanfile.sniff() on it and check the record layout before "
               "trusting anything from this file."
               .format(path, len(lines), sorted(known)))
        if strict:
            raise SpanFileError(msg)
        out.notes.append(msg)
    return out


def load(path: str, **kw) -> RiskFile:
    """Dispatch on what the file turns out to be, rather than on its extension."""
    info = sniff(path)
    if info.get("format") == "csv":
        header = " ".join(_norm(h) for h in info.get("header", []))
        if any(a in header for a in _ALIASES["credit_rate"]) and \
                any(a in header for a in _ALIASES["leg_a"]):
            return parse_cross_margin_csv(path, **kw)
        return parse_margin_csv(path, **kw)
    return parse_span_risk_file(path, **kw)


def load_dir(directory: str) -> list:
    """Every readable parameter file in a directory, oldest business date first."""
    out = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() not in (
                ".csv", ".txt", ".tsv", ".spn", ".zip", ""):
            continue
        out.append(load(full))
    return sorted(out, key=lambda r: r.business_date)


# ---------------------------------------------------------------------------
# turning published files into model parameters
# ---------------------------------------------------------------------------

def to_span_parameters(files, base: margin_mod.SpanParameters | None = None,
                       ) -> margin_mod.SpanParameters:
    """
    Fold one or more published files into a `SpanParameters`.

    Published values REPLACE the corresponding assumption; anything no file supplies
    keeps its default, and `source` names every file that contributed so a report can
    state exactly which numbers are published and which are still assumed. That
    distinction is the whole point: a parameter set that cannot say where each number
    came from is indistinguishable from a guess.
    """
    if isinstance(files, RiskFile):
        files = [files]
    files = list(files)
    if not files:
        raise SpanFileError("no files to load")

    base = base or margin_mod.SpanParameters()
    scans = dict(base.price_scan_range_pct)
    elms = dict(base.elm_pct_by_symbol)
    adds = dict(base.additional_pct_by_symbol)
    pair_credits: dict = {}
    kw: dict = {}
    published: set = set()

    for f in sorted(files, key=lambda r: r.business_date):
        for code, c in f.commodities.items():
            scans[code] = c.scan_pct()
            published.add("price_scan_range_pct")
            # Per commodity, because these contracts carry different rates and
            # collapsing them to a scalar applies one leg's margin to the other.
            if c.elm_pct is not None:
                elms[code] = c.elm_pct
                published.add("elm_pct")
            if c.additional_pct is not None:
                adds[code] = c.additional_pct
                published.add("additional_margin_pct")
            if c.delivery_pct is not None:
                kw["delivery_margin_floor_pct"] = c.delivery_pct
                published.add("delivery_margin_floor_pct")
            if c.intra_spread_charge:
                kw["intra_spread_charge_pct"] = c.intra_spread_charge
                published.add("intra_spread_charge_pct")
        for cr in f.credits:
            pair_credits[cr.key()] = cr.credit_rate
            published.add("inter_spread_credit")

    if elms:
        kw["elm_pct_by_symbol"] = elms
    if adds:
        kw["additional_pct_by_symbol"] = adds

    if pair_credits:
        kw["inter_spread_credit_pairs"] = pair_credits
        # The scalar stays as the fallback for pairs the file does not name, and is
        # set to the published minimum rather than the mean: an unnamed pair is more
        # likely to be unrecognised than generously credited.
        kw["inter_spread_credit"] = min(pair_credits.values())

    dates = sorted({f.business_date for f in files})
    # Name what was published AND what was not. A parameter set that cannot say
    # which of its numbers came from the exchange is indistinguishable from a guess,
    # and the fields below genuinely are not in these files -- they come from the
    # contract specification and from circulars.
    still_assumed = [f for f in ("im_floor_pct", "delivery_var_days",
                                 "delivery_var_z", "concentration_surcharge_pct",
                                 "recognised_groups")
                     if f not in published]
    kw["source"] = "MCX published {} from {} ({} .. {}); still assumed: {}".format(
        ", ".join(sorted(published)) or "nothing",
        ", ".join(os.path.basename(f.source_path) for f in files[:3])
        + ("..." if len(files) > 3 else ""),
        dates[0], dates[-1], ", ".join(still_assumed) or "nothing")
    return replace(base, price_scan_range_pct=scans, **kw)


def schedule_from_files(files) -> dict:
    """
    A `{(date, symbol): scan_pct}` schedule from dated published files.

    Shaped like `margin.volatility_schedule` so it drops into the backtest in its
    place -- but every value is published rather than estimated, and it covers only
    the dates whose files are held. Sessions outside that coverage keep the estimated
    schedule and remain flagged as estimated, because pretending a published value
    existed on a date whose file is missing is the look-ahead this module exists to
    prevent.
    """
    out: dict = {}
    for f in sorted(files, key=lambda r: r.business_date):
        for code, c in f.commodities.items():
            try:
                out[(f.business_date, code)] = c.scan_pct()
            except SpanFileError:
                continue
    return out


def coverage(files, sessions) -> dict:
    """How much of a backtest's sessions the published files actually cover."""
    have = {f.business_date for f in files}
    sess = sorted(set(sessions))
    covered = [d for d in sess if d in have]
    return {
        "sessions": len(sess),
        "covered": len(covered),
        "coverage": len(covered) / len(sess) if sess else 0.0,
        "first_file": min(have) if have else None,
        "last_file": max(have) if have else None,
        "gaps": len(sess) - len(covered),
    }
