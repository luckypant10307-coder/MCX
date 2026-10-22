"""
Loading PUBLISHED MCX risk parameters.

The fixtures here are hand-built in the shape of the real publications, not copies of
them. That is enough to pin the behaviour that matters, which is not "can it parse a
happy file" but:

  * does it REFUSE a file it cannot read with confidence, rather than defaulting;
  * does it keep published and assumed values distinguishable;
  * does it treat an unrecognised spread as earning ZERO rather than the assumed rate;
  * does it carry the business date, so a parameter cannot be applied to a session it
    did not exist for.

The last one is a look-ahead control. A SPAN file describes the risk regime as of its
date, and back-filling history from the newest file flatters the result exactly when
margin was rising.
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from mcxrv import margin, spanfile
from mcxrv.spanfile import SpanFileError

# --------------------------------------------------------------------- fixtures

MARGIN_CSV = """MULTI COMMODITY EXCHANGE CLEARING CORPORATION
Applicable Margins as on 18-09-2026
Commodity,Price Scan Range (%),Extreme Loss Margin (%),Additional Margin (%),Delivery Period Margin (%)
GOLDM,5.50,1.25,0.00,3.00
GOLDTEN,5.50,1.25,0.00,3.00
GOLDGUINEA,6.10,1.50,0.00,3.00
GOLDPETAL,6.40,1.50,0.50,3.00
SILVERM,7.20,1.50,0.00,3.00
"""

CROSS_CSV = """MCXCCL Cross Margin Benefit
Effective 18-09-2026
Commodity 1,Commodity 2,Ratio 1,Ratio 2,Spread Credit (%)
GOLDM,GOLDTEN,1,10,75
GOLDM,GOLDGUINEA,1,12,60
GOLDTEN,GOLDGUINEA,1,1,60
"""

SPAN_TXT = """B MCX 18-09-2026 INR
C GOLDM 0.0550 0.0050 0.0300
C GOLDTEN 0.0550 0.0050 0.0300
C GOLDGUINEA 0.0610 0.0050 0.0300
C GOLDPETAL 0.0640 0.0050 0.0300
S GOLDM GOLDTEN 75
S GOLDM GOLDGUINEA 60
X something the parser does not know about
"""


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ the readers

def test_margin_csv_is_read_with_its_business_date(tmp_path):
    f = spanfile.parse_margin_csv(_write(tmp_path, "margins.csv", MARGIN_CSV))
    assert f.business_date == date(2026, 9, 18)
    assert set(f.commodities) == {"GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"}
    assert f.commodities["GOLDM"].scan_pct() == pytest.approx(0.055)
    assert f.commodities["GOLDPETAL"].elm_pct == pytest.approx(0.015)
    assert f.commodities["GOLDPETAL"].additional_pct == pytest.approx(0.005)
    # SILVERM is in the file and not in this model: noted, never silently dropped
    assert any("other commodities" in n for n in f.notes)


def test_the_column_header_decides_the_units_not_the_magnitude(tmp_path):
    """
    A cell reading "0.50" in a column headed "(%)" is 0.50 %, not 50 %. Inferring
    units from magnitude gets that wrong by a factor of a hundred, silently, on
    exactly the small ad-hoc margins that are easiest to overlook -- so a header that
    declares its units is authoritative and the magnitude is not consulted.
    """
    body = MARGIN_CSV.replace("GOLDM,5.50", "GOLDM,0.055")
    f = spanfile.parse_margin_csv(_write(tmp_path, "m2.csv", body))
    assert f.commodities["GOLDM"].scan_pct() == pytest.approx(0.00055)
    # and the small additional margin keeps its true size
    assert f.commodities["GOLDPETAL"].additional_pct == pytest.approx(0.005)


def test_magnitude_decides_only_when_the_header_declares_nothing(tmp_path):
    body = MARGIN_CSV.replace("Price Scan Range (%)", "Price Scan Range")
    f = spanfile.parse_margin_csv(_write(tmp_path, "m3.csv", body))
    assert f.commodities["GOLDM"].scan_pct() == pytest.approx(0.055)

    frac = body.replace("GOLDM,5.50", "GOLDM,0.055")
    f2 = spanfile.parse_margin_csv(_write(tmp_path, "m4.csv", frac))
    assert f2.commodities["GOLDM"].scan_pct() == pytest.approx(0.055)


def test_an_implausible_rate_raises_rather_than_being_coerced(tmp_path):
    body = MARGIN_CSV.replace("GOLDM,5.50", "GOLDM,550")
    with pytest.raises(SpanFileError, match="outside 0..1"):
        spanfile.parse_margin_csv(_write(tmp_path, "bad.csv", body))


def test_cross_margin_file_yields_per_pair_rates(tmp_path):
    f = spanfile.parse_cross_margin_csv(_write(tmp_path, "cross.csv", CROSS_CSV))
    assert f.business_date == date(2026, 9, 18)
    rates = {c.key(): c.credit_rate for c in f.credits}
    assert rates[("GOLDM", "GOLDTEN")] == pytest.approx(0.75)
    assert rates[("GOLDGUINEA", "GOLDM")] == pytest.approx(0.60)
    assert ("GOLDM", "GOLDPETAL") not in rates


def test_span_text_file_reads_scans_and_credits(tmp_path):
    f = spanfile.parse_span_risk_file(_write(tmp_path, "span_20260918.txt", SPAN_TXT))
    assert f.business_date == date(2026, 9, 18)
    assert f.commodities["GOLDGUINEA"].scan_pct() == pytest.approx(0.061)
    assert f.unrecognised.get("X") == 1, "unknown record types are counted, not hidden"
    rates = {c.key(): c.credit_rate for c in f.credits}
    assert rates[("GOLDM", "GOLDTEN")] == pytest.approx(0.75)


def test_load_dispatches_on_content_not_extension(tmp_path):
    m = spanfile.load(_write(tmp_path, "whatever.csv", MARGIN_CSV))
    c = spanfile.load(_write(tmp_path, "other.csv", CROSS_CSV))
    assert m.kind == "risk_collateral"
    assert c.kind == "cross_margin"


# ------------------------------------------------------------------- refusals

def test_a_file_with_no_date_is_refused(tmp_path):
    """
    A parameter with no date cannot be applied to a session without risking
    look-ahead, so it is refused rather than dated to today.
    """
    body = MARGIN_CSV.replace("Applicable Margins as on 18-09-2026", "Applicable Margins")
    with pytest.raises(SpanFileError, match="business date"):
        spanfile.parse_margin_csv(_write(tmp_path, "nodate.csv", body))


def test_the_right_format_and_the_wrong_file_is_refused(tmp_path):
    body = MARGIN_CSV.replace("GOLD", "COPPER").replace("COPPERM", "COPPER")
    with pytest.raises(SpanFileError, match="right file format and the wrong file"):
        spanfile.parse_margin_csv(_write(tmp_path, "copper.csv", body))


def test_unmatched_header_names_the_columns_it_saw(tmp_path):
    body = "alpha,beta,gamma\n1,2,3\n"
    with pytest.raises(SpanFileError, match="no header row matched"):
        spanfile.parse_margin_csv(_write(tmp_path, "odd.csv", body))


def test_a_span_file_that_yields_nothing_raises_in_strict_mode(tmp_path):
    body = "B MCX 18-09-2026 INR\nC SILVERM 0.0720 0.0050\n"
    path = _write(tmp_path, "silver_20260918.txt", body)
    with pytest.raises(SpanFileError, match="recognised no scan range"):
        spanfile.parse_span_risk_file(path)
    lenient = spanfile.parse_span_risk_file(path, strict=False)
    assert lenient.notes, "lenient mode must still say what it could not read"


def test_currency_scan_without_contract_value_refuses_to_invent_a_percentage():
    c = spanfile.CommodityRisk(
        commodity="GOLDM", business_date=date(2026, 9, 18),
        price_scan_range=42000.0, scan_basis="inr_per_unit")
    with pytest.raises(SpanFileError, match="cannot be expressed as a percentage"):
        c.scan_pct()
    ok = spanfile.CommodityRisk(
        commodity="GOLDM", business_date=date(2026, 9, 18),
        price_scan_range=42000.0, scan_basis="inr_per_unit",
        contract_value=700_000.0)
    assert ok.scan_pct() == pytest.approx(0.06)


# ------------------------------------------------------- into the margin model

def test_published_values_replace_assumptions_and_say_so(tmp_path):
    files = [spanfile.parse_margin_csv(_write(tmp_path, "m.csv", MARGIN_CSV)),
             spanfile.parse_cross_margin_csv(_write(tmp_path, "c.csv", CROSS_CSV))]
    sp = spanfile.to_span_parameters(files)
    base = margin.SpanParameters()
    assert sp.price_scan_range_pct["GOLDGUINEA"] == pytest.approx(0.061)
    assert sp.price_scan_range_pct["GOLDGUINEA"] != base.price_scan_range_pct["GOLDGUINEA"]
    assert "MCX published" in sp.source
    assert "assumed defaults" not in sp.source


def test_an_unrecognised_pair_earns_zero_not_the_assumed_rate(tmp_path):
    """
    THE test for this module. Treating "MCX does not recognise this spread" as "MCX
    credits it at 70%" is the single most expensive mistake available here: it makes
    an uncredited two-leg position look like a capital-efficient spread.
    """
    f = spanfile.parse_cross_margin_csv(_write(tmp_path, "c.csv", CROSS_CSV))
    sp = spanfile.to_span_parameters([f])
    assert sp.credit_for("GOLDM", "GOLDTEN") == pytest.approx(0.75)
    assert sp.credit_for("GOLDM", "GOLDPETAL") == 0.0
    assert sp.credit_for("GOLDTEN", "GOLDPETAL") == 0.0


def test_scalar_credit_still_applies_when_no_file_names_pairs():
    sp = margin.SpanParameters()
    assert sp.credit_for("GOLDM", "GOLDPETAL") == pytest.approx(sp.inter_spread_credit)


def test_an_uncredited_pair_costs_more_margin(tmp_path):
    """The per-pair rate must actually reach the margin computation."""
    f = spanfile.parse_cross_margin_csv(_write(tmp_path, "c.csv", CROSS_CSV))
    published = spanfile.to_span_parameters([f], base=margin.SpanParameters(
        im_floor_basis="none"))
    assumed = margin.SpanParameters(im_floor_basis="none")

    credited = [margin.MarginPosition("GOLDM", "a", 1, 35485.0, 20),
                margin.MarginPosition("GOLDTEN", "b", -10, 35616.0, 20)]
    uncredited = [margin.MarginPosition("GOLDM", "a", 1, 35485.0, 20),
                  margin.MarginPosition("GOLDPETAL", "c", -99, 3596.5, 20)]

    assert (margin.span_margin(uncredited, published)["span"]
            > margin.span_margin(uncredited, assumed)["span"]), \
        "an unrecognised pair must cost MORE than the assumed blanket credit"
    assert (margin.span_margin(credited, published)["span"]
            < margin.span_margin(uncredited, published)["span"])


def test_schedule_is_dated_and_covers_only_what_was_published(tmp_path):
    f = spanfile.parse_margin_csv(_write(tmp_path, "m.csv", MARGIN_CSV))
    sched = spanfile.schedule_from_files([f])
    assert (date(2026, 9, 18), "GOLDM") in sched
    assert (date(2026, 9, 17), "GOLDM") not in sched, \
        "a session with no file must NOT inherit another day's parameters"

    cov = spanfile.coverage([f], [date(2026, 9, 17), date(2026, 9, 18)])
    assert cov["covered"] == 1 and cov["gaps"] == 1
    assert cov["coverage"] == pytest.approx(0.5)


def test_zip_with_several_snapshots_refuses_to_pick_one(tmp_path):
    import zipfile as zf
    p = tmp_path / "span.zip"
    with zf.ZipFile(p, "w") as z:
        z.writestr("span_0900.txt", SPAN_TXT)
        z.writestr("span_1500.txt", SPAN_TXT)
    with pytest.raises(SpanFileError, match="archive holds 2 members"):
        spanfile.sniff(str(p))


# ------------------------------------------------------------------- sniffing

def test_sniff_describes_without_interpreting(tmp_path):
    info = spanfile.sniff(_write(tmp_path, "m.csv", MARGIN_CSV))
    assert info["format"] == "csv"
    assert any("Commodity" in h for h in info["header"])

    info2 = spanfile.sniff(_write(tmp_path, "s.txt", SPAN_TXT))
    assert info2["format"].startswith("positional")
    assert info2["record_types"]["C"] == 4
    assert "S" in info2["record_types"]


def test_load_dir_is_ordered_by_business_date(tmp_path):
    d = tmp_path / "risk"
    d.mkdir()
    (d / "margins_18-09-2026.csv").write_text(MARGIN_CSV, encoding="utf-8")
    (d / "cross_18-09-2026.csv").write_text(CROSS_CSV, encoding="utf-8")
    files = spanfile.load_dir(str(d))
    assert len(files) == 2
    assert files[0].business_date <= files[1].business_date


# ---------------------------------------------------------------------------
# regressions: every one of these was a live defect, and every one was silent
# ---------------------------------------------------------------------------

def test_published_credits_survive_a_json_round_trip(tmp_path):
    """
    REGRESSION. `--json` writes "GOLDM|GOLDTEN" because JSON has no tuple keys, and
    `credit_for` looks up a tuple. Unnormalised, every lookup missed -- and because a
    populated mapping suppresses the scalar fallback, EVERY spread came back credited
    at zero. Published rates loaded and then discarded, with no error anywhere.
    """
    import json
    f = spanfile.parse_cross_margin_csv(_write(tmp_path, "c.csv", CROSS_CSV))
    sp = spanfile.to_span_parameters([f])
    as_json = {"{}|{}".format(*k): v for k, v in sp.inter_spread_credit_pairs.items()}
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"inter_spread_credit_pairs": as_json}), encoding="utf-8")

    loaded = margin.SpanParameters.from_mcx_file(str(path))
    assert loaded.credit_for("GOLDM", "GOLDTEN") == pytest.approx(0.75)
    assert loaded.credit_for("GOLDTEN", "GOLDM") == pytest.approx(0.75), "order-free"
    assert loaded.credit_for("GOLDM", "GOLDPETAL") == 0.0


def test_a_tier_field_is_not_read_as_a_scan_range(tmp_path):
    """
    REGRESSION. The SPAN parser took the first numeric field, so a record whose
    leading fields are a tier number and a contract size read "1" as a 100 % scan
    range -- an absurd margin, accepted silently.
    """
    body = "B MCX 18-09-2026 INR\nC GOLDM 1 100 0.0550 0.0050\n"
    f = spanfile.parse_span_risk_file(_write(tmp_path, "s.txt", body))
    assert f.commodities["GOLDM"].price_scan_range == pytest.approx(0.055)
    assert f.commodities["GOLDM"].scan_pct() <= spanfile.MAX_PLAUSIBLE_SCAN


def test_a_record_with_no_plausible_rate_is_refused(tmp_path):
    body = "B MCX 18-09-2026 INR\nC GOLDM 1 100 4200\n"
    with pytest.raises(SpanFileError, match="no field that reads as a scan range"):
        spanfile.parse_span_risk_file(_write(tmp_path, "s2.txt", body))


def test_a_multi_date_file_is_not_collapsed_to_one_date(tmp_path):
    """
    REGRESSION. A file holding several business dates took the FIRST date while the
    commodity dict kept the LAST row -- storing the newest rates under the oldest
    date. That is the look-ahead this module exists to prevent, produced by the
    module itself.
    """
    body = ("MCXCCL\nCommodity,Date,Price Scan Range (%),Extreme Loss Margin (%)\n"
            "GOLDM,01-09-2026,5.00,1.25\nGOLDM,18-09-2026,7.50,1.25\n")
    path = _write(tmp_path, "hist.csv", body)
    with pytest.raises(SpanFileError, match="business dates"):
        spanfile.parse_margin_csv(path)

    files = spanfile.parse_margin_history(path)
    assert [f.business_date for f in files] == [date(2026, 9, 1), date(2026, 9, 18)]
    assert files[0].commodities["GOLDM"].scan_pct() == pytest.approx(0.05)
    assert files[1].commodities["GOLDM"].scan_pct() == pytest.approx(0.075)


def test_a_file_with_no_scan_column_says_so(tmp_path):
    """
    REGRESSION. Rows were skipped when the scan column was absent, and the loader
    then blamed the file for being the wrong one -- while silently discarding the ELM
    and delivery columns that WERE present.
    """
    body = ("MCXCCL\nMargins as on 18-09-2026\n"
            "Commodity,Extreme Loss Margin (%),Delivery Period Margin (%)\n"
            "GOLDM,1.25,3.00\n")
    with pytest.raises(SpanFileError, match="no price-scan"):
        spanfile.parse_margin_csv(_write(tmp_path, "noscan.csv", body))


def test_an_uncovered_session_falls_back_to_estimated_not_to_published():
    """
    REGRESSION, and the most consequential one. `params_for_session` had no fallback,
    so a session with no published file kept whatever was in `base` -- which, when
    base was folded from published files, meant the NEWEST file's parameters applied
    retroactively to sessions that never saw them.
    """
    base = margin.SpanParameters(price_scan_range_pct={"GOLDM": 0.99})
    published = {(date(2026, 9, 18), "GOLDM"): 0.055}
    estimated = {(date(2026, 9, 17), "GOLDM"): 0.042,
                 (date(2026, 9, 18), "GOLDM"): 0.041}

    covered = margin.params_for_session(base, date(2026, 9, 18), published, estimated)
    assert covered.scan_pct("GOLDM") == pytest.approx(0.055), "published wins"
    assert "published" in covered.source

    gap = margin.params_for_session(base, date(2026, 9, 17), published, estimated)
    assert gap.scan_pct("GOLDM") == pytest.approx(0.042), \
        "an uncovered session must NOT inherit the published file"
    assert "ESTIMATED" in gap.source, "and the report must say it was estimated"
