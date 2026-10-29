"""
Published margin rates (broker sheets).

What must hold: both page formats parse to the same numbers a person reads off them;
a delivery-period month never sets the rate; a sheet is used only from the session it
could have been known, and not forever after; the outright margin the engine computes
from it equals the published rate; and a re-read of the same day never alerts twice.
"""
from __future__ import annotations

from datetime import date

import pytest

from mcxrv.stage3_gates_decision import margin_capital as margin
from mcxrv import published_margins as pm

ZERODHA = """
<span class="modified">Last updated: 27 Oct 2026</span>
<tr data-scrip="GOLDM" data-id="31" id="entry-31"
    data-expiry="Nov 2026" data-lot_size="100" data-margin="9.25"
    data-nrml_margin="130841" data-mis_margin="65420" data-price="141451"><td>GOLDM</td></tr>
<tr data-scrip="GOLDM" data-id="32" id="entry-32"
    data-expiry="Dec 2026" data-lot_size="100" data-margin="9.25"
    data-nrml_margin="132251" data-mis_margin="66125" data-price="142971"><td>GOLDM</td></tr>
<tr data-scrip="SILVER" data-id="70" id="entry-70"
    data-expiry="Dec 2026" data-lot_size="30" data-margin="13.41"
    data-nrml_margin="787872" data-mis_margin="1" data-price="195793"><td>SILVER</td></tr>
"""

# Lines as pypdf extracts them from the 27 Feb 2026 sheet (trimmed).
ABML = """ADITYA BIRLA MONEY LTD      Date:
52 MCX GOLDGUINEA 27-Feb-26 8 GRMS 1 8 GRMS 15 25 25 0 0 0 0 1.25 1.25 0.03 0.45 66.25 66.25 84,570 84,570
53 MCX GOLDGUINEA 31-Mar-26 8 GRMS 1 8 GRMS 16 0 0 0 0 0 0 1.25 1.25 0.03 0.47 16.86 16.86 21,928 21,928
54 MCX GOLDGUINEA 30-Apr-26 8 GRMS 1 8 GRMS 16 0 0 0 0 0 0 1.25 1.25 0.03 0.49 17.39 17.39 22,967 22,967
58 MCX GOLDM 05-Mar-26 100 GRMS 10 10 GRMS 16 0 0 2 2 0 0 1.25 1.25 0.03 0.49 17.40 17.40 2,74,203 2,74,203
96 MCX MENTHAOIL 27-Feb-26 360 KGS 360 1 KGS 8 15 25 0 0 0 0 1.25 1.25 0.01 0.13 49.25 49.25 1,67,637 1,67,637
27-Feb-26
"""


def test_zerodha_page_parses_gold_only():
    s = pm.parse_zerodha(ZERODHA)
    assert s["date"] == date(2026, 10, 27)
    assert {c["symbol"] for c in s["contracts"]} == {"GOLDM"}
    assert pm.symbol_rates(s) == {"GOLDM": pytest.approx(0.0925)}


def test_abml_sheet_parses_breakdown_from_the_right():
    s = pm.parse_abml(ABML)
    assert s["date"] == date(2026, 2, 27)
    gm = next(c for c in s["contracts"] if c["symbol"] == "GOLDM")
    assert (gm["initial_pct"], gm["additional_pct"], gm["elm_pct"], gm["total_pct"]) == (16, 2, 1.25, 17.40)
    assert gm["margin_inr"] == 274203


def test_a_delivery_month_never_sets_the_rate():
    # The expiring GOLDGUINEA month carries 66.25%; the rate is the other months' median.
    rates = pm.symbol_rates(pm.parse_abml(ABML))
    assert rates["GOLDGUINEA"] == pytest.approx((0.1686 + 0.1739) / 2)


def test_unparseable_pages_raise():
    with pytest.raises(ValueError):
        pm.parse_zerodha("<html>maintenance</html>")
    with pytest.raises(ValueError):
        pm.parse_abml("nothing here")


def _store(*days):
    st = {}
    for d, r in days:
        pm.add(st, {"date": d, "contracts": [{"symbol": "GOLDM", "expiry": "x", "total_pct": r}]}, "zerodha")
    return st


def test_schedule_uses_a_sheet_from_when_it_was_known_and_not_forever():
    sp = margin.SpanParameters()
    st = _store((date(2026, 2, 3), 17.0))
    sessions = [date(2026, 1, 30), date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 27)]
    sch = pm.schedule(st, sessions, sp.elm_for, sp.additional_for)
    # 30 Jan: next session is 2 Feb, the sheet is dated 3 Feb -> not yet known.
    assert (date(2026, 1, 30), "GOLDM") not in sch
    # 2 Feb: its t+1 fill happens on 3 Feb, when the sheet is in force.
    assert sch[(date(2026, 2, 2), "GOLDM")] == pytest.approx(0.17 - 0.0125)
    # 27 Feb is 24 days later: too stale to call published.
    assert (date(2026, 2, 27), "GOLDM") not in sch


def test_outright_margin_equals_the_published_rate():
    sp = margin.SpanParameters()
    st = _store((date(2026, 10, 27), 9.25))
    sch = pm.schedule(st, [date(2026, 10, 26)], sp.elm_for, sp.additional_for)
    p = margin.params_for_session(sp, date(2026, 10, 26), sch)
    # One GOLDM lot (100 g, quoted per 10 g) at 1,00,000 = 10,00,000 of contract value.
    leg = margin.MarginPosition(symbol="GOLDM", contract_id="GOLDM-NOV26", lots=1, price=100_000.0)
    m = margin.span_margin([leg], p)
    assert m["initial_margin"] == pytest.approx(92_500.0, rel=1e-6)


def test_change_detection_and_message():
    st = _store((date(2026, 2, 26), 9.25), (date(2026, 2, 27), 17.5))
    ch = pm.latest_change(st)
    assert ch == [{"symbol": "GOLDM", "from_pct": 9.25, "to_pct": 17.5,
                   "from_date": "2026-02-26", "to_date": "2026-02-27"}]
    assert "9.25% -> 17.50%" in pm.message(ch)
    assert pm.latest_change(_store((date(2026, 2, 26), 9.25), (date(2026, 2, 27), 9.3))) == []


def test_a_breakdown_is_never_replaced_by_a_total_only_page():
    st = {}
    assert pm.add(st, pm.parse_abml(ABML), "abml")
    assert not pm.add(st, {"date": date(2026, 2, 27), "contracts": []}, "zerodha")
    assert st["2026-02-27"]["source"] == "abml"


def test_status_for_the_dashboard(tmp_path):
    st = _store((date(2026, 2, 26), 9.25), (date(2026, 2, 27), 17.5))
    p = tmp_path / "m.json"
    pm.save(st, str(p))
    out = pm.status(pm.load(str(p)), sessions=[])
    assert out["available"] and out["days"] == 2 and out["latest"] == {"GOLDM": 17.5}
    assert [h["GOLDM"] for h in out["history"]] == [9.25, 17.5]
    assert pm.status({})["available"] is False
