"""Contract mechanics and normalisation. These guard the project's central error."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from mcxrv import normalize, specs


def test_quote_base_is_not_trading_unit():
    """GOLDM: 100 g trading unit, 10 g quotation base. Conflating them is the bug."""
    m = specs.get("GOLDM")
    assert m.trading_unit_g == 100.0
    assert m.quote_base_g == 10.0
    assert m.trading_unit_g != m.quote_base_g


def test_goldm_and_goldten_share_quote_base():
    """
    They differ ONLY in purity, so normalisation between them is a purity
    adjustment and nothing else. Dividing GOLDM by 100 g makes it look 10x cheap.
    """
    assert specs.get("GOLDM").quote_base_g == specs.get("GOLDTEN").quote_base_g


def test_purity_offset_is_40bp():
    """
    GOLDM (995) should quote ~40 bp BELOW a 999 contract mechanically, with no
    mispricing at all. Any spread model must remove this before saying anything.
    """
    ratio = specs.get("GOLDTEN").purity / specs.get("GOLDM").purity
    assert ratio == pytest.approx(1.00402, abs=1e-5)


@pytest.mark.parametrize("symbol,quote,expected", [
    ("GOLDM", 12000.0, 12000.0 * 10 / (10 * 0.995)),
    ("GOLDTEN", 12100.0, 12100.0 * 10 / (10 * 0.999)),
    ("GOLDGUINEA", 9700.0, 9700.0 * 10 / (8 * 0.999)),
    ("GOLDPETAL", 1215.0, 1215.0 * 10 / (1 * 0.999)),
])
def test_normalisation_formula(symbol, quote, expected):
    assert specs.get(symbol).to_inr_per_10g_fine(quote) == pytest.approx(expected)


def test_normalisation_roundtrip():
    for s in specs.ALL_SYMBOLS:
        sp = specs.get(s)
        q = 1234.5
        assert sp.from_inr_per_10g_fine(sp.to_inr_per_10g_fine(q)) == pytest.approx(q)


def test_equal_fine_gold_normalises_to_equal_price():
    """
    The real invariant: four contracts priced at the same fine-gold value must all
    normalise to the same number, whatever their size or purity.
    """
    target = 12345.0
    for s in specs.ALL_SYMBOLS:
        sp = specs.get(s)
        assert sp.to_inr_per_10g_fine(sp.from_inr_per_10g_fine(target)) == \
            pytest.approx(target)


def test_lot_notional_is_not_the_quote():
    """GOLDM at 12000 is a Rs 1.2 lakh lot. The 1.2 lakh is never in the bhavcopy."""
    assert specs.get("GOLDM").lot_notional(12000.0) == pytest.approx(120000.0)
    assert specs.get("GOLDPETAL").lot_notional(1215.0) == pytest.approx(1215.0)


def test_hedge_ratios_are_not_integers():
    """
    Lot granularity is a real constraint, not an inconvenience. GOLDM against
    GOLDGUINEA is 12.45:1 in fine grams, so whole lots cannot hedge exactly.
    """
    assert specs.hedge_ratio_fine("GOLDM", "GOLDGUINEA") == pytest.approx(12.45, abs=1e-3)
    assert specs.hedge_ratio_fine("GOLDM", "GOLDPETAL") == pytest.approx(99.5996, abs=1e-3)
    r = specs.hedge_ratio_fine("GOLDM", "GOLDGUINEA")
    assert abs(r - round(r)) > 0.01


def test_maturity_matching():
    """GOLDM never shares an expiry window with the 999 trio."""
    assert specs.is_maturity_matched("GOLDTEN", "GOLDPETAL")
    assert specs.is_maturity_matched("GOLDGUINEA", "GOLDPETAL")
    assert not specs.is_maturity_matched("GOLDM", "GOLDTEN")
    assert not specs.is_maturity_matched("GOLDM", "GOLDPETAL")


def test_symbol_lookup_tolerates_padding():
    assert specs.get("GOLDM   ").symbol == "GOLDM"
    assert specs.get("  goldpetal ").symbol == "GOLDPETAL"
    with pytest.raises(KeyError):
        specs.get("SILVERM")


def test_add_normalized_columns():
    df = pd.DataFrame({
        "date": [date(2025, 1, 2)] * 2,
        "symbol": ["GOLDM", "GOLDPETAL"],
        "close": [12000.0, 1215.0],
    })
    out = normalize.add_normalized(df)
    assert out.loc[0, "norm_close"] == pytest.approx(12000.0 * 10 / 9.95)
    assert out.loc[1, "norm_close"] == pytest.approx(1215.0 * 10 / 0.999)
    assert out.loc[0, "fine_g_per_lot"] == pytest.approx(99.5)
    assert np.isclose(out.loc[0, "log_norm"], np.log(out.loc[0, "norm_close"]))
