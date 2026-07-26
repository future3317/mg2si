import math

from mg2si.io.parsers import parse_numeric_range, parse_pvp_mw


def test_pvp_molecular_weight_units():
    assert parse_pvp_mw("40 kDa") == 40000
    assert parse_pvp_mw("40000 Da") == 40000
    assert parse_pvp_mw("40000 g/mol") == 40000
    assert parse_pvp_mw("4万") == 40000
    assert parse_pvp_mw("0.04 MDa") == 40000


def test_unrecognized_pvp_is_not_guessed():
    assert math.isnan(parse_pvp_mw("PVP unknown"))


def test_range_keeps_bounds():
    parsed = parse_numeric_range("100-200 nm", "nm")
    assert (parsed.value_numeric, parsed.range_lower, parsed.range_upper) == (150, 100, 200)

