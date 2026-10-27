from fractions import Fraction

import pytest

from factors.common.option_units import option_payoff, premium_cash_value, underlying_notional_fits


def test_inverse_payoff_coin_and_cash_reconcile_in_both_directions():
    for kind, settlement in [("call", 120000), ("put", 80000)]:
        cash = option_payoff("0.1", 100000, settlement, kind, settlement_units="cash")
        coin = option_payoff("0.1", 100000, settlement, kind, settlement_units="underlying_coin")
        assert cash == 2000 and coin * settlement == cash
    assert option_payoff("0.1", 100000, 100000, "call", settlement_units="cash") == 0


def test_native_coin_premium_is_not_underlying_notional_or_dollar_premium():
    assert premium_cash_value("0.1", "0.0425", "93370.2") == Fraction("396.82335")
    assert not underlying_notional_fits("0.1", "93370.2", 10000, "0.8")
    assert underlying_notional_fits(
        "0.01", "93370.2", 10000, "0.8"
    )  # Arithmetic only, not a legal lot.


def test_mark_quote_difference_cannot_be_credited_as_attainable_sell_fill():
    observed_bid = premium_cash_value("0.1", "0.0425", "93370.2")
    model_mark = premium_cash_value("0.1", "0.0433", "93370.2")
    assert model_mark - observed_bid == Fraction("7.469616")


def test_bad_units_nonfinite_and_zero_settlement_rejected():
    for x in [None, "NaN", "Infinity", -1]:
        with pytest.raises(ValueError):
            premium_cash_value(".1", x, 100000)
    with pytest.raises(ValueError):
        option_payoff(".1", 100000, 0, "put", settlement_units="underlying_coin")
    with pytest.raises(ValueError):
        option_payoff(".1", 100000, 120000, "call", settlement_units="guess")
