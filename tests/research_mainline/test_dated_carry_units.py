"""Economic counterexamples that a superficially hedged carry result must retain."""

import pytest

from factors.common.dated_carry import basis_reference, linear_pair_pnl

UNITS = {"spot_asset": "BTC", "future_asset": "BTC", "spot_quote": "USDT", "future_quote": "USDT"}


@pytest.mark.parametrize("settlement", [30, 100, 1000])
def test_common_price_shock_cancels_only_with_matched_units_and_converged_exit(settlement):
    r = linear_pair_pnl(
        100,
        settlement,
        103,
        settlement,
        net_spot_quantity=2,
        short_future_base_quantity=2,
        all_fees_in_quote=1,
        **UNITS,
    )
    assert r["net_pnl_quote"] == 5 and r["base_quantity_matched"]


def test_base_currency_spot_fee_can_leave_real_directional_risk():
    r = linear_pair_pnl(
        100,
        1000,
        103,
        1000,
        net_spot_quantity="0.99",
        short_future_base_quantity=1,
        all_fees_in_quote=0,
        **UNITS,
    )
    assert r["residual_base_exposure"] == -0.01
    assert r["net_pnl_quote"] == -6 and not r["base_quantity_matched"]


def test_nonconverged_exit_removes_the_apparent_locked_premium():
    r = linear_pair_pnl(
        100,
        120,
        103,
        125,
        net_spot_quantity=1,
        short_future_base_quantity=1,
        all_fees_in_quote=1,
        **UNITS,
    )
    assert r["net_pnl_quote"] == -3


def test_large_annualized_quote_can_still_lose_after_roundtrip_costs():
    b = basis_reference(100, "100.01", verified_days_to_expiry=1 / 1440, **UNITS)
    assert b["simple_basis_annualized"] > 50 and not b["executable_or_net_return"]
    r = linear_pair_pnl(
        100,
        100,
        "100.01",
        100,
        net_spot_quantity=1,
        short_future_base_quantity=1,
        all_fees_in_quote="0.1",
        **UNITS,
    )
    assert r["net_pnl_quote"] == -0.09


def test_unknown_maturity_does_not_produce_an_annualized_yield():
    r = basis_reference(100, 103, **UNITS)
    assert r["simple_basis_total"] == 0.03 and r["simple_basis_annualized"] is None


@pytest.mark.parametrize(
    "change", [{"future_quote": "USD"}, {"future_asset": "ETH"}, {"contract_style": "inverse"}]
)
def test_currency_identity_and_inverse_payoffs_cannot_be_silently_mixed(change):
    with pytest.raises(ValueError):
        basis_reference(100, 103, **{**UNITS, **change})


def test_no_borrowing_cost_is_hidden_in_a_zero_spread():
    r = linear_pair_pnl(
        100,
        120,
        100,
        120,
        net_spot_quantity=1,
        short_future_base_quantity=1,
        all_fees_in_quote="0.5",
        **UNITS,
    )
    assert r["net_pnl_quote"] == -0.5
