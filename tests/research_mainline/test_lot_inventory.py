"""Exact discrete inventory counterexamples, not fee/venue certifications."""

from copy import deepcopy
from decimal import localcontext

import pytest

from factors.common.lot_inventory import bounded_cover, exact_pair_lattice
from research_tools.dated_carry.audit_lot_inventory import audit


@pytest.mark.parametrize("spot_step", ["0.00001", "0.0001"])
def test_base_fee_can_force_one_whole_coin_minimum_exact_pair(spot_step):
    result = exact_pair_lattice(
        spot_step=spot_step, future_base_step="0.001", base_fee_rate="0.001"
    )
    assert result["gross_spot"] == "1" and result["short_base"] == "0.999"


def test_no_base_fee_is_a_different_inventory_contract():
    result = exact_pair_lattice(spot_step="0.00001", future_base_step="0.001", base_fee_rate="0")
    assert result["gross_spot"] == result["net_spot"] == "0.001"


def test_cover_uses_ceil_and_keeps_dust_not_a_symmetric_round():
    result = bounded_cover(
        gross_spot_cap="0.013344781233983596",
        short_base_cap="0.013331436452749612",
        spot_step="0.00001",
        future_base_step="0.001",
        base_fee_rate="0.001",
    )
    assert result["short_base"] == "0.013" and result["gross_spot"] == "0.01302"
    assert result["net_spot"] == "0.01300698" and result["unhedged_long_base"] == "0.00000698"
    assert (
        result["spot_sell_on_same_grid"] == "0.013"
        and result["retained_exit_dust_base"] == "0.00000698"
    )
    assert not result["matched_exactly"]


def test_too_small_net_inventory_must_not_open_one_naked_short_lot():
    result = bounded_cover(
        gross_spot_cap="0.001",
        short_base_cap="0.001",
        spot_step="0.00001",
        future_base_step="0.001",
        base_fee_rate="0.001",
    )
    assert not result["nonzero_discrete_pair"] and result["gross_spot"] == "0"


def test_zero_inventory_caps_stay_zero():
    result = bounded_cover(
        gross_spot_cap="0",
        short_base_cap="0",
        spot_step="100",
        future_base_step="1",
        base_fee_rate="0",
    )
    assert result["spot_lots"] == result["future_lots"] == 0


def test_stock_integer_lot_reuse_does_not_invent_fractional_shares():
    result = bounded_cover(
        gross_spot_cap="250",
        short_base_cap="1000",
        spot_step="100",
        future_base_step="1",
        base_fee_rate="0",
    )
    assert result["gross_spot"] == result["short_base"] == "200"


def test_exact_rational_result_does_not_depend_on_global_decimal_precision():
    with localcontext() as context:
        context.prec = 5
        result = bounded_cover(
            gross_spot_cap="0.004314438541992159",
            short_base_cap="0.004310124103450167",
            spot_step="0.0001",
            future_base_step="0.001",
            base_fee_rate="0.001",
        )
    assert result["gross_spot"] == "0.0041" and result["net_spot"] == "0.0040959"


@pytest.mark.parametrize("fee", ["1", "-0.001", "NaN"])
def test_invalid_fees_cannot_create_free_or_negative_inventory(fee):
    with pytest.raises(ValueError):
        exact_pair_lattice(spot_step="0.001", future_base_step="0.001", base_fee_rate=fee)


def test_independent_integer_audit_rejects_writing_off_unsellable_inventory():
    inputs = {
        "gross_spot_cap": "0.004314438541992159",
        "short_base_cap": "0.004310124103450167",
        "spot_step": "0.0001",
        "future_base_step": "0.001",
        "base_fee_rate": "0.001",
    }
    case = {
        "inputs": inputs,
        "bounded_cover": bounded_cover(**inputs),
        "minimum_exact_pair": exact_pair_lattice(
            **{k: inputs[k] for k in ["spot_step", "future_base_step", "base_fee_rate"]}
        ),
    }
    assert audit(case)["passed"]
    forged = deepcopy(case)
    forged["bounded_cover"]["retained_exit_dust_base"] = "0"
    with pytest.raises(AssertionError):
        audit(forged)
