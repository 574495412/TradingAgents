"""Independent integer-unit proof of bounded inventories, without factor imports."""

from decimal import Decimal
from math import gcd


def audit(case):
    inputs, cover, exact = case["inputs"], case["bounded_cover"], case["minimum_exact_pair"]
    strings = [
        inputs[k] for k in ["spot_step", "future_base_step", "gross_spot_cap", "short_base_cap"]
    ]
    fee_decimal = Decimal(inputs["base_fee_rate"])
    scale = max(
        max(-Decimal(v).as_tuple().exponent for v in strings),
        -Decimal(inputs["spot_step"]).as_tuple().exponent - fee_decimal.as_tuple().exponent,
    )
    base = 10**scale

    def units(value):
        # Decimal scaling must be exact before converting to integers.
        d = Decimal(str(value)) * base
        assert d == d.to_integral_value()
        return int(d)

    fs = 10 ** max(0, -fee_decimal.as_tuple().exponent)
    fn = int(fee_decimal * fs)
    s, f, cap_s, cap_f = [units(v) for v in strings]
    a, b = s * (fs - fn), f * fs
    g = gcd(a, b)
    assert exact["spot_lots"] == b // g and exact["future_lots"] == a // g
    assert gcd(exact["spot_lots"], exact["future_lots"]) == 1
    assert units(exact["gross_spot"]) == exact["spot_lots"] * s
    assert units(exact["short_base"]) == units(exact["net_spot"]) == exact["future_lots"] * f
    assert units(exact["gross_spot"]) * (fs - fn) == units(exact["net_spot"]) * fs
    ns, nf = cover["spot_lots"], cover["future_lots"]
    gross, fee, net, short = [
        units(cover[k]) for k in ["gross_spot", "base_fee", "net_spot", "short_base"]
    ]
    assert gross == ns * s <= cap_s and short == nf * f <= cap_f
    assert gross * fn == fee * fs and gross - fee == net
    assert net >= short and (ns == 0 or (ns - 1) * s * (fs - fn) < short * fs)
    # The next short lot must violate at least one registered cap.
    next_needed = ((nf + 1) * f * fs + s * (fs - fn) - 1) // (s * (fs - fn))
    assert (nf + 1) * f > cap_f or next_needed * s > cap_s
    assert units(cover["unhedged_long_base"]) == net - short
    sell, dust = units(cover["spot_sell_on_same_grid"]), units(cover["retained_exit_dust_base"])
    assert sell % s == 0 and 0 <= dust < s and sell + dust == net
    return {
        "passed": True,
        "integer_base_decimals": scale,
        "minimum_exact_pair_coprime_proof": True,
        "maximum_capped_short_and_minimum_covering_spot_proved": True,
        "base_fee_and_inventory_conservation_exact": True,
        "residual_and_dust_retained_not_written_off": True,
        "factor_module_imported": False,
        "actual_historical_filters_or_client_cash_certified": False,
    }
