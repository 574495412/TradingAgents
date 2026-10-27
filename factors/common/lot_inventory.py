"""Exact discrete inventory arithmetic shared by cash-equity/crypto research.

This module does not infer exchange rules from prices or submit orders. A base
fee reduces purchased inventory; quote-currency commissions belong in a separate
cash ledger. Explicit quantity caps must already reflect cash/liquidity/risk.
All results remain inventory diagnostics, not an executable-order certificate.
"""

from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction


def _number(value, *, zero=False):
    try:
        d = Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError("Finite decimal required") from e
    if not d.is_finite() or (d < 0 if zero else d <= 0):
        raise ValueError("Positive finite step or nonnegative quantity/rate required")
    if len(d.as_tuple().digits) > 36 or abs(d.as_tuple().exponent) > 18:
        raise ValueError("Unsupported decimal magnitude/precision")
    return Fraction(d)


def _text(value):
    with localcontext() as context:
        context.prec = 120
        d = Decimal(value.numerator) / Decimal(value.denominator)
        return format(d, "f").rstrip("0").rstrip(".") if "." in format(d, "f") else format(d, "f")


def _inputs(spot_step, future_base_step, base_fee_rate):
    spot, future, fee = (
        _number(spot_step),
        _number(future_base_step),
        _number(base_fee_rate, zero=True),
    )
    if fee >= 1:
        raise ValueError("Base fee must be below one")
    return spot, future, fee


def exact_pair_lattice(*, spot_step, future_base_step, base_fee_rate):
    """Find the smallest strictly positive *exactly* matched discrete pair.

    Reduced ratio a/b=(spot_step*(1-fee))/future_step implies a*n_spot
    =b*n_future. Coprimality forces the first positive solution (b,a).
    No floating tolerance or best-return search is involved.
    """
    spot, future, fee = _inputs(spot_step, future_base_step, base_fee_rate)
    ratio = spot * (1 - fee) / future
    ns, nf = ratio.denominator, ratio.numerator
    gross, net = ns * spot, nf * future
    assert gross * (1 - fee) == net
    return {
        "spot_lots": ns,
        "future_lots": nf,
        "gross_spot": _text(gross),
        "net_spot": _text(net),
        "short_base": _text(net),
        "base_fee": _text(gross * fee),
        "ratio_a": ratio.numerator,
        "ratio_b": ratio.denominator,
        "venue_order_eligibility_certified": False,
    }


def bounded_cover(*, gross_spot_cap, short_base_cap, spot_step, future_base_step, base_fee_rate):
    """Largest capped short lots, bought with the smallest covering spot lots.

    Buy-first net inventory covers the requested short target in this arithmetic
    only. Actual independent fills can still fail. The unhedged long and unsellable
    remainder are retained; no automatic dust conversion or quote fees assumed.
    """
    spot, future, fee = _inputs(spot_step, future_base_step, base_fee_rate)
    cap_s, cap_f = _number(gross_spot_cap, zero=True), _number(short_base_cap, zero=True)
    available_spot_lots = cap_s // spot
    maximum_net_spot = available_spot_lots * spot * (1 - fee)
    nf = min(maximum_net_spot, cap_f) // future
    short = nf * future
    required_spot_lots = short / (spot * (1 - fee))
    ns = -(-required_spot_lots.numerator // required_spot_lots.denominator)
    gross = ns * spot
    net = gross * (1 - fee)
    residual = net - short
    sell = (net // spot) * spot
    dust = net - sell
    assert 0 <= gross <= cap_s and 0 <= short <= cap_f
    assert 0 <= residual < spot * (1 - fee)
    assert 0 <= dust < spot and sell + dust == net
    if nf:
        assert (ns - 1) * spot * (1 - fee) < short
    return {
        "spot_lots": ns,
        "future_lots": nf,
        "gross_spot": _text(gross),
        "base_fee": _text(gross * fee),
        "net_spot": _text(net),
        "short_base": _text(short),
        "unhedged_long_base": _text(residual),
        "spot_sell_on_same_grid": _text(sell),
        "retained_exit_dust_base": _text(dust),
        "matched_exactly": residual == 0,
        "nonzero_discrete_pair": nf > 0,
        "spot_cap_not_exceeded": True,
        "future_cap_not_exceeded": True,
        "actual_hedge_fill_or_venue_order_eligibility_certified": False,
    }
