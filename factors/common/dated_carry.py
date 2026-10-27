"""Unit-checked linear dated-futures references; no implicit executable alpha.

Prices are quote currency per base unit. Quantities must be actual net base
inventory after any coin-denominated spot fee and contract-multiplier mapping.
Inverse contracts require their own payoff implementation and are rejected.
The optional horizon must come from separate maturity evidence, not this module.
"""

import math
from decimal import Decimal


def _number(value, positive=False):
    d = Decimal(str(value))
    if not d.is_finite() or (d <= 0 if positive else d < 0):
        raise ValueError("Finite positive price or nonnegative quantity/cost required")
    return d


def _units(spot_asset, future_asset, spot_quote, future_quote, contract_style):
    if contract_style != "linear":
        raise ValueError("Inverse/unknown contract cannot use linear payoff")
    if not spot_asset or spot_asset != future_asset:
        raise ValueError("Underlying identity must match")
    if not spot_quote or spot_quote != future_quote:
        raise ValueError("Quote currencies must match; USD and USDT are distinct")


def basis_reference(
    spot_price,
    future_price,
    *,
    spot_asset,
    future_asset,
    spot_quote,
    future_quote,
    contract_style="linear",
    verified_days_to_expiry=None,
):
    """Raw matched-price basis, not bid/ask execution, a fee-adjusted yield or NAV.

    Omit the horizon when historic maturity is unverified. A high annualized
    number near expiry must not replace the total spread available for costs.
    """
    _units(spot_asset, future_asset, spot_quote, future_quote, contract_style)
    spot, future = _number(spot_price, True), _number(future_price, True)
    simple = float(future / spot - 1)
    log = math.log(float(future / spot))
    years = None
    if verified_days_to_expiry is not None:
        years = float(_number(verified_days_to_expiry, True)) / 365
    return {
        "simple_basis_total": simple,
        "log_basis_total": log,
        "spread_quote_per_base": float(future - spot),
        "simple_basis_annualized": simple / years if years else None,
        "log_basis_annualized": log / years if years else None,
        "verified_days_to_expiry": float(verified_days_to_expiry)
        if verified_days_to_expiry is not None
        else None,
        "executable_or_net_return": False,
    }


def linear_pair_pnl(
    spot_entry,
    spot_exit,
    future_entry,
    future_exit,
    *,
    net_spot_quantity,
    short_future_base_quantity,
    all_fees_in_quote,
    spot_asset,
    future_asset,
    spot_quote,
    future_quote,
    contract_style="linear",
):
    """Reconcile retained actual spot-long/linear-future-short fills in one quote.

    This is a payoff identity, not an execution or margin model. Unmatched net
    quantities are deliberately retained and exposed instead of assumed hedged.
    """
    _units(spot_asset, future_asset, spot_quote, future_quote, contract_style)
    si, so, fi, fo = [_number(p, True) for p in [spot_entry, spot_exit, future_entry, future_exit]]
    qs, qf, fees = [
        _number(p) for p in [net_spot_quantity, short_future_base_quantity, all_fees_in_quote]
    ]
    spot_pnl, future_pnl = qs * (so - si), qf * (fi - fo)
    return {
        "spot_price_pnl": float(spot_pnl),
        "future_price_pnl": float(future_pnl),
        "fees_quote": float(fees),
        "net_pnl_quote": float(spot_pnl + future_pnl - fees),
        "residual_base_exposure": float(qs - qf),
        "base_quantity_matched": qs == qf,
        "complete_account_or_margin_certificate": False,
    }
