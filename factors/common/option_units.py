"""Exact payoff/unit checks, not option valuation, margin or strategy admission.

Source-bound example: Deribit BTC inverse option quotes/settles in BTC, while
strike/index areUSD and native amount is underlyingBTC (not USD premium).
Linear cash-settled payoff arithmetic is shareable; equity calendars/accounts
and every venue's contract multiplier/fees need separate verification.
"""

from decimal import Decimal, InvalidOperation
from fractions import Fraction


def number(value, *, positive=False):
    try:
        x = Decimal(str(value))
    except (ValueError, InvalidOperation) as error:
        raise ValueError("Finite decimal source required") from error
    if not x.is_finite() or x < 0 or (positive and x == 0):
        raise ValueError("Invalid nonnegative source amount or positive price")
    return Fraction(x)


def option_payoff(quantity_underlying, strike, settlement_index, option_type, *, settlement_units):
    q = number(quantity_underlying)
    k = number(strike, positive=True)
    s = number(settlement_index, positive=True)
    if option_type not in {"call", "put"}:
        raise ValueError("Unknown option type")
    intrinsic = max(s - k, 0) if option_type == "call" else max(k - s, 0)
    cash = q * intrinsic
    if settlement_units == "cash":
        return cash
    if settlement_units == "underlying_coin":
        return cash / s
    raise ValueError("Explicit cash or inverse-coin settlement units required")


def premium_cash_value(quantity_underlying, quote_price_coin, index_price):
    return (
        number(quantity_underlying) * number(quote_price_coin) * number(index_price, positive=True)
    )


def underlying_notional_fits(quantity_underlying, index_price, budget_cash, gross_cap):
    q = number(quantity_underlying)
    s = number(index_price, positive=True)
    budget = number(budget_cash, positive=True)
    cap = number(gross_cap)
    if cap > 1:
        raise ValueError("This necessary check uses an unlevered gross cap")
    return q * s <= budget * cap
