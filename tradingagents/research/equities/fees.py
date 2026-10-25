"""Dated statutory components, separate from assumed broker and slippage costs.

Only registered ETFs and 2023-2026Q3 are supported. Rounding up US pass-through
components is conservative; SRO charge date vs broker trade date is unverified.
"""

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal


def charge(market, day, side, quantity, price, spec):
    day = str(day)[:10]
    if not "2023-01-01" <= day < "2026-10-01" or side not in {"BUY", "SELL"}:
        raise ValueError("Fee schedule outside reviewed dates or side")
    if market not in {"US", "HK"} or any(s.get("instrument") != "ETF" for s in spec["symbols"]):
        raise ValueError("Only reviewed ETF fee treatment supported")
    value = Decimal(str(price)) * quantity

    def cents(amount, up=False):
        return float(
            amount.quantize(Decimal(".01"), rounding=ROUND_CEILING if up else ROUND_HALF_UP)
        )

    parts = {
        "commission": cents(
            max(
                Decimal(str(spec["minimum_fee"])),
                value * Decimal(str(spec["commission_bps"])) / 10000,
            )
        )
    }
    if market == "US":
        rate = "22.90"
        for effective, new_rate in [
            ("2023-02-27", "8"),
            ("2024-05-22", "27.80"),
            ("2025-05-14", "0"),
            ("2026-04-04", "20.60"),
        ]:
            if day >= effective:
                rate = new_rate
        taf_rate, cap = (
            (".000145", "7.27")
            if day < "2024-01-01"
            else (".000166", "8.30")
            if day < "2026-01-01"
            else (".000195", "9.79")
        )
        parts.update(
            {
                "sec31": cents(value * Decimal(rate) / 1000000, True) if side == "SELL" else 0.0,
                "taf": cents(min(Decimal(cap), Decimal(taf_rate) * quantity), True)
                if side == "SELL"
                else 0.0,
            }
        )
    else:
        settlement = (
            min(Decimal(100), max(Decimal(2), value * Decimal(".00002")))
            if day < "2025-06-30"
            else value * Decimal(".000042")
        )
        parts.update(
            {
                "sfc": cents(value * Decimal(".000027")),
                "afrc": cents(value * Decimal(".0000015")),
                "exchange": cents(value * Decimal(".0000565")),
                "settlement": cents(settlement),
                "stamp": 0.0,
            }
        )
    return parts
