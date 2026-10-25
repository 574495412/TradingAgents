"""Known terminal boundaries are data availability rules, not settlement-price proxies."""

import pandas as pd


class UnverifiedTerminalSettlement(ValueError):
    pass


def validate(events):
    by_symbol = {}
    for e in events:
        if e["contract_type"] != "USD_M_PERPETUAL" or e["symbol"] in by_symbol:
            raise ValueError("Overlapping or unsupported contract lifecycle")
        available = pd.Timestamp(e["available_at_assumed"])
        entry = pd.Timestamp(e["entry_stop_at"])
        end = pd.Timestamp(e["settlement_at"])
        if not available < entry <= end:
            raise ValueError("Notice availability or contract event clocks invalid")
        by_symbol[e["symbol"]] = e
    return by_symbol


def blocked_new_entry(event, clock):
    t = pd.Timestamp(clock)
    return t >= pd.Timestamp(event["available_at_assumed"]) and t >= pd.Timestamp(
        event["entry_stop_at"]
    )


def whole_daily_feature_allowed(event, bar_open, decision):
    t, known = pd.Timestamp(bar_open), pd.Timestamp(decision)
    if t + pd.Timedelta(days=1) > known:
        raise ValueError("Incomplete future candle")
    if known < pd.Timestamp(event["available_at_assumed"]):
        return True
    return t + pd.Timedelta(days=1) <= pd.Timestamp(event["settlement_at"])


def require_terminal_value(event, clock, quantity):
    if (
        quantity
        and pd.Timestamp(clock) >= pd.Timestamp(event["settlement_at"])
        and (
            event.get("terminal_settlement_price") is None
            or event.get("terminal_settlement_fee") is None
        )
    ):
        raise UnverifiedTerminalSettlement(
            "Actual terminal price and fee are unverified; do not use the last archived close or zero"
        )
