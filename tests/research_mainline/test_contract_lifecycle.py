"""No future notice access, no postterminal entry and no invented settlement value."""

import pytest

from research_tools.historical_universe.lifecycle import (
    UnverifiedTerminalSettlement,
    blocked_new_entry,
    require_terminal_value,
    validate,
    whole_daily_feature_allowed,
)


def event():
    return {
        "symbol": "MATICUSDT",
        "contract_type": "USD_M_PERPETUAL",
        "available_at_assumed": "2024-08-28T09:05:07Z",
        "entry_stop_at": "2024-09-04T08:30:00Z",
        "settlement_at": "2024-09-04T09:00:00Z",
        "terminal_settlement_price": None,
        "terminal_settlement_fee": None,
    }


def test_new_entry_boundary_is_distinct_from_final_settlement():
    e = event()
    assert not blocked_new_entry(e, "2024-09-04T08:29:59Z")
    assert blocked_new_entry(e, "2024-09-04T08:30:00Z")


def test_future_notice_cannot_disable_earlier_entry():
    assert not blocked_new_entry(event(), "2024-08-20T00:00:00Z")


def test_daily_bar_spanning_termination_is_not_a_complete_tradable_day():
    e = event()
    assert whole_daily_feature_allowed(e, "2024-09-03T00:00Z", "2024-09-04T00:00Z")
    assert not whole_daily_feature_allowed(e, "2024-09-04T00:00Z", "2024-09-05T00:00Z")
    assert not whole_daily_feature_allowed(e, "2024-09-09T00:00Z", "2024-09-10T00:00Z")


def test_incomplete_future_daily_bar_rejected():
    with pytest.raises(ValueError, match="Incomplete future"):
        whole_daily_feature_allowed(event(), "2024-09-03T00:00Z", "2024-09-03T12:00Z")


def test_open_inventory_at_terminal_requires_actual_cash_settlement_not_last_close():
    with pytest.raises(UnverifiedTerminalSettlement):
        require_terminal_value(event(), "2024-09-04T09:00Z", 100)
    require_terminal_value(event(), "2024-09-04T08:59Z", 100)
    require_terminal_value(event(), "2024-09-04T09:00Z", 0)


def test_relisting_requires_separate_contract_identity():
    with pytest.raises(ValueError, match="Overlapping"):
        validate([event(), event()])
