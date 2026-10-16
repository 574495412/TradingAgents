"""人工端点反证买卖反弹、基线移动与时间泄露，不是市场收益。"""

import copy
from decimal import Decimal

import pytest

from tradingagents.dataflows.research.live_capture import canonical
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.microstructure import changes, endpoint, parse_message


def message(role, clock=120_000, price="100"):
    data = {"s": "AUSDT", "E": clock - 90, "T": clock - 100, "st": 1}
    suffix = "@bookTicker" if role == "quote" else "@aggTrade"
    if role == "quote":
        data.update(e="bookTicker", u=1, b="99.9", a="100.1", B="10", A="10")
    else:
        data.update(e="aggTrade", a=1, f=10, l=11, p=price, q="1", nq="0", m=True)
    record = received_record(
        canonical({"stream": "ausdt" + suffix, "data": data}), received_ms=clock - 50
    )
    record.update(source_role=role, generation="observer:1", clock_verified=True)
    return record


def point(clock, price="100", median="0", quote=True, trade=True):
    epoch = {"generation": "observer:1", "confirmed_ms": 60_000, "end_ms": 500_000}
    return endpoint(
        price,
        "100",
        median,
        symbol="AUSDT",
        activity_id="人工活动",
        ledger_identity="人工压力账户",
        data_ms=clock,
        decision_ms=clock + 2000,
        price_received_ms=clock + 100,
        index_received_ms=clock + 200,
        quote=parse_message(message("quote", clock), ["AUSDT"]) if quote else None,
        quote_epoch=epoch,
        trade=parse_message(message("aggregate_trade", clock, price), ["AUSDT"]) if trade else None,
        trade_epoch=epoch,
    )


def test_aggregate_normal_quantity_can_be_zero_without_excluding_rpi_volume():
    value = parse_message(message("aggregate_trade"), ["AUSDT"])
    assert value["quantity"] == "1" and value["normal_quantity"] == "0"
    assert value["last_all_type_trade_time_proven"] is False


def test_trade_price_repair_with_unchanged_midpoint_is_not_reported_as_account_profit():
    first, second = point(120_000, "99"), point(180_000, "100")
    result = changes(first, second, 1)
    assert result["directional_changes"]["basis"] == "0.01"
    assert Decimal(result["directional_changes"]["mid_basis"]) == 0
    assert result["directional_changes"]["print_component"] == "0.01"
    assert not result["account_return_computed"] and not result["causal_artifact_proven"]


def test_median_only_residual_repair_cannot_turn_into_price_repair():
    result = changes(point(120_000, median="0.006"), point(180_000, median="0"), 1)
    assert result["directional_changes"]["basis"] == "0"
    assert result["directional_changes"]["residual"] == "0.006"
    assert result["directional_changes"]["median_basis"] == "-0.006"


@pytest.mark.parametrize(
    "failure", ["post_endpoint_quote", "late_receipt", "clock_unknown", "gap", "missing"]
)
def test_unavailable_quote_is_unknown_rather_than_filled_from_the_future(failure):
    value = parse_message(message("quote"), ["AUSDT"])
    epoch = {"generation": "observer:1", "confirmed_ms": 60_000, "end_ms": 500_000}
    if failure == "post_endpoint_quote":
        value["transaction_ms"] = 120_001
    elif failure == "late_receipt":
        value["received_ms"] = 123_000
    elif failure == "clock_unknown":
        value["clock_verified"] = False
    elif failure == "gap":
        epoch["end_ms"] = 121_000
    else:
        value = None
    result = endpoint(
        "100",
        "100",
        "0",
        symbol="AUSDT",
        activity_id="人工活动",
        ledger_identity="人工压力账户",
        data_ms=120_000,
        decision_ms=122_000,
        price_received_ms=120_100,
        index_received_ms=120_200,
        quote=value,
        quote_epoch=epoch,
    )
    assert result["quote_unknown_reason"] is not None and result["mid_basis"] is None


@pytest.mark.parametrize("key", ["symbol", "activity_id", "ledger_identity"])
def test_changes_cannot_join_other_symbols_activities_or_ledgers(key):
    first, second = point(120_000), copy.deepcopy(point(180_000))
    second[key] = "另一个归属"
    with pytest.raises(ValueError):
        changes(first, second, 1)


def test_future_index_receipt_cannot_be_used_in_an_earlier_diagnostic():
    with pytest.raises(ValueError):
        endpoint(
            "100",
            "100",
            "0",
            symbol="AUSDT",
            activity_id="人工活动",
            ledger_identity="人工压力账户",
            data_ms=120_000,
            decision_ms=122_000,
            price_received_ms=120_100,
            index_received_ms=123_000,
        )
