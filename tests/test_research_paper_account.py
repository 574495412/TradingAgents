"""连续公开报价账户的因果时序、全部成本、活动与恢复反证。"""

import base64
import copy
import json
from decimal import Decimal
from urllib.parse import urlencode

import pytest

from tradingagents.dataflows.research.live_capture import BASE, MINUTE, ROUTES
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS
from tradingagents.research.quant.backtesting.paper_account import (
    PaperAccount,
    boundary,
    joint_step,
)
from tradingagents.research.quant.backtesting.paper_audit import audit
from tradingagents.research.quant.backtesting.paper_inputs import funding_window


def public(kind, value, requested, received, symbol=None, params=None):
    result = received_record(json.dumps(value), received_ms=received)
    return result | {
        "kind": kind,
        "status": 200,
        "symbol": symbol,
        "requested_ms": requested,
        "local_clock_consistent": True,
        "credentials_used": False,
        "url": BASE + ROUTES[kind] + ("?" + urlencode(params) if params else ""),
    }


def frame(
    minute, symbols=("A",), *, direction=0, price="100", now=None, funding=(), quantity="10000"
):
    closing = minute * MINUTE
    now = now or closing + 5000
    contracts, quotes, marks, evidence = [], [], {}, {}
    for symbol in symbols:
        contracts.append(
            {
                "symbol": symbol,
                "status": "TRADING",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "contractType": "TRADIFI_PERPETUAL",
                "onboardDate": 0,
                "deliveryDate": 0,
                "orderTypes": ["MARKET"],
                "filters": [
                    {"filterType": kind, "minQty": "0.01", "maxQty": "10000", "stepSize": "0.01"}
                    for kind in ("LOT_SIZE", "MARKET_LOT_SIZE")
                ]
                + [{"filterType": "MIN_NOTIONAL", "notional": "5"}],
            }
        )
        quotes.append(
            {
                "symbol": symbol,
                "bidPrice": price,
                "askPrice": price,
                "bidQty": quantity,
                "askQty": quantity,
                "time": now - 3000,
            }
        )
        marks[symbol] = {
            "row": [
                closing - MINUTE,
                price,
                price,
                price,
                price,
                "1",
                closing - 1,
                price,
                1,
                "0.5",
                str(Decimal(price) / 2),
                "0",
            ],
            "received_ms": closing + 100,
            "source_sha256": "已封存测试行情",
        }
        evidence[symbol] = {
            "hypothesis": HYPOTHESIS,
            "symbol": symbol,
            "ready": True,
            "direction": direction,
            "data_clock_ms": closing,
            "actual_decision_ms": now - 4000,
            "latest_input_received_ms": closing + 500,
            "two_completed_minutes_within_band": direction == 0,
            "orders_enabled": False,
            "production_eligible": False,
        }
    return {
        "actual_ms": now,
        "canonical_close_ms": closing,
        "clock_verified": True,
        "marks": marks,
        "evidence": evidence,
        "contracts": public("exchange", {"symbols": contracts}, now - 4000, now - 3500),
        "quotes": public("quotes", quotes, now - 3000, now - 2000),
        "funding": list(funding),
    }


def payment(minute, events, *, start=100 * MINUTE, end=None, now=None):
    closing = minute * MINUTE
    now = now or closing + 2000
    params = {
        "symbol": "A",
        "startTime": start,
        "endTime": end if end is not None else closing,
        "limit": 1000,
    }
    return public("funding", events, now - 1000, now, symbol="A", params=params)


def event(minute, rate, kind="Regular"):
    return {
        "symbol": "A",
        "fundingTime": minute * MINUTE,
        "fundingRate": rate,
        "markPrice": "100",
        "rateType": kind,
    }


def entered(*, direction=1, symbols=("A",), slip=3):
    account, frames = PaperAccount(symbols, slip), []
    for minute in range(100, 106):
        value = frame(minute, symbols, direction=direction if minute >= 103 else 0)
        account.apply(value)
        frames.append(value)
    return account, frames


@pytest.mark.parametrize("direction", [-1, 1])
def test_fill_waits_from_actual_signal_rechecks_then_uses_later_received_quote(direction):
    account, frames = entered(direction=direction)
    assert len(account.fills) == 1
    fill = account.fills[0]
    assert fill["scheduled_boundary_ms"] == boundary(frames[3]["actual_ms"])
    assert fill["entry_ms"] > fill["recheck_ms"] > fill["scheduled_boundary_ms"]
    assert fill["entry_ms"] - fill["signal_ms"] >= MINUTE
    expected = Decimal("100") * (1 + direction * Decimal("0.0003"))
    assert fill["entry_price"] == expected
    assert fill["quantity"] * expected <= Decimal("125")
    assert account.cash == Decimal("1000") - fill["entry_fee"]
    assert Decimal(account.metrics()["equity"]) < 1000


def test_startup_active_missing_input_and_rejected_recheck_do_not_recycle_activity():
    startup = PaperAccount(["A"], 3)
    for minute in range(100, 110):
        startup.apply(frame(minute, direction=1))
    assert not startup.pending and not startup.fills
    account, _ = entered()
    assert len(account.positions) == 1
    rejected = PaperAccount(["A"], 3)
    for minute in range(100, 106):
        value = frame(minute, direction=1 if minute >= 103 else 0)
        if minute == 105:
            value["evidence"]["A"] = {"ready": False}
        rejected.apply(value)
    for minute in range(106, 110):
        rejected.apply(frame(minute, direction=1))
    assert not rejected.fills and not rejected.pending
    assert rejected.declines[0]["reason"] == "RECHECK_FAILED"


def test_short_and_long_have_round_trip_fees_slippage_and_float_before_time_exit():
    for direction in (-1, 1):
        account, _ = entered(direction=direction, slip=10)
        for minute in range(106, 126):
            empty = payment(minute, [])
            account.apply(
                frame(
                    minute,
                    direction=direction,
                    funding=[empty],
                    now=minute * MINUTE + (8000 if minute == 125 else 5000),
                )
            )
        assert not account.positions
        trade = account.trades[0]
        assert trade["reason"] == "time" and trade["net_pnl"] < 0
        assert trade["entry_price"] == Decimal("100") * (1 + direction * Decimal("0.001"))
        assert trade["exit_price"] == Decimal("100") * (1 - direction * Decimal("0.001"))
        assert account.cash == Decimal("1000") + trade["net_pnl"]
        assert account.metrics()["cash_and_pnl_are_provisional"]


def test_same_timestamp_regular_and_special_fees_are_distinct_and_duplicate_download_is_idempotent():
    account, _ = entered()
    rows = [event(106, "0.001"), event(106, "-0.002", "Special")]
    record = payment(106, rows)
    before = account.cash
    account.apply(frame(106, direction=1, funding=[record]))
    expected = account.positions["A"]["quantity"] * Decimal("100") * Decimal("0.001")
    assert len(account.payments) == 2 and account.cash == before + expected
    account.apply(frame(107, direction=1, funding=[record]))
    assert len(account.payments) == 2 and account.cash == before + expected


def test_late_fee_is_charged_to_closed_trade_and_invalidates_financial_timing():
    account, _ = entered()
    for minute in range(106, 126):
        account.apply(
            frame(minute, direction=1, now=minute * MINUTE + (8000 if minute == 125 else 5000))
        )
    trade = account.trades[0]
    before = trade["net_pnl"]
    account.apply(frame(126, direction=1, funding=[payment(126, [event(106, "0.1", "Special")])]))
    assert account.trades[0]["net_pnl"] < before
    assert account.cash == Decimal("1000") + account.trades[0]["net_pnl"]
    assert (
        account.halted
        and account.quality_errors[0]["reason"] == "FUNDING_ARRIVED_AFTER_A_FINANCIAL_DECISION"
    )


def test_lot_minimum_does_not_expand_budget_and_shared_capacity_is_ordered_by_symbol():
    account, _ = entered(symbols=("E", "D", "C", "B", "A"))
    assert set(account.positions) == {"A", "B", "C", "D"}
    assert account.declines[-1]["symbol"] == "E"
    assert account.declines[-1]["reason"] == "HALT_OR_SHARED_CAPACITY"
    assert joint_step([(1, 10, Decimal("0.02")), (1, 10, Decimal("0.03"))]) == Decimal("0.06")
    expensive = PaperAccount(["A"], 3)
    for minute in range(100, 106):
        value = frame(minute, direction=1 if minute >= 103 else 0, price="1000000")
        expensive.apply(value)
    assert not expensive.fills
    assert expensive.declines[-1]["reason"] == "MINIMUM_LOT_OR_NOTIONAL_EXCEEDS_BUDGET"


def test_stale_or_pre_recheck_quote_and_insufficient_displayed_size_never_fill():
    for mode in ("early", "stale", "small"):
        account = PaperAccount(["A"], 3)
        for minute in range(100, 106):
            value = frame(
                minute,
                direction=1 if minute >= 103 else 0,
                quantity="0.01" if mode == "small" else "10000",
            )
            if minute == 105 and mode == "early":
                value["quotes"]["requested_ms"] = value["evidence"]["A"]["actual_decision_ms"] - 1
            if minute == 105 and mode == "stale":
                rows = json.loads(base64.b64decode(value["quotes"]["body_base64"]))
                rows[0]["time"] = value["actual_ms"] - 5001
                value["quotes"] = public(
                    "quotes", rows, value["actual_ms"] - 3000, value["actual_ms"] - 2000
                )
            account.apply(value)
        assert not account.fills and len(account.declines) == 1


def test_checkpoint_replay_preserves_consumed_activity_and_rejects_missing_or_modified_input():
    account, frames = entered()
    checkpoint = account.checkpoint()
    restored = PaperAccount.restore(frames, checkpoint)
    assert restored.checkpoint() == checkpoint
    with pytest.raises(ValueError, match="不能由完整输入"):
        PaperAccount.restore(frames[:-1], checkpoint)
    changed = copy.deepcopy(checkpoint)
    changed["cash"] = "1000"
    with pytest.raises(ValueError, match="不能由完整输入"):
        PaperAccount.restore(frames, changed)
    invalid = frame(106, direction=1)
    invalid["marks"]["A"]["received_ms"] = invalid["actual_ms"] + 1
    with pytest.raises(ValueError, match="实际接收证据"):
        account.apply(invalid)
    assert account.checkpoint() == checkpoint


def test_funding_unknown_type_or_truncated_window_cannot_be_assumed_zero():
    for rows in ([event(105, "0.01", "Unknown")], [event(105, "0.01")] * 1000):
        record = payment(106, rows)
        with pytest.raises(ValueError):
            funding_window(record, 106 * MINUTE + 5000)


def test_independent_financial_rebuild_rejects_omitted_fee_and_wrong_quote_source():
    account, frames = entered()
    rows = [event(106, "0.001"), event(106, "-0.002", "Special")]
    for minute in range(106, 128):
        value = frame(
            minute,
            direction=1,
            now=minute * MINUTE + (8000 if minute == 125 else 5000),
            funding=[payment(minute, rows)],
        )
        frames.append(value)
        account.apply(value)
    checked = audit(frames, account.checkpoint())
    assert checked["fills_verified"] == 2 and checked["funding_payments_verified"] == 2
    assert account.funding_verified() and not account.metrics()["cash_and_pnl_are_provisional"]
    damaged = copy.deepcopy(account.checkpoint())
    damaged["fills"][0]["entry_fee"] = "0"
    with pytest.raises(ValueError, match="入场费"):
        audit(frames, damaged)
    damaged = copy.deepcopy(account.checkpoint())
    damaged["fills"][0]["quote_source_sha256"] = "其他盘口"
    with pytest.raises(ValueError, match="来源被替换"):
        audit(frames, damaged)


def test_observed_close_profit_waits_and_high_price_alone_cannot_trigger_exit():
    account, _ = entered()
    value = frame(106, direction=1)
    value["marks"]["A"]["row"][2] = "110"
    account.apply(value)
    assert "profit_due_ms" not in account.positions["A"]
    account.apply(frame(107, direction=1, price="102"))
    due = account.positions["A"]["profit_due_ms"]
    assert due == boundary(107 * MINUTE + 5000)
    account.apply(frame(108, direction=1, price="102"))
    assert account.positions
    account.apply(frame(109, direction=1, price="102"))
    assert not account.positions and account.trades[0]["reason"] == "profit"
    assert account.trades[0]["exit_ms"] >= due


def test_account_risk_halts_entries_and_exits_after_fixed_delay():
    account, _ = entered()
    account.apply(frame(106, direction=1, price="30"))
    assert account.halted and account.max_dd > Decimal("0.08")
    due = account.positions["A"]["risk_due_ms"]
    account.apply(frame(107, direction=1, price="30"))
    assert account.positions
    account.apply(frame(108, direction=1, price="30"))
    assert not account.positions and account.trades[0]["reason"] == "risk"
    assert account.trades[0]["exit_ms"] >= due


def test_funding_recheck_cannot_withdraw_a_previous_special_event():
    account, _ = entered()
    account.apply(frame(106, direction=1, funding=[payment(106, [event(106, "0.01", "Special")])]))
    before = account.checkpoint()
    with pytest.raises(ValueError, match="撤回了已确认事件"):
        account.apply(frame(107, direction=1, funding=[payment(107, [])]))
    assert account.checkpoint() == before


def test_missing_valuation_is_unknown_but_does_not_block_a_due_exit_with_fresh_quote():
    account, frames = entered()
    for minute in range(106, 126):
        value = frame(minute, direction=1, now=minute * MINUTE + (8000 if minute == 125 else 5000))
        value["marks"] = {}
        account.apply(value)
        frames.append(value)
    assert not account.positions
    assert account.curve[-2]["equity"] is None
    assert account.curve[-1]["equity"] == account.cash
    assert not account.metrics()["drawdown_fully_observed"]
    audit(frames, account.checkpoint())
    hidden = account.checkpoint()
    hidden["quality_errors"] = []
    with pytest.raises(ValueError, match="质量失败被删除"):
        audit(frames, hidden)


def test_pre_registered_period_end_closes_and_cannot_create_new_entries():
    account = PaperAccount(["A"], 3, period_end_ms=108 * MINUTE)
    frames = []
    for minute in range(100, 113):
        value = frame(minute, direction=1 if 103 <= minute <= 108 or minute >= 111 else 0)
        account.apply(value)
        frames.append(value)
    assert account.halted and not account.positions and not account.pending
    assert len(account.trades) == 1 and account.trades[0]["reason"] == "scheduled_end"
    audit(frames, account.checkpoint())


def test_gap_does_not_count_unobserved_drawdown_as_complete():
    account, frames = entered()
    value = frame(108, direction=1)
    account.apply(value)
    frames.append(value)
    assert not account.metrics()["drawdown_fully_observed"]
    audit(frames, account.checkpoint())


def test_streamed_financial_frames_match_complete_list_and_reject_missing_tail():
    account, frames = entered()
    for minute in range(106, 126):
        value = frame(minute, direction=1, now=minute * MINUTE + (8000 if minute == 125 else 5000))
        account.apply(value)
        frames.append(value)
    checkpoint = account.checkpoint()
    assert len(account.trades) == 1
    assert audit(iter(frames), checkpoint) == audit(frames, checkpoint)
    with pytest.raises(ValueError, match="权益曲线与完整输入帧数量不一致"):
        audit(iter(frames[:-1]), checkpoint)
