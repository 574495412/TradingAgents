"""固定期限收盘退出须使用完整分钟，并由独立账本核验时钟、价格和资金费。"""

from copy import deepcopy

import pytest

from tradingagents.research.quant.backtesting.account import Account
from tradingagents.research.quant.backtesting.search import Audit
from tradingagents.research.quant.backtesting.signals import MINUTE, Signal


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_close_exit_option_cannot_be_implicitly_enabled_by_nonboolean_values(value):
    with pytest.raises(ValueError, match="布尔"):
        Account("typed_option", 20, 3, [], timed_exit_at_close=value)


def quotes(open_price=100.0, close_price=101.0, symbols=("PUBLIC",)):
    return dict.fromkeys(symbols, (open_price, close_price, True, True))


def complete(account, *, final_minute=7, prices=None, pulses=None):
    """只推进原始分钟；审计另行读取成交和对应行情，不复用账户计算。"""
    history = []
    for minute in range(1, final_minute + 1):
        clock = minute * MINUTE
        market = prices(minute) if prices else quotes()
        signals = (
            pulses(minute)
            if pulses
            else ([Signal("PUBLIC", 1, 1.0, MINUTE, "sealed")] if minute == 1 else [])
        )
        account.open_phase(clock - MINUTE, market)
        account.close_phase(clock, market, signals, {}, final=minute == final_minute)
        history.append((clock, market, minute == final_minute))
    return history


def audit_all(account, funding, history):
    auditor = Audit(account, funding)
    for clock, market, final in history:
        auditor.consume(clock, market, final=final)
    return auditor.finish()


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("hold", [1, 2])
@pytest.mark.parametrize("cost", [0, 11])
def test_fixed_close_exit_uses_complete_minute_close_and_declared_slippage(direction, hold, cost):
    account = Account(
        "close_exit", hold, cost, [], take_profit_fraction=None, timed_exit_at_close=True
    )

    def pulses(minute):
        return [Signal("PUBLIC", direction, 1.0, MINUTE, "sealed")] if minute == 1 else []

    history = complete(account, pulses=pulses)
    trade = account.trades[0]
    assert trade["entry_ms"] == 2 * MINUTE
    assert trade["exit_ms"] == (2 + hold) * MINUTE
    assert trade["reason"] == "time_close"
    assert trade["exit_price"] == pytest.approx(101 * (1 - direction * cost / 10_000))
    assert audit_all(account, [], history)["passed"]


def test_close_mode_does_not_exit_at_open_when_the_clock_reaches_the_hold_boundary():
    account = Account("close_exit", 2, 3, [], timed_exit_at_close=True, take_profit_fraction=None)
    account.schedule(Signal("PUBLIC", 1, 1.0, MINUTE, "sealed"), MINUTE)
    account.open_phase(2 * MINUTE, quotes())
    account.open_phase(4 * MINUTE, quotes())
    assert "PUBLIC" in account.positions and not account.trades


@pytest.mark.parametrize("direction", [-1, 1])
def test_funding_is_exclusive_at_close_exit_and_precedes_same_clock_entries(direction):
    event_times = [2 * MINUTE, 2 * MINUTE + 1, 3 * MINUTE, 4 * MINUTE - 1, 4 * MINUTE]
    funding = [
        {"symbol": "PUBLIC", "time_ms": stamp, "mark_price": 100.0, "rate": 0.001}
        for stamp in event_times
    ]
    account = Account(
        "close_funding", 2, 3, funding, take_profit_fraction=None, timed_exit_at_close=True
    )

    def pulses(minute):
        return [Signal("PUBLIC", direction, 1.0, MINUTE, "sealed")] if minute == 1 else []

    history = complete(account, pulses=pulses)
    assert [event["time_ms"] for event in account.funding_ledger] == event_times[1:-1]
    expected = -direction * account.trades[0]["quantity"] * 100 * 0.001 * 3
    assert account.trades[0]["funding"] == pytest.approx(expected)
    assert audit_all(account, funding, history)["passed"]


def test_same_clock_funding_keeps_other_inventory_after_one_position_exits_at_close():
    funding = [
        {"symbol": symbol, "time_ms": 4 * MINUTE, "mark_price": 100.0, "rate": 0.001}
        for symbol in ("FIRST", "SECOND")
    ]
    account = Account(
        "shared_inventory", 2, 3, funding, take_profit_fraction=None, timed_exit_at_close=True
    )

    def pulses(minute):
        symbol = {1: "FIRST", 2: "SECOND"}.get(minute)
        return [Signal(symbol, 1, 1.0, minute * MINUTE, "sealed")] if symbol else []

    history = complete(
        account, prices=lambda minute: quotes(symbols=("FIRST", "SECOND")), pulses=pulses
    )
    assert [event["symbol"] for event in account.funding_ledger] == ["SECOND"]
    assert [(trade["symbol"], trade["exit_ms"]) for trade in account.trades] == [
        ("FIRST", 4 * MINUTE),
        ("SECOND", 5 * MINUTE),
    ]
    assert audit_all(account, funding, history)["passed"]


def test_risk_exit_still_waits_for_later_open_and_uses_that_open_price():
    funding = [{"symbol": "PUBLIC", "time_ms": 5 * MINUTE, "mark_price": 17.0, "rate": 0.001}]
    account = Account(
        "close_risk", 60, 3, funding, take_profit_fraction=None, timed_exit_at_close=True
    )

    def prices(minute):
        if minute < 4:
            return quotes(100, 100)
        return quotes(100, 20) if minute == 4 else quotes(17, 20)

    history = complete(account, prices=prices)
    trade = account.trades[0]
    assert trade["reason"] == "risk"
    assert trade["exit_ms"] == 5 * MINUTE
    assert trade["exit_price"] == pytest.approx(17 * (1 - 3 / 10_000))
    # 风险平仓在开盘执行，同钟资金费仍先归旧库存，区别于固定期限的收盘退出。
    assert [event["time_ms"] for event in account.funding_ledger] == [5 * MINUTE]
    assert trade["funding"] < 0
    assert account.halted
    assert audit_all(account, funding, history)["passed"]


@pytest.mark.parametrize("hold", [2, 60])
def test_scheduled_end_has_priority_over_close_expiry(hold):
    account = Account(
        "close_final", hold, 3, [], take_profit_fraction=None, timed_exit_at_close=True
    )
    history = complete(account, final_minute=4)
    assert account.trades[0]["reason"] == "scheduled_end"
    assert account.trades[0]["exit_ms"] == 4 * MINUTE
    assert audit_all(account, [], history)["passed"]


@pytest.mark.parametrize(
    "mutation",
    ["price", "entry", "hold", "late", "reason", "direction", "action", "disabled", "clock"],
)
def test_independent_audit_rejects_forged_current_clock_fills(mutation):
    account = Account("close_tamper", 2, 3, [], take_profit_fraction=None, timed_exit_at_close=True)
    history = complete(account)
    account = deepcopy(account)
    fill = account.fills[1]
    if mutation == "price":
        fill["price"] *= 1.01
    elif mutation == "entry":
        account.trades[0]["entry_ms"] -= MINUTE
    elif mutation == "hold":
        account.hold += MINUTE
    elif mutation == "late":
        account.hold -= MINUTE
    elif mutation == "reason":
        account.trades[0]["reason"] = "risk"
    elif mutation == "direction":
        fill["direction"] *= -1
    elif mutation == "action":
        fill["action"] = "OPEN"
    elif mutation == "disabled":
        account.timed_exit_at_close = False
    else:
        fill["clock"] -= MINUTE
    with pytest.raises(ValueError):
        audit_all(account, [], history)


def test_independent_audit_rejects_funding_incorrectly_charged_at_exact_close_exit():
    funding = [{"symbol": "PUBLIC", "time_ms": 4 * MINUTE, "mark_price": 100.0, "rate": 0.01}]
    account = Account(
        "close_wrong_funding", 2, 3, funding, take_profit_fraction=None, timed_exit_at_close=True
    )
    history = []
    for minute in range(1, 7):
        clock, market = minute * MINUTE, quotes()
        account.open_phase(clock - MINUTE, market)
        if minute == 4:
            account.settle(clock, True)
        pulse = [Signal("PUBLIC", 1, 1.0, MINUTE, "sealed")] if minute == 1 else []
        account.close_phase(clock, market, pulse, {}, final=minute == 6)
        history.append((clock, market, minute == 6))
    assert account.funding_ledger
    with pytest.raises(ValueError):
        audit_all(account, funding, history)


def test_default_and_explicit_false_preserve_the_entire_existing_account_path():
    default = Account("default", 2, 3, [], take_profit_fraction=None)
    explicit = Account("default", 2, 3, [], take_profit_fraction=None, timed_exit_at_close=False)
    first, second = complete(default), complete(explicit)
    assert vars(default) == vars(explicit)
    assert default.trades[0]["reason"] == "time"
    assert default.trades[0]["exit_ms"] == 4 * MINUTE
    assert default.trades[0]["exit_price"] == pytest.approx(100 * (1 - 3 / 10_000))
    assert audit_all(default, [], first)["passed"]
    assert audit_all(explicit, [], second)["passed"]
