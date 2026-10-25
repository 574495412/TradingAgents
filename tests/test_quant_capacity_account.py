"""合成验证离散容量批次、部分退出和原cycle现金守恒，不读取真实行情。"""

from copy import deepcopy
from decimal import Decimal as D

import pytest

from tradingagents.research.quant.backtesting.capacity_account import CapacityAccount, QuantityRule
from tradingagents.research.quant.backtesting.signals import MINUTE, PROTOCOL

FIRST = 100 * MINUTE
PAIR = [("L", 1, 0.003), ("S", -1, 0.003)]
FOUR = [("L", 1, 0.003), ("L2", 1, 0.003), ("S", -1, 0.003), ("S2", -1, 0.003)]


def rule(*, step="0.1", minimum="0.1", maximum="100", notional="5", other=None):
    """两条规则保留真实minQty偏移，不借显示精度代替步长。"""
    first = {
        "filter_type": "LOT_SIZE",
        "min_qty": minimum,
        "max_qty": maximum,
        "step_size": step,
        "offset": minimum,
    }
    second = {**first, "filter_type": "MARKET_LOT_SIZE", **(other or {})}
    return {
        "symbol": "合成",
        "rule_known": True,
        "unknown_reason": None,
        "lots": [first, second],
        "min_notional": notional,
        "source_snapshot_sha256": "0" * 64,
        "historical_pti_proven": False,
        "min_notional_uses_raw_price_proxy": True,
        "market_average_fill_requires_tick_grid": False,
    }


def account(*, hold=3, length=20, slip=0, funding=(), rules=None):
    return CapacityAccount(
        rules or {symbol: rule() for symbol in ("L", "L2", "S", "S2")},
        funding,
        slippage_bps=slip,
        holding_minutes=hold,
        end_ms=FIRST + length * MINUTE,
        factor="合成模型",
    )


def market(clock, *, budget=1000, decision=1000, prices=None):
    return {
        symbol: {
            "open": float((prices or {}).get(symbol, 100)),
            "close": float((prices or {}).get(symbol, 100)),
            "observed": True,
            "qualified": True,
            "open_budget": budget,
            "decision_budget": decision,
            "open_budget_asof_ms": clock - MINUTE,
            "decision_budget_asof_ms": clock,
        }
        for symbol in ("L", "L2", "S", "S2")
    }


def run(book, *, selected=PAIR, source=None, propose_every=False):
    for clock in range(FIRST, book.end_ms + MINUTE, MINUTE):
        frame = source(clock) if source else market(clock)
        book.step(clock, frame, selected if clock == FIRST or propose_every else [])
    return book


def event(stamp, symbol="L", *, kind="Regular", rate="0.001", mark="100"):
    return {"time_ms": stamp, "symbol": symbol, "rate_type": kind, "rate": rate, "mark_price": mark}


def assert_conserved(book):
    """另从公开fill/fund重算现金，不调用内核的账户权益方法。"""
    cash = D(str(PROTOCOL["initial_equity"]))
    entries = {}
    for fill in book.fills:
        fee = fill["quantity"] * fill["price"] * D(str(PROTOCOL["fee_bps"])) / 10000
        assert fee == fill["fee"]
        cash -= fee
        if fill["action"] == "OPEN":
            entries[fill["cycle_id"]] = fill["price"]
        else:
            gross = (
                D(fill["direction"])
                * fill["quantity"]
                * (fill["price"] - entries[fill["cycle_id"]])
            )
            assert gross == fill["gross_pnl"]
            cash += gross
    for row in book.funding_ledger:
        fund = -D(row["direction"]) * row["quantity"] * row["mark_price"] * D(str(row["rate"]))
        assert fund == row["cash"]
        cash += fund
    assert cash == book.cash
    for trade in book.trades:
        reductions = [book.fills[index] for index in trade["reduction_fill_indices"]]
        assert sum((row["quantity"] for row in reductions), D(0)) == trade["quantity"]
        assert sum((row["fee"] for row in reductions), D(0)) == trade["exit_fee"]
        assert sum((row["gross_pnl"] for row in reductions), D(0)) == trade["gross_pnl"]
        assert (
            trade["net_pnl"]
            == trade["gross_pnl"] - trade["entry_fee"] - trade["exit_fee"] + trade["funding"]
        )


def test_crt_intersection_preserves_both_nonzero_offsets_and_floors():
    quantity = QuantityRule(
        rule(
            step="0.04",
            minimum="0.03",
            other={"min_qty": "0.05", "offset": "0.05", "step_size": "0.06"},
        )
    )
    assert quantity.known and quantity.step == D("0.12") and quantity.offset == D("0.11")
    assert quantity.floor(D("0.50")) == D("0.47")
    assert quantity.floor(D("0.10")) == 0
    assert (quantity.floor(D("0.50")) - D("0.03")) % D("0.04") == 0
    assert (quantity.floor(D("0.50")) - D("0.05")) % D("0.06") == 0


def test_incompatible_lot_offsets_remain_unknown():
    quantity = QuantityRule(
        rule(
            step="0.04",
            minimum="0.02",
            other={"min_qty": "0.03", "offset": "0.03", "step_size": "0.06"},
        )
    )
    assert not quantity.known and quantity.floor(100) == 0


def test_decimal_sub_satoshi_lattice_matches_integer_enumeration():
    quantity = QuantityRule(
        rule(
            step="0.00000003",
            minimum="0.00000002",
            other={"min_qty": "0.00000005", "offset": "0.00000005", "step_size": "0.00000007"},
        )
    )
    expected = max(n for n in range(1, 101) if (n - 2) % 3 == 0 and (n - 5) % 7 == 0)
    assert quantity.floor(D("0.000001")) == D(expected) / 100_000_000


def test_minimum_is_not_raised_and_two_maximums_are_both_enforced():
    quantity = QuantityRule(rule(minimum="0.2", maximum="5", other={"max_qty": "0.6"}))
    assert quantity.floor(D("0.19")) == 0
    assert quantity.floor(10) == D("0.6")


def test_entry_delay_uses_one_shared_pre_batch_equity_for_all_four_legs():
    book = account()
    assert book.step(FIRST, market(FIRST), FOUR)
    assert book.pending["available_ms"] == FIRST + MINUTE
    assert book.pending["expires_ms"] == FIRST + 5 * MINUTE
    book.step(FIRST + MINUTE, market(FIRST + MINUTE), [])
    assert not book.fills
    book.step(FIRST + 2 * MINUTE, market(FIRST + 2 * MINUTE), [])
    assert len(book.fills) == 4
    assert {row["clock"] for row in book.fills} == {FIRST + MINUTE}
    assert {row["processing_clock_ms"] for row in book.fills} == {FIRST + 2 * MINUTE}
    assert {row["pre_batch_equity"] for row in book.fills} == {D(1000)}
    assert {row["quantity"] for row in book.fills} == {D("1.2")}
    assert book.cash == D("999.76")


def test_tightest_leg_capacity_and_market_max_shrink_entire_batch():
    rules = {symbol: rule() for symbol, _, _ in FOUR}
    rules["S2"] = rule(other={"max_qty": "0.4"})
    book = account(rules=rules)

    def source(clock):
        frame = market(clock)
        frame["L"]["decision_budget"] = 60.0
        return frame

    run(book, selected=FOUR, source=source)
    opens = [row for row in book.fills if row["action"] == "OPEN"]
    assert len(opens) == 4 and {row["quantity"] for row in opens} == {D("0.4")}
    assert all(
        row["capacity_charge"] <= min(row["frozen_decision_budget"], row["current_open_budget"])
        for row in opens
    )
    assert_conserved(book)


def test_shared_capital_proxy_cannot_be_exceeded_even_under_extreme_synthetic_slip():
    book = account(slip=9000)
    for clock in range(FIRST, FIRST + 3 * MINUTE, MINUTE):
        book.step(clock, market(clock), PAIR if clock == FIRST else [])
    opens = [row for row in book.fills if row["action"] == "OPEN"]
    assert len(opens) == 2
    assert sum((row["capacity_charge"] + row["fee"] for row in opens), D(0)) <= D(1000)
    plan = next(row for row in book.entry_plans if row["event"] == "ENTRY_ATTEMPT")
    assert plan["unscaled_shared_capital_proxy"] > 1000 and plan["common_scale"] < 1


@pytest.mark.parametrize(
    "frozen,current,expected", [(50, 1000, "0.5"), (1000, 50, "0.5"), (1000, 1000, "1.2")]
)
def test_frozen_decision_and_current_open_budgets_take_the_minimum(frozen, current, expected):
    book = account()
    run(book, source=lambda clock: market(clock, decision=frozen, budget=current))
    assert {row["quantity"] for row in book.fills if row["action"] == "OPEN"} == {D(expected)}


@pytest.mark.parametrize("bad", [None, 0, 5])
def test_any_unknown_zero_or_below_project_minimum_cancels_whole_batch_once(bad):
    book = account()

    def source(clock):
        frame = market(clock)
        if clock == FIRST:
            frame["S"]["decision_budget"] = bad
        return frame

    run(book, source=source)
    assert not book.fills and book.pending is None
    attempts = [row for row in book.entry_plans if row["event"] == "ENTRY_ATTEMPT"]
    assert len(attempts) == 1 and attempts[0]["status"] == "ZERO_BATCH"
    assert attempts[0]["legs"]
    assert len(book.curve) == 21 and book.metrics()["closed_trades"] == 0


def test_unknown_rule_and_one_bad_entry_open_cancel_all_legs_without_replacement():
    for bad in ("rule", "open", "qualification"):
        rules = {symbol: rule() for symbol in ("L", "S")}
        if bad == "rule":
            rules["S"] = {"rule_known": False, "unknown_reason": "无快照", "lots": []}
        book = account(rules=rules)

        def source(clock, bad=bad):
            frame = market(clock)
            if clock == FIRST + 2 * MINUTE:
                if bad == "open":
                    frame["S"]["open"] = float("nan")
                if bad == "qualification":
                    frame["S"]["qualified"] = False
            return frame

        run(book, source=source)
        assert not book.fills and book.invalid is None


def test_rounding_dollar_residual_is_recorded_without_enlarging_any_leg():
    book = account()
    run(book, source=lambda clock: market(clock, prices={"L": 97, "S": 103}))
    plan = next(row for row in book.entry_plans if row["event"] == "ENTRY_ATTEMPT")
    assert plan["status"] == "FILLED" and plan["signed_fill_notional_residual"] != 0
    assert plan["signed_fill_notional_residual"] == sum(
        (D(row["direction"]) * row["quantity"] * row["price"] for row in plan["legs"]), D(0)
    )
    assert all(row["quantity"] <= row["desired_quantity"] for row in plan["legs"])


def test_partial_exits_keep_cycle_and_count_only_two_completed_legs():
    book = account()

    def source(clock):
        return market(clock, budget=30 if clock >= FIRST + 5 * MINUTE else 1000)

    run(book, source=source)
    assert book.invalid is None and not book.positions
    assert len(book.reduction_fills) == 8 and len(book.trades) == 2
    assert sum(row["action"] == "REDUCE" for row in book.fills) == 6
    assert sum(row["action"] == "CLOSE" for row in book.fills) == 2
    assert {row["first_exit_ms"] for row in book.trades} == {FIRST + 4 * MINUTE}
    assert {row["exit_ms"] for row in book.trades} == {FIRST + 7 * MINUTE}
    assert book.metrics()["closed_trades"] == 2
    assert all(
        row["entry_fee"] == D("0.06") and row["exit_fee"] == D("0.06") for row in book.trades
    )
    assert_conserved(book)


def test_independent_nonzero_partial_price_pnl_fees_and_funding_totals():
    funds = [
        event(FIRST + 6 * MINUTE, rate="-0.01"),
        event(FIRST + 6 * MINUTE, symbol="S", kind="Special", rate="-0.02"),
    ]
    book = account(funding=funds)

    def source(clock):
        increment = max(0, (clock - FIRST) // MINUTE - 4)
        return market(
            clock,
            budget=30 if increment else 1000,
            prices={"L": 100 + increment, "S": 100 - increment},
        )

    run(book, source=source)
    trades = {row["symbol"]: row for row in book.trades}
    assert trades["L"]["gross_pnl"] == D("4.2")
    assert trades["S"]["gross_pnl"] == D("3.0")
    assert trades["L"]["funding"] == D("0.8")
    assert trades["S"]["funding"] == D("-1.2")
    assert trades["L"]["net_pnl"] == D("4.8779")
    assert trades["S"]["net_pnl"] == D("1.6815")
    assert len(book.reduction_fills) == 10 and book.metrics()["closed_trades"] == 2
    assert_conserved(book)


def test_funding_same_clock_uses_old_remaining_quantity_and_special_is_kept():
    entry, reduce = FIRST + MINUTE, FIRST + 4 * MINUTE
    funds = [
        event(entry),
        event(entry + 1),
        event(reduce),
        event(reduce, kind="Special", rate="-0.002"),
        event(reduce + 1),
        event(reduce + MINUTE),
    ]
    book = account(funding=funds)
    run(
        book, source=lambda clock: market(clock, budget=30 if clock >= FIRST + 5 * MINUTE else 1000)
    )
    by_clock = {}
    for row in book.funding_ledger:
        by_clock.setdefault(row["time_ms"], []).append(row)
    assert entry not in by_clock
    assert by_clock[entry + 1][0]["quantity"] == D("1.2")
    assert {row["quantity"] for row in by_clock[reduce]} == {D("1.2")}
    assert {row["rate_type"] for row in by_clock[reduce]} == {"Regular", "Special"}
    assert by_clock[reduce + 1][0]["quantity"] == D("0.9")
    assert by_clock[reduce + MINUTE][0]["quantity"] == D("0.9")
    assert_conserved(book)


def test_final_full_exit_includes_same_clock_cash_but_end_close_does_not_settle_same_clock():
    final_exit = FIRST + 4 * MINUTE
    book = account(funding=[event(final_exit)], length=6)
    run(book)
    assert book.funding_ledger[0]["time_ms"] == final_exit
    assert book.funding_ledger[0]["quantity"] == D("1.2")
    residual = account(hold=240, length=6, funding=[event(FIRST + 6 * MINUTE)])
    run(residual)
    assert residual.invalid["type"] == "RESIDUAL_INVENTORY_AT_END"
    assert not residual.funding_ledger and residual.positions


@pytest.mark.parametrize("budget", [None, 0])
def test_unknown_or_zero_exit_budget_keeps_inventory_and_funding_until_end(budget):
    book = account(length=10, funding=[event(FIRST + 9 * MINUTE, rate="-0.01")])
    run(
        book,
        source=lambda clock: market(clock, budget=budget if clock >= FIRST + 5 * MINUTE else 1000),
    )
    assert book.invalid["type"] == "RESIDUAL_INVENTORY_AT_END" and book.finished
    assert len(book.positions) == 2 and not book.trades and not book.reduction_fills
    assert book.funding_ledger and book.metrics() is None
    assert all(row["quantity"] == D("1.2") for row in book.positions.values())
    assert_conserved(book)


def test_future_price_drop_can_still_make_reserved_inventory_invalid_at_end():
    rules = {symbol: rule(step="0.05", minimum="0.05", notional="30") for symbol in ("L", "S")}
    book = account(rules=rules, length=12)

    def source(clock):
        return market(
            clock,
            budget=100 if clock >= FIRST + 5 * MINUTE else 1000,
            prices={"L": 50, "S": 50} if clock >= FIRST + 6 * MINUTE else None,
        )

    run(book, source=source)
    assert not book.trades and len(book.reduction_fills) == 2
    assert {p["quantity"] for p in book.positions.values()} == {D("0.30")}
    assert book.invalid["type"] == "RESIDUAL_INVENTORY_AT_END" and book.metrics() is None
    assert_conserved(book)


def test_partial_exit_reserves_a_legal_project_ten_remaining_instead_of_five_dust():
    rules = {symbol: rule(step="0.05", minimum="0.05", notional="5") for symbol in ("L", "S")}
    book = account(rules=rules, length=8)
    run(
        book,
        source=lambda clock: market(clock, budget=120 if clock >= FIRST + 5 * MINUTE else 1000),
    )
    first = [row for row in book.reduction_fills if row["clock"] == FIRST + 4 * MINUTE]
    assert len(first) == 2
    assert {row["capacity_quantity_floor"] for row in first} == {D("1.20")}
    assert {row["quantity"] for row in first} == {D("1.15")}
    assert {row["qty_remaining"] for row in first} == {D("0.10")}
    assert {row["dust_avoidance_quantity_reduction"] for row in first} == {D("0.05")}
    assert book.invalid is None and len(book.trades) == 2 and not book.positions
    assert_conserved(book)


def test_nonzero_affine_lot_origin_waits_for_full_capacity_without_illegal_split():
    shape = rule(
        step="0.04",
        minimum="0.03",
        other={"min_qty": "0.05", "offset": "0.05", "step_size": "0.06"},
    )
    book = account(rules={symbol: deepcopy(shape) for symbol in ("L", "S")}, length=10)

    def source(clock):
        return market(
            clock, budget=50 if FIRST + 5 * MINUTE <= clock < FIRST + 7 * MINUTE else 1000
        )

    run(book, source=source)
    assert not [row for row in book.fills if row["action"] == "REDUCE"]
    assert len(book.trades) == 2 and not book.invalid
    assert {row["quantity"] for row in book.trades} == {D("1.19")}
    assert {row["exit_ms"] for row in book.trades} == {FIRST + 6 * MINUTE}
    assert any(row["reason"] == "exit_budget_or_minimum_wait" for row in book.declines)
    assert_conserved(book)


def test_240_minute_open_target_is_not_shortened_at_end():
    book = account(hold=240, length=243)
    run(book)
    assert {row["entry_ms"] for row in book.trades} == {FIRST + MINUTE}
    assert {row["exit_ms"] for row in book.trades} == {FIRST + 241 * MINUTE}
    assert book.metrics()["closed_trades"] == 2
    short = run(account(hold=240, length=30))
    assert short.invalid and not short.trades and short.positions
    assert not [row for row in short.fills if row["action"] != "OPEN"]


def test_last_eligible_entry_is_not_avoided_using_the_future_end_boundary():
    book = account(hold=240, length=6)
    for clock in range(FIRST, book.end_ms + MINUTE, MINUTE):
        book.step(clock, market(clock), PAIR if clock == book.end_ms - 2 * MINUTE else [])
    assert len(book.fills) == 2
    assert {row["processing_clock_ms"] for row in book.fills} == {book.end_ms}
    assert book.invalid["type"] == "RESIDUAL_INVENTORY_AT_END" and book.positions


def test_zero_rate_is_retained_as_an_explicit_typed_cash_event():
    book = run(account(funding=[event(FIRST + MINUTE + 1, kind="Special", rate="0")]))
    assert len(book.funding_ledger) == 1
    assert book.funding_ledger[0]["rate_type"] == "Special"
    assert book.funding_ledger[0]["cash"] == 0
    assert_conserved(book)


def test_risk_halt_is_permanent_and_partial_exits_retry_after_signal_expiry():
    book = account(hold=240, length=20)

    def source(clock):
        frame = market(clock, budget=30 if clock >= FIRST + 4 * MINUTE else 1000)
        if clock == FIRST + 2 * MINUTE:
            frame["L"]["close"] = 10.0
        return frame

    run(book, source=source, propose_every=True)
    assert book.halted and not book.positions and not book.invalid
    assert len(book.trades) == 2 and len(book.reduction_fills) == 8
    assert {row["reason"] for row in book.trades} == {"risk"}
    assert max(row["exit_ms"] for row in book.trades) > FIRST + 5 * MINUTE
    assert len([row for row in book.fills if row["action"] == "OPEN"]) == 2


def test_open_risk_collision_cannot_delay_an_already_mature_time_exit():
    book = account(hold=3, length=10)

    def source(clock):
        return market(clock, prices={"L": 10} if clock >= FIRST + 5 * MINUTE else None)

    run(book, source=source)
    assert book.halted and book.invalid is None
    assert {row["entry_ms"] for row in book.trades} == {FIRST + MINUTE}
    assert {row["exit_ms"] for row in book.trades} == {FIRST + 4 * MINUTE}
    assert {row["reason"] for row in book.trades} == {"time"}
    assert all(
        row["exit_request"]["superseded_exit_request"]
        == {"available_ms": FIRST + 5 * MINUTE, "reason": "risk"}
        for row in book.reduction_fills
    )
    assert {row["processing_clock_ms"] for row in book.reduction_fills} == {FIRST + 5 * MINUTE}
    assert_conserved(book)


def test_risk_time_collision_keeps_earliest_time_request_for_partial_inventory():
    book = account(hold=3, length=7)
    for clock in range(FIRST, FIRST + 6 * MINUTE, MINUTE):
        frame = market(clock, budget=30 if clock >= FIRST + 5 * MINUTE else 1000)
        if clock >= FIRST + 5 * MINUTE:
            frame["L"]["open"] = frame["L"]["close"] = 10.0
        book.step(clock, frame, PAIR if clock == FIRST else [])
    assert book.halted and "S" in book.positions
    request = book.closes["S"]
    assert request["available_ms"] == FIRST + 4 * MINUTE and request["reason"] == "time"
    assert request["superseded_exit_request"] == {
        "available_ms": FIRST + 5 * MINUTE,
        "reason": "risk",
    }
    assert any(
        row["symbol"] == "S" and row["clock"] == FIRST + 4 * MINUTE for row in book.reduction_fills
    )


def test_missing_held_price_or_qualification_preserves_remaining_inventory():
    for bad in ("price", "qualification"):
        book = account()
        for clock in range(FIRST, FIRST + 6 * MINUTE, MINUTE):
            frame = market(clock)
            if clock == FIRST + 3 * MINUTE:
                frame["L"]["open" if bad == "price" else "qualified"] = (
                    float("nan") if bad == "price" else False
                )
            book.step(clock, frame, PAIR if clock == FIRST else [])
        assert book.invalid and len(book.positions) == 2 and not book.trades
        assert book.metrics() is None and len(book.fills) == 2


@pytest.mark.parametrize("phase", ["open", "decision"])
def test_future_budget_identity_is_rejected_when_the_phase_consumes_it(phase):
    book = account()
    for clock in range(FIRST, FIRST + 3 * MINUTE, MINUTE):
        frame = market(clock)
        if phase == "decision" and clock == FIRST:
            frame["S"]["decision_budget_asof_ms"] = clock + 1
        if phase == "open" and clock == FIRST + 2 * MINUTE:
            frame["S"]["open_budget_asof_ms"] = clock
        book.step(clock, frame, PAIR if clock == FIRST else [])
    assert book.invalid and "预算引用" in book.invalid["reason"] and not book.fills


def test_future_close_and_trade_minute_quote_do_not_change_entry_quantity():
    books = [account(hold=240, length=5) for _ in range(2)]
    for index, book in enumerate(books):
        for clock in range(FIRST, FIRST + 3 * MINUTE, MINUTE):
            frame = market(clock)
            for row in frame.values():
                row["quote_volume"] = 0 if index == 0 else 1e100
                if clock == FIRST + 2 * MINUTE:
                    row["close"] = 100.0 if index == 0 else 101.0
            book.step(clock, frame, PAIR if clock == FIRST else [])
    assert books[0].fills == books[1].fills


def test_missing_future_close_does_not_prevent_real_open_then_fails_with_inventory():
    book = account()
    for clock in range(FIRST, FIRST + 3 * MINUTE, MINUTE):
        frame = market(clock)
        if clock == FIRST + 2 * MINUTE:
            frame["L"]["close"] = float("nan")
        book.step(clock, frame, PAIR if clock == FIRST else [])
    assert len(book.fills) == 2 and len(book.positions) == 2
    assert book.invalid and not book.trades and book.metrics() is None


def test_each_symbol_budget_is_shared_by_all_actions_and_fill_evidence_is_complete():
    book = run(
        account(),
        source=lambda clock: market(clock, budget=30 if clock >= FIRST + 5 * MINUTE else 1000),
    )
    usage = {}
    for fill in book.fills:
        key = fill["symbol"], fill["processing_clock_ms"]
        usage[key] = usage.get(key, D(0)) + fill["capacity_charge"]
        assert fill["phase"] == "open" and fill["processing_clock_ms"] == fill["clock"] + MINUTE
        assert {
            "cycle_id",
            "quantity_before",
            "qty_remaining",
            "rule",
            "historical_mark_used",
        } <= set(fill)
        assert not fill["historical_mark_used"]
    assert all(
        value <= (D(30) if clock >= FIRST + 5 * MINUTE else D(1000))
        for (_, clock), value in usage.items()
    )


def test_market_average_fill_is_not_rounded_to_tick_and_project_ten_is_enforced():
    book = run(account(slip=3))
    assert {row["price"] for row in book.fills if row["action"] == "OPEN"} == {
        D("100.0300"),
        D("99.9700"),
    }
    tiny = account(
        rules={symbol: rule(step="0.01", minimum="0.01", notional="5") for symbol in ("L", "S")}
    )
    run(tiny, source=lambda clock: market(clock, decision=9))
    assert not tiny.fills and tiny.metrics()["closed_trades"] == 0


def test_all_state_is_continuous_across_external_refit_boundaries():
    book = account(hold=12, length=20)
    states = []
    for clock in range(FIRST, book.end_ms + MINUTE, MINUTE):
        book.step(clock, market(clock), PAIR if clock == FIRST else [])
        if clock in {FIRST + 3 * MINUTE, FIRST + 8 * MINUTE}:
            states.append((book.cash, deepcopy(book.positions)))
    assert states[0] == states[1]
    assert book.metrics()["closed_trades"] == 2


@pytest.mark.parametrize("bad", ["unsorted", "implicit", "negative_time"])
def test_funding_requires_explicit_types_and_original_sorted_integer_clock(bad):
    events = [event(FIRST), event(FIRST + 1)]
    if bad == "unsorted":
        events.reverse()
    elif bad == "implicit":
        del events[0]["rate_type"]
    else:
        events[0]["time_ms"] = -1
    with pytest.raises(ValueError):
        account(funding=events)


def test_minute_gap_invalidates_without_restarting_or_reusing_a_budget():
    book = account()
    book.step(FIRST, market(FIRST), PAIR)
    book.step(FIRST + 2 * MINUTE, market(FIRST + 2 * MINUTE), [])
    assert book.invalid and not book.fills
    assert not book.step(FIRST + 3 * MINUTE, market(FIRST + 3 * MINUTE), PAIR)
