import json

import numpy as np
import pandas as pd
import pytest

from research_tools.leader_lab.damorne_research import (
    ContractBook,
    account,
    make_candidates,
    reconstruct_orders,
    score,
)


def config():
    return make_candidates(
        {"train_small": {"gross_price_return": {"0.5": 0.0001}}},
        np.zeros((30, 8)),
        np.zeros(30),
    )[0]


def metadata():
    return {"XUSDT": {"step": 0.01, "min_qty": 0.01, "max_qty": 100000, "min_notional": 5}}


def frame():
    index = np.arange(20) * 60000
    return pd.DataFrame(
        {
            "open": 100.0,
            "mark": 100.0,
            "funding": 0.0,
            "return5": 0.001,
            "return15": -0.002,
            "return60": -0.01,
            "range_location": 0.1,
            "atr": 0.001,
            "volume_ratio": 1.0,
            "flow": 0.2,
            "quote60": 2_000_000.0,
        },
        index=index,
    )


def test_hedge_add_trim_whole_position_and_cash():
    book = ContractBook(config(), metadata())
    marks = {"XUSDT": 100.0}
    book.transact("OPEN", "XUSDT", 1, 100, 0, marks)
    book.transact("OPEN", "XUSDT", -1, 100, 0, marks)
    assert len(book.positions) == 2
    book.transact("ADD", "XUSDT", 1, 101, 60000, {"XUSDT": 101.0})
    book.transact("TRIM", "XUSDT", 1, 102, 120000, {"XUSDT": 102.0})
    assert not book.closed
    assert book.positions[("XUSDT", 1)]["adds"] == 1
    book.transact("CLOSE", "XUSDT", 1, 103, 180000, {"XUSDT": 103.0})
    assert len(book.closed) == 1
    book.curve = [{"timestamp": 0, "equity": book.equity({"XUSDT": 103.0})}]
    result = score(book, {"XUSDT": 103.0})
    assert result["whole_position_cash_reconciled"]
    assert result["complete_positions"] == 1
    assert result["terminal_open_positions"] == 1


def test_shared_margin_includes_opposite_side_and_fees():
    c = config()
    c["capital"] = 101
    c["max_margin_fraction"] = 1
    book = ContractBook(c, metadata())
    book.transact("OPEN", "XUSDT", 1, 100, 0, {"XUSDT": 100})
    book.transact("OPEN", "XUSDT", -1, 100, 0, {"XUSDT": 100})
    assert len(book.positions) == 2
    book.transact("ADD", "XUSDT", 1, 100, 0, {"XUSDT": 100})
    assert book.rejections["shared_margin_cap"] == 1


def test_next_minute_delay_funding_and_restart():
    f = frame()
    f.loc[180000, "funding"] = 0.001
    c = config()
    whole, state = account(c, {"XUSDT": f}, metadata(), 0, 20 * 60000)
    assert state["positions"][0]["opened_ms"] == 120000
    assert whole["funding"] < 0
    _, prefix = account(c, {"XUSDT": f}, metadata(), 0, 20 * 60000, break_ms=600000)
    resumed, tail = account(
        c, {"XUSDT": f}, metadata(), 600000, 20 * 60000, state=json.loads(json.dumps(prefix))
    )
    assert resumed == whole and tail == state


def test_future_bars_and_leader_columns_do_not_change_prefix():
    f = frame()
    altered = f.copy()
    altered.loc[600000:, "open"] = 200
    altered.loc[600000:, "return5"] = -0.1
    altered["leader_action"] = "OPEN_SHORT"
    altered["future_profit"] = 1e12
    _, first = account(config(), {"XUSDT": f}, metadata(), 0, 600000)
    _, second = account(config(), {"XUSDT": altered}, metadata(), 0, 600000)
    assert first == second


def order(when, side, quantity):
    return {
        "symbol": "XUSDT",
        "positionSide": "LONG",
        "side": side,
        "orderTime": when,
        "orderUpdateTime": when,
        "executedQty": quantity,
        "avgPrice": 100,
        "totalPnl": 0,
        "type": "LIMIT",
    }


def test_missing_window_never_manufactures_known_inventory():
    d = {
        "start_ms": 0,
        "unavailable_windows": [{"startTime": 20, "endTime": 30}],
        "rows": [order(10, "BUY", 1), order(40, "SELL", 1)],
    }
    events, closed, problems = reconstruct_orders(d)
    assert not closed and len(problems) == 1
    assert events[-1]["event"] == "CLOSE_UNKNOWN_INVENTORY"


def test_additions_reductions_and_unknown_close_remain_separate():
    d = {
        "start_ms": 0,
        "unavailable_windows": [],
        "rows": [
            order(1, "BUY", 1),
            order(2, "BUY", 1),
            order(3, "SELL", 0.5),
            order(4, "SELL", 1.5),
        ],
    }
    events, closed, problems = reconstruct_orders(d)
    assert [r["event"] for r in events] == ["OPEN_OR_UNKNOWN_REENTRY", "ADD", "REDUCE", "CLOSE"]
    assert len(closed) == 1 and not problems
    assert closed[0]["adds"] == closed[0]["reductions"] == 1


def test_candidate_cost_floor_not_claimed_as_leader_target():
    assert config()["small_target"] == 0.003
    assert config()["background_is_sampled_control_not_observed_hold"]
    assert not config()["production_eligible"]


def test_below_lot_open_rejected():
    meta = metadata()
    meta["XUSDT"]["min_notional"] = 1000
    book = ContractBook(config(), meta)
    book.transact("OPEN", "XUSDT", 1, 100, 0, {"XUSDT": 100})
    assert not book.positions and book.rejections["lot_or_notional"] == 1


def test_restart_after_numpy_trades_preserves_summary_exactly():
    f = pd.concat([frame()] * 50, ignore_index=True)
    f.index = np.arange(len(f)) * 60000
    f["open"] = 100 + np.sin(np.arange(len(f)) / 20)
    f["mark"] = f["open"]
    c = config()
    end, middle = len(f) * 60000, 501 * 60000
    whole, state = account(c, {"XUSDT": f}, metadata(), 0, end)
    _, prefix = account(c, {"XUSDT": f}, metadata(), 0, end, break_ms=middle)
    assert prefix["closed"]
    resumed, restored = account(
        c, {"XUSDT": f}, metadata(), middle, end, state=json.loads(json.dumps(prefix))
    )
    assert whole == resumed and state == restored


def test_paper_cannot_publish_backdated_forecast():
    from research_tools.leader_lab.damorne_paper import check_collection_deadline

    check_collection_deadline(60000, 119999)
    with pytest.raises(ValueError, match="backdated"):
        check_collection_deadline(60000, 120000)
    with pytest.raises(ValueError, match="backdated"):
        check_collection_deadline(60000, 59999)
