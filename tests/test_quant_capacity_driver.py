"""验证整批连续驱动、失败前缀和Decimal封存，不读取真实研究结果。"""

import gzip
import json
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from tradingagents.research.quant.backtesting import capacity as driver, extended_funding
from tradingagents.research.quant.backtesting.signals import MINUTE

FIRST = 100 * MINUTE


def rules():
    """两符号的已知合成网格，不借真实赢家定义案例。"""
    lots = [
        {
            "filter_type": kind,
            "min_qty": "0.01",
            "max_qty": "1000",
            "step_size": "0.01",
            "offset": "0.01",
        }
        for kind in ("LOT_SIZE", "MARKET_LOT_SIZE")
    ]
    return {
        symbol: {"rule_known": True, "lots": lots, "min_notional": "5"} for symbol in ("L", "S")
    }


def frames(end, *, skip=None, reduced_budget=False):
    """实际open预算和decision预算均显式给出阶段时钟。"""
    for clock in range(FIRST, end + MINUTE, MINUTE):
        if clock == skip:
            continue
        cap = 20.0 if reduced_budget and clock >= FIRST + 242 * MINUTE else 1000.0
        yield (
            clock,
            {
                symbol: {
                    "open": 100.0,
                    "close": 100.0,
                    "qualified": True,
                    "observed": True,
                    "open_budget": cap,
                    "decision_budget": cap,
                    "open_budget_asof_ms": clock - MINUTE,
                    "decision_budget_asof_ms": clock,
                }
                for symbol in ("L", "S")
            },
        )


def schedule(definitions):
    return {
        (method, asset, model, FIRST): [("L", 1, 0.003), ("S", -1, 0.003)]
        for method, asset, model, _ in definitions.values()
    }


def test_complete_48_cases_share_one_calendar_and_keep_cross_block_inventory():
    definitions = extended_funding.cases()
    end = FIRST + 245 * MINUTE
    bounds = (FIRST, FIRST + 120 * MINUTE, end)
    accounts, count, states, failure = driver.drive(
        definitions, schedule(definitions), frames(end), [], rules(), bounds
    )
    assert len(accounts) == 48 and count == 246 and failure is None
    for key, book in accounts.items():
        assert book.finished and book.invalid is None and book.metrics()["closed_trades"] == 2
        assert {row["entry_ms"] for row in book.trades} == {FIRST + MINUTE}
        assert {row["exit_ms"] for row in book.trades} == {FIRST + 241 * MINUTE}
        assert states[key][FIRST + 119 * MINUTE]["open_positions"] == 2
        blocks = driver.block_summaries(book, bounds, states[key])
        assert blocks[0]["closing_equity"] == blocks[1]["opening_equity"]
        assert sum(row["complete_legs"] for row in blocks) == 2
        assert float(sum(row["pnl_over_initial"] for row in blocks)) == pytest.approx(
            book.metrics()["net_return"]
        )


def test_global_gap_invalidates_every_case_without_losing_earlier_failure_prefix():
    definitions = extended_funding.cases()
    end = FIRST + 8 * MINUTE
    accounts, count, _, failure = driver.drive(
        definitions,
        schedule(definitions),
        frames(end, skip=FIRST + 5 * MINUTE),
        [],
        rules(),
        (FIRST, end),
    )
    assert count == 5 and failure["at"] == FIRST + 5 * MINUTE
    assert len(accounts) == 48 and all(book.metrics() is None for book in accounts.values())
    for book in accounts.values():
        prefix, suffix = driver.ledger_data(book)
        assert len(prefix["curve"]) == 5 and prefix["metrics"] is None
        assert prefix["fills"] and suffix["untrusted_ending_state"]["positions"]


def test_unknown_cash_trims_partial_unclosed_cycle_without_full_trade_reason(tmp_path, monkeypatch):
    definitions = extended_funding.cases()
    end = FIRST + 244 * MINUTE
    accounts, _, _, failure = driver.drive(
        definitions,
        schedule(definitions),
        frames(end, reduced_budget=True),
        [],
        rules(),
        (FIRST, end),
    )
    assert failure is None
    assert all(
        book.invalid and book.reduction_fills and not book.trades for book in accounts.values()
    )
    old = {key: book.invalid for key, book in accounts.items()}
    monkeypatch.setattr(driver.historical_panel, "MONTHS", ("合成",))
    monkeypatch.setattr(driver.historical_panel, "month_bounds", lambda _: (FIRST, end + MINUTE))
    driver.cash_failures(accounts, {("L", "合成"): False, ("S", "合成"): True})
    for key, book in accounts.items():
        assert book.invalid["at"] == FIRST + 2 * MINUTE < end
        assert book.invalid["prior_execution_failure"] == old[key]
        assert book.invalid["type"] == "UNKNOWN_FUNDING_COVERAGE"
    driver.save_ledgers(tmp_path, definitions, accounts, None)
    assert len(list(tmp_path.glob("*.json.gz"))) == 48
    for key in definitions:
        name = key.replace(":", "_") + ".json.gz"
        with gzip.open(tmp_path / name, "rt", encoding="utf-8") as stream:
            prefix = json.load(stream)
        with gzip.open(tmp_path / "counterfactual" / name, "rt", encoding="utf-8") as stream:
            suffix = json.load(stream)
        assert prefix["metrics"] is None and prefix["ending_state"] is None
        assert all(row["processing_clock_ms"] < FIRST + 2 * MINUTE for row in prefix["fills"])
        assert any(row["action"] == "REDUCE" for row in suffix["fills"])
        assert suffix["numeric_conclusions_allowed"] is False


@pytest.mark.parametrize(
    "end_delta,entry_delta,expected", [(0, -1, True), (-1, -2, False), (1, 0, True)]
)
def test_cash_month_boundary_is_old_inventory_inclusive_and_entry_exclusive(
    monkeypatch, end_delta, entry_delta, expected
):
    edge = 1000 * MINUTE
    monkeypatch.setattr(driver.historical_panel, "MONTHS", ("合成",))
    monkeypatch.setattr(
        driver.historical_panel, "month_bounds", lambda _: (edge, edge + 100 * MINUTE)
    )
    book = SimpleNamespace(
        invalid=None,
        last_clock=edge + 10 * MINUTE,
        trades=[{"symbol": "L", "entry_ms": edge + entry_delta, "exit_ms": edge + end_delta}],
        positions={},
    )
    driver.cash_failures({"case": book}, {("L", "合成"): False})
    assert (book.invalid is not None) is expected


def test_plain_preserves_decimal_and_rejects_nonfinite():
    assert driver.plain({"money": D("1.00000000000000000001"), "clock": 3, "known": False}) == {
        "money": "1.00000000000000000001",
        "clock": 3,
        "known": False,
    }
    for value in (D("NaN"), float("inf")):
        with pytest.raises(ValueError):
            driver.plain(value)


def test_preflight_rejects_output_inside_actual_dependency_before_writing(monkeypatch, tmp_path):
    source = tmp_path / "predictions"
    consumed = tmp_path / "consumed"
    source.mkdir()
    consumed.mkdir()
    leaf = consumed / "input.json"
    leaf.write_text("{}", encoding="utf-8")

    def bind(_, ids):
        ids[str(leaf)] = driver.sha(leaf)
        return {}, {}, None, None, None, None

    monkeypatch.setattr(driver.extended_funding, "_bind_sources", bind)
    monkeypatch.setattr(driver.capacity_sources, "load_rules", lambda *a, **k: ({}, {}))
    output = consumed / "must-not-be-written"
    with pytest.raises(ValueError, match="消费来源"):
        driver.run(source, output)
    assert not output.exists() and list(consumed.iterdir()) == [leaf]


def test_dependency_change_during_read_only_preflight_is_not_rebased(monkeypatch, tmp_path):
    dependency = tmp_path / "dependency.py"
    dependency.write_text("原始", encoding="utf-8")

    def bind(_, ids):
        dependency.write_text("改变", encoding="utf-8")
        return {}, {}, None, None, None, None

    monkeypatch.setattr(driver, "execution_sources", lambda: (dependency,))
    monkeypatch.setattr(driver.extended_funding, "_bind_sources", bind)
    monkeypatch.setattr(driver.capacity_sources, "load_rules", lambda *a, **k: ({}, {}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="依赖改变"):
        driver.run(tmp_path / "source", output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["rho", "minimum"])
def test_registered_thresholds_cannot_silently_drift(monkeypatch, kind):
    driver.registered_contract()
    if kind == "rho":
        monkeypatch.setattr(driver.capacity_sources, "RHO", D("0.01"))
    else:
        monkeypatch.setitem(driver.signals.PROTOCOL, "minimum_trade_notional", 5.0)
    with pytest.raises(ValueError, match="登记合同"):
        driver.registered_contract()
