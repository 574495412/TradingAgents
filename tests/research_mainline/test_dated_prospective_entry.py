"""Failure and economic-contract tests; no market data or profitability claims."""

import copy
import importlib.util
from decimal import Decimal as D
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT.parent / "freqtrade/research/crypto/dated_forward_paper_v1.py"
spec = importlib.util.spec_from_file_location("dated_forward_test", ENGINE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def config():
    return {
        "capital": "10000",
        "future_wallet_fraction": "0.70",
        "spot_fraction": "0.25",
        "asset_count": "2",
        "spot_base_fee_fraction": "0.001",
        "spot_exit_fee_fraction": "0.001",
        "future_fee_fraction": "0.0005",
        "settlement_fee_fraction": "0.0005",
        "adverse_price_fraction": "0.001",
        "maximum_book_walk_fraction": "0.001",
        "depth_fraction": "0.01",
        "terminal_price_cost_stress_multiple": "3",
        "minimum_useful_pnl_per_asset_usdt": "5",
    }


def symbol(step, *, spot=False):
    return {
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": step, "maxQty": "100000", "stepSize": step},
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0" if spot else step,
                "maxQty": "100000",
                "stepSize": "0" if spot else step,
            },
            {
                "filterType": "NOTIONAL",
                "minNotional": "5",
                "maxNotional": "1000000",
                "applyMinToMarket": True,
                "applyMaxToMarket": False,
            }
            if spot
            else {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ]
    }


def book(bid, ask, quantity="100000"):
    return {"bids": [[str(bid), quantity]], "asks": [[str(ask), quantity]]}


def observation():
    phase = {"spot": book("99.99", "100"), "future": book("105", "105.01")}
    return {
        "symbols": {"spot": symbol("0.00001", spot=True), "future": symbol("0.001")},
        "decision": copy.deepcopy(phase),
        "arrival": copy.deepcopy(phase),
        "unwind": copy.deepcopy(phase),
        "mark": copy.deepcopy(phase),
        "decision_ns": 100,
        "arrival_ns": 200,
        "unwind_ns": 300,
        "mark_ns": 400,
        "public_inputs_valid": True,
        "book_valid": {
            phase: {"spot": True, "future": True}
            for phase in ["decision", "arrival", "unwind", "mark"]
        },
        "spot_notional_reference": "100",
        "spot_execution_reference": None,
        "future_notional_reference": "105",
        "execution_rules": [],
    }


def test_zero_market_step_does_not_disable_reported_lot_grid():
    s = symbol("0.00001", spot=True)
    assert engine.quantity_ok("0.00002", s)
    assert not engine.quantity_ok("0.000021", s)
    assert not engine.notional_ok("0.01", s, "100")
    with pytest.raises(ValueError):
        engine.atoms("0.000000001")


def test_absent_reference_does_not_turn_average_into_execution_reference():
    rules = [
        {
            "ruleType": "PRICE_RANGE",
            "bidLimitMultDown": "0.99",
            "bidLimitMultUp": "1.01",
            "askLimitMultDown": "0.99",
            "askLimitMultUp": "1.01",
        }
    ]
    raw = book("99", "100")
    # A missing execution reference disables PRICE_RANGE; NOTIONAL still uses100.
    assert (
        engine.modeled_fill(raw, "buy", "1", symbol("0.001", spot=True), config(), "100", rules)[
            "quantity"
        ]
        == 1
    )
    assert (
        engine.modeled_fill(
            raw,
            "buy",
            "1",
            symbol("0.001", spot=True),
            config(),
            "100",
            rules,
            execution_reference="200",
        )["quantity"]
        == 0
    )


def test_target_covers_base_commission_without_increasing_budget():
    o = observation()
    t = engine.target(
        o["decision"]["spot"],
        o["decision"]["future"],
        o["symbols"]["spot"],
        o["symbols"]["future"],
        config(),
    )
    net = t["gross"] * D("0.999")
    assert net >= t["short"]
    assert net - t["short"] < D("0.00001")
    assert t["gross"] * D("100.1") <= D("1250")
    assert t["short"] % D("0.001") == 0


def test_costs_can_refute_positive_raw_basis():
    cfg = config()
    # Positive0.2USDT raw basis, but terminal + entry costs exceed it.
    x = engine.economic_screen(
        "1",
        "0.999",
        {"quantity": D("1"), "quote": D("100")},
        {"quantity": D("0.999"), "quote": D("100.2")},
        cfg,
    )
    assert x["gross_basis_cash"] > 0
    assert x["net_terminal_scenario"] < 0
    assert not x["eligible"]


def test_leg_depth_failure_does_not_borrow_other_leg_fill_or_erase_dust():
    o = observation()
    o["arrival"]["future"] = book("105", "105.01", "0")
    s = engine.run({"BTC": o}, config())
    p = s["positions"]["BTC"]
    assert p["short_atoms"] == 0
    assert p["spot_atoms"] > 0  # unliquidatable rounding remains owned
    assert p["state"] == "failed_entry_owned_inventory"
    assert p["dust_reserve"] > 0
    assert s["complete_lifecycles"] == 0 and s["win_rate"] is None
    assert s["quote_cash"] + s["future_cash"] < D("10000")
    assert any(e["type"] == "paper_emergency" for e in s["events"])


def test_unknown_public_constraints_prevent_orders_even_with_positive_screen():
    o = observation()
    o["public_inputs_valid"] = False
    s = engine.run({"BTC": o}, config())
    assert s["decisions"][0]["screen"]["eligible"]
    assert s["positions"] == {}
    assert s["quote_cash"] + s["future_cash"] == D("10000")
    assert [e["type"] for e in s["events"]] == ["stay_cash"]


def test_later_mark_missing_does_not_retroactively_change_entry():
    o = observation()
    first = engine.run({"BTC": o}, config())
    later = copy.deepcopy(o)
    later["book_valid"]["mark"]["future"] = False
    second = engine.run({"BTC": later}, config())
    assert first["decisions"] == second["decisions"]
    assert first["events"] == second["events"]
    assert first["positions"]["BTC"]["short_atoms"] == second["positions"]["BTC"]["short_atoms"]
    assert second["observations"][-1]["equity_after_exit_reserve"] is None
