"""验证公开证据桥接、时间因果、全成本记账及实验恢复。"""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from tradingagents.research.quant.comparison import experiment
from tradingagents.research.quant.comparison.account import Protocol, replay, validate_market
from tradingagents.research.quant.comparison.bridge import validate_response
from tradingagents.research.quant.comparison.common import digest, file_hash, read, save
from tradingagents.research.quant.comparison.dataset import HOUR, demo_dataset


@pytest.fixture
def flat_market():
    dataset = demo_dataset()
    dataset["bars"] = dataset["bars"][:7]
    for bar in dataset["bars"]:
        bar.update(open=100.0, high=101.0, low=99.0, close=100.0)
    first = dataset["bars"][0]["open_ms"]
    dataset["funding"] = [
        {"symbol": "DEMOUSDT", "time_ms": first + 2 * HOUR, "rate": 0.01, "mark_price": 100.0}
    ]
    return dataset


def entry(dataset, direction=1):
    clock = dataset["bars"][0]["open_ms"]
    return {
        "symbol": "DEMOUSDT",
        "as_of_ms": clock + 1,
        "available_ms": clock + 1000,
        "expires_ms": clock + 6 * HOUR,
        "direction": direction,
        "rank": 1.0,
    }


@pytest.mark.parametrize("direction,expected", [(1, -2.75), (-1, 2.25)])
def test_funding_sign_round_trip_costs_and_next_open(flat_market, direction, expected):
    protocol = Protocol(slippage_bps=0.0, max_hold_bars=2)
    result = replay(flat_market, [entry(flat_market, direction)], protocol)
    trade = result["trades"][0]
    assert trade["entry_ms"] == flat_market["bars"][1]["open_ms"]
    assert trade["exit_ms"] == flat_market["bars"][3]["open_ms"]
    assert trade["net_pnl"] == pytest.approx(expected)
    assert result["metrics"]["ending_equity"] == pytest.approx(1000 + expected)
    assert result["metrics"]["fees"] == pytest.approx(0.25)
    assert result["metrics"]["remaining_positions"] == 0


def test_boundary_funding_is_owned_by_old_position(flat_market):
    flat_market["funding"][0]["time_ms"] = flat_market["bars"][1]["open_ms"]
    result = replay(flat_market, [entry(flat_market)], Protocol(slippage_bps=0.0))
    assert result["funding_ledger"] == []
    flat_market["funding"][0]["time_ms"] = flat_market["bars"][3]["open_ms"]
    result = replay(flat_market, [entry(flat_market)], Protocol(slippage_bps=0.0, max_hold_bars=2))
    assert result["funding_ledger"][0]["amount"] == pytest.approx(-2.5)


def test_signal_cannot_reenter_forever_and_inputs_are_immutable(flat_market):
    decisions = [entry(flat_market)]
    before = deepcopy(decisions)
    a = replay(flat_market, decisions, Protocol(max_hold_bars=1))
    b = replay(flat_market, decisions, Protocol(max_hold_bars=1))
    assert decisions == before and a == b and len(a["trades"]) == 1


def test_future_close_does_not_change_prior_fill(flat_market):
    a = replay(flat_market, [entry(flat_market)], Protocol())
    changed = deepcopy(flat_market)
    changed["bars"][3].update(close=80.0, low=79.0)
    b = replay(changed, [entry(changed)], Protocol())
    assert a["fills"][0] == b["fills"][0]
    assert b["metrics"]["maximum_close_drawdown"] > a["metrics"]["maximum_close_drawdown"]


def test_final_unrealized_loss_is_not_hidden(flat_market):
    flat_market["funding"] = []
    flat_market["bars"][-1].update(close=90.0, low=89.0)
    result = replay(flat_market, [entry(flat_market)], Protocol(slippage_bps=0, max_hold_bars=99))
    assert result["metrics"]["net_return"] < -0.025
    assert result["trades"][0]["reason"] == "scheduled_end"
    assert result["equity_curve"][-1]["unrealized"] == 0


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "nan", "funding_duplicate", "funding_nan"]
)
def test_incomplete_or_invalid_market_rejected(flat_market, corruption):
    if corruption == "missing":
        flat_market["bars"].pop(2)
    elif corruption == "duplicate":
        flat_market["bars"].append(flat_market["bars"][0])
    elif corruption == "nan":
        flat_market["bars"][1]["close"] = float("nan")
    elif corruption == "funding_duplicate":
        flat_market["funding"] *= 2
    else:
        flat_market["funding"][0]["mark_price"] = float("nan")
    with pytest.raises(ValueError):
        validate_market(flat_market)


def test_stress_costs_reduce_fixed_signal_account(flat_market):
    base = Protocol()
    a = replay(flat_market, [entry(flat_market)], base)
    b = replay(flat_market, [entry(flat_market)], replace(base, slippage_bps=10))
    assert b["metrics"]["net_return"] < a["metrics"]["net_return"]


def test_two_symbols_share_capital(flat_market):
    flat_market["symbols"].append("OTHER")
    flat_market["bars"] += [{**b, "symbol": "OTHER"} for b in list(flat_market["bars"])]
    decisions = [entry(flat_market), {**entry(flat_market), "symbol": "OTHER", "rank": 0.5}]
    result = replay(flat_market, decisions, Protocol(position_fraction=0.8, max_positions=1))
    assert result["trades"][0]["symbol"] == "DEMOUSDT"
    assert all(point["open_positions"] <= 1 for point in result["equity_curve"])
    assert result["trades"][1]["entry_ms"] >= result["trades"][0]["exit_ms"]
    assert result["metrics"]["max_close_gross_fraction"] <= 1


@pytest.mark.parametrize("direction", [-1, 1])
def test_full_position_includes_adverse_fill_and_fees(flat_market, direction):
    flat_market["funding"] = []
    result = replay(
        flat_market,
        [entry(flat_market, direction)],
        Protocol(position_fraction=1.0, slippage_bps=10),
    )
    # 满仓边界需要同时扣除不利成交造成的浮亏与手续费，空头也不能超额。
    first_position = next(p for p in result["equity_curve"] if p["open_positions"])
    assert first_position["gross_notional"] <= first_position["equity"] + 1e-9


def fake_reply(request, python, source):
    snapshot = request["snapshot"]
    features = snapshot["features"]
    return {
        "schema_version": "ta-evidence-response.v1",
        "request_sha256": digest(request),
        "snapshot_sha256": digest(snapshot),
        "backend": request["backend"],
        "mode": request["mode"],
        "framework_source_sha256": source,
        "orders_enabled": False,
        "external_data_tools_enabled": False,
        "persistent_memory_enabled": False,
        "baseline_score": sum(features[k] for k in ("momentum", "flow", "crowding")) / 3,
        "rule_event_score": 0.5,
        "assessment": {"event_score": 0.25, "evidence_ids": [snapshot["evidence"][0]["id"]]},
        "elapsed_seconds": 0.1,
        "calls": [],
        "runtime": {"test": "fixture"},
    }


@pytest.fixture
def lab(tmp_path, monkeypatch):
    dataset = tmp_path / "data.json"
    save(dataset, demo_dataset())
    root = tmp_path / "ta" / "tradingagents"
    root.mkdir(parents=True)
    (root / "fake.py").write_text("# 测试源码身份\n")
    monkeypatch.setattr(experiment, "invoke", fake_reply)
    monkeypatch.setattr(
        experiment,
        "describe_worker",
        lambda python: {
            "framework_source_sha256": experiment.framework_identity(root.parent),
            "runtime": {"test": "fixture"},
        },
    )
    return dataset, tmp_path / "run", root.parent, "/unused/python"


def test_four_arms_eight_accounts_and_exact_resume(lab):
    result = experiment.run(*lab)
    assert len(result["metrics"]) == 8 and result["production_eligible"] is False
    assert experiment.verify(lab[1])["verified_accounts"] == 8
    assert experiment.run(*lab, resume=True) == result
    with pytest.raises(FileExistsError):
        experiment.run(*lab)


def test_failed_worker_has_no_fabricated_hold_and_resume_reuses_single(lab, monkeypatch):
    calls = []

    def failing(request, python, source):
        calls.append(request["mode"])
        if request["mode"] == "multi":
            raise RuntimeError("模拟服务失败")
        return fake_reply(request, python, source)

    monkeypatch.setattr(experiment, "invoke", failing)
    with pytest.raises(RuntimeError):
        experiment.run(*lab)
    assert not (lab[1] / "result.json").exists()
    assert len(list((lab[1] / "responses").glob("*.json"))) == 1
    monkeypatch.setattr(experiment, "invoke", fake_reply)
    assert experiment.run(*lab, resume=True)["backend"] == "fixture"


def test_protocol_change_invalidates_resume(lab):
    experiment.run(*lab)
    with pytest.raises(ValueError, match="发生变化"):
        experiment.run(*lab, protocol=Protocol(fee_bps=6), resume=True)


def test_tampered_response_and_decision_are_rejected(lab):
    experiment.run(*lab)
    path = lab[1] / "decisions.json"
    v = read(path)
    v["single"][0]["direction"] = -1
    path.write_text(json.dumps(v))
    # 即便更新外层文件哈希，复核仍须从原研究证据重建决策。
    result_path = lab[1] / "result.json"
    result = read(result_path)
    result["decisions_sha256"] = file_hash(path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="忠实"):
        experiment.verify(lab[1])


@pytest.mark.parametrize(
    "field,value",
    [("backend", "codex"), ("orders_enabled", True), ("framework_source_sha256", "other")],
)
def test_bridge_rejects_changed_identity(field, value):
    request = {"snapshot": demo_dataset()["snapshots"][0], "mode": "single", "backend": "fixture"}
    response = fake_reply(request, None, "source")
    response[field] = value
    with pytest.raises(ValueError):
        validate_response(request, response, "source")
