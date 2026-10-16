"""用完整合成来源检验账户封存、期末平仓、重训时延续库存与独立财务路径。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import save, sha, verify_receipts
from tradingagents.research.quant.backtesting.forecast import FUNDING_MODELS, TESTED_MODELS, run


def sources(tmp_path, family):
    """短分钟样例只检验账户控制流程，不用于估计市场收益或训练系数。"""
    models = FUNDING_MODELS if family == "settled_funding" else TESTED_MODELS
    source, market, funding, output = (
        tmp_path / name for name in ("source", "market", "funding", "output")
    )
    for root in (source, market, funding):
        root.mkdir()
    first, last = pd.Timestamp("2026-10-06T00:00:00Z"), pd.Timestamp("2026-10-06T00:05:00Z")
    start_ms, end_ms = (int(t.value // 1_000_000) for t in (first, last))
    contracts, samples = {}, []
    for asset in ("甲", "乙"):
        for number in range(40):
            symbol = f"{asset}{number:03}"
            contracts[symbol] = {"asset_class": asset}
            samples.append(
                {"symbol": symbol, "asset_class": asset, "clock_ms": start_ms, "factor": number}
            )
    protocol = {
        "phases": {
            "retrospective_validation": {"start": first.isoformat(), "end": last.isoformat()}
        },
        "account_family": family,
        "universe": contracts,
        "features": ["factor"],
        "horizons_minutes": [20, 60, 240],
        "orders_enabled": False,
        "leader_labels_read": False,
    }
    save(
        source / "result.json",
        {
            "groups": {
                f"{asset}:{h}m": {
                    "frozen_training_coefficients": {m: {"factor": 0.01} for m in models}
                }
                for asset in ("甲", "乙")
                for h in (20, 60, 240)
            }
        },
    )
    # 不提供任何未来标签列，回测仍必须能完整执行。
    pd.DataFrame(samples).to_parquet(source / "hourly_samples.parquet", index=False)
    index = pd.MultiIndex.from_product(
        [pd.date_range(first, last, freq="min"), contracts], names=["bar_time", "symbol"]
    )
    data = pd.DataFrame(
        {"open": np.full(len(index), 100.0), "close": 100.0, "eligible": True, "observed": True},
        index=index,
    )
    data.to_parquet(market / "day.parquet")
    save(
        market / "manifest.json",
        {
            "contracts": contracts,
            "orders_enabled": False,
            "leader_labels_read": False,
            "market_panel_contract": {"timestamp_semantics": "CLOSE_TIME"},
            "daily_files": [
                {
                    "first_close_ms": start_ms,
                    "last_close_ms": end_ms,
                    "path": "day.parquet",
                    "sha256": sha(market / "day.parquet"),
                }
            ],
        },
    )
    save(
        funding / "capture_plan.json",
        {
            "panel": str(market),
            "manifest_sha256": sha(market / "manifest.json"),
            "start": first.isoformat(),
            "end": last.isoformat(),
        },
    )
    save(
        funding / "funding_manifest.json", {"api_time_range_coverage_verified": True, "leaves": []}
    )
    save(funding / "funding_events.json", [])
    if family == "settled_funding":
        protocol["input_identities"] = {
            str(funding / name): sha(funding / name)
            for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json")
        }
    save(source / "protocol.json", protocol)
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    return source, market, funding, output


@pytest.mark.parametrize("family", ["market_response", "settled_funding"])
def test_all_declared_accounts_execute_and_seal_end_to_end_without_future_labels(tmp_path, family):
    source, market, funding, output = sources(tmp_path, family)
    result = run(source, market, funding, output)
    verify_receipts(output)
    expected_cases = 72 if family == "settled_funding" else 48
    assert (
        result["cases"]
        == len(result["metrics"])
        == len(result["financial_proofs"])
        == expected_cases
    )
    assert not result["invalid_accounts"] and result["frames"] == 6
    for metrics in result["metrics"].values():
        assert metrics["closed_trades"] == 4 and metrics["complete_batches"] == 1
        assert metrics["net_return"] < 0
    assert result["production_eligible"] is result["independent_oos"] is False
    assert result["historical_assessment"]["real_time_forward_required"] is False
    adverse = run(source, market, funding, tmp_path / "adverse", extra_slippage_bps=11)
    assert adverse["cases"] == len(adverse["metrics"]) == 72
    for key, metrics in result["metrics"].items():
        assert adverse["metrics"][key] == metrics
        if key.endswith(":10bps"):
            assert (
                adverse["metrics"][key.removesuffix(":10bps") + ":11bps"]["net_return"]
                < metrics["net_return"]
            )


def test_refit_reverses_new_decisions_without_resetting_old_inventory_or_block_equity(
    tmp_path, monkeypatch
):
    from tradingagents.research.quant import rolling

    source, market, funding, output = sources(tmp_path, "settled_funding")
    samples = pd.read_parquet(source / "hourly_samples.parquet")
    first = int(samples.clock_ms.iloc[0])
    later = samples.copy()
    later.clock_ms += 2 * 60_000
    pd.concat([samples, later]).to_parquet(source / "hourly_samples.parquet", index=False)
    (source / "receipts.json").unlink()
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    # 重训数学另有真实标签测试；这里缩短边界并替换系数，专门反证重训导致强平或清账。
    monkeypatch.setattr(rolling, "REFITS", tuple(first + i * 60_000 for i in range(4)))
    monkeypatch.setattr(rolling, "fit_expanding", lambda frame, protocol: {"synthetic": True})
    monkeypatch.setattr(
        rolling,
        "coefficients_at",
        lambda training, asset, horizon, model, clock: {
            "factor": 0.01 if clock == first else -0.01
        },
    )
    result = run(source, market, funding, output, historical_rolling=True)
    verify_receipts(output)
    assert result["cases"] == len(result["financial_proofs"]) == 72
    assert not result["invalid_accounts"]
    for metrics in result["metrics"].values():
        assert metrics["closed_trades"] == 4 and metrics["complete_batches"] == 1
        blocks = metrics["historical_blocks"]
        assert len(blocks) == 4
        assert sum(block["pnl_over_initial"] for block in blocks) == pytest.approx(
            metrics["net_return"]
        )
        assert np.prod([1 + block["equity_return"] for block in blocks]) - 1 == pytest.approx(
            metrics["net_return"]
        )
        assert sum(block["closed_legs"] for block in blocks) == 4
        assert sum(block["started_batches"] for block in blocks) == 1
        assert sum(block["completed_batches"] for block in blocks) == 1
        assert sum(block["opportunity_clocks"] for block in blocks) == 2
        assert sum(block["funding_cash"] for block in blocks) == metrics["funding_cash"] == 0
        assert sum(
            block["fee_cash_outflow"] + block["modelled_slippage_drag_cash"] for block in blocks
        ) == pytest.approx(-1000 * metrics["net_return"])
        for block in blocks:
            assert block["fee_cash_outflow"] + block[
                "modelled_slippage_drag_cash"
            ] == pytest.approx(-1000 * block["pnl_over_initial"], abs=1e-10)
        assert blocks[1]["fee_cash_outflow"] == 0
        assert blocks[2]["fee_cash_outflow"] > 0
        assert blocks[2]["started_batches"] == 1
        assert blocks[2]["ending_account_state"]["open_positions"] == 4
        assert blocks[-1]["ending_account_state"]["open_positions"] == 0
        assert all(
            left["closing_equity"] == right["opening_equity"]
            for left, right in zip(blocks[:-1], blocks[1:], strict=True)
        )
    assert all(value == 2 for value in result["opportunity_clocks"].values())
    assert result["historical_assessment"]["validation_route"] == "historical_rolling"
    assert result["historical_assessment"]["real_time_forward_required"] is False
