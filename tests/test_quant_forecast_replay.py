"""用完整合成来源检验48个账户的封存、入场、期末平仓与独立财务路径。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import save, sha, verify_receipts
from tradingagents.research.quant.backtesting.forecast import FUNDING_MODELS, TESTED_MODELS, run


@pytest.mark.parametrize("family", ["market_response", "settled_funding"])
def test_all_declared_accounts_execute_and_seal_end_to_end_without_future_labels(tmp_path, family):
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
    adverse = run(source, market, funding, tmp_path / "adverse", extra_slippage_bps=11)
    assert adverse["cases"] == len(adverse["metrics"]) == 72
    for key, metrics in result["metrics"].items():
        assert adverse["metrics"][key] == metrics
        if key.endswith(":10bps"):
            assert (
                adverse["metrics"][key.removesuffix(":10bps") + ":11bps"]["net_return"]
                < metrics["net_return"]
            )
