"""人工来源验证48账户及跨重训持仓，专门区分到期收盘与下一开盘。"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.backtesting.settlement import decisions, run
from tradingagents.research.quant.settlement_study import FEATURES, MODELS


def sources(tmp_path):
    """合成市场只证明执行控制流程，不是策略利润证据。"""
    source, market, funding, output = (
        tmp_path / name for name in ("source", "market", "funding", "output")
    )
    for path in (source, market, funding):
        path.mkdir()
    first = pd.Timestamp("2026-10-06T00:05:00Z")
    end = first + pd.Timedelta(minutes=25)
    begin, last = (int(value.value // 1_000_000) for value in (first, end))
    middle = begin + 10 * 60_000
    contracts, samples = {}, []
    for asset in ("甲", "乙"):
        for number in range(40):
            symbol = f"{asset}{number:03}"
            contracts[symbol] = {"asset_class": asset}
            for clock in (begin, middle):
                samples.append(
                    {
                        "symbol": symbol,
                        "asset_class": asset,
                        "clock_ms": clock,
                        **dict.fromkeys(FEATURES, 0.0),
                        FEATURES[0]: number / 39 - 0.5,
                    }
                )
    pd.DataFrame(samples).to_parquet(source / "hourly_samples.parquet", index=False)
    coefficients = {"features": FEATURES, "models": MODELS, "refits": {}}
    for refit in (begin, middle):
        record = {
            "means": dict.fromkeys(FEATURES, 0.0),
            "scales": dict.fromkeys(FEATURES, 1.0),
            "coefficients": {
                model: {
                    "intercept": 0.0,
                    **{
                        feature: (0.01 if refit == begin else -0.01)
                        if feature == FEATURES[0]
                        else 0.0
                        for feature in columns
                    },
                }
                for model, columns in MODELS.items()
            },
            "status_by_model": dict.fromkeys(MODELS, "READY"),
        }
        coefficients["refits"][str(refit)] = {
            "refit_ms": refit,
            "end_ms": last,
            "groups": {f"{asset}:20m": record for asset in ("甲", "乙")},
        }
    save(source / "coefficients.json", coefficients)
    index = pd.MultiIndex.from_product(
        [pd.date_range(first, end, freq="min"), contracts], names=["bar_time", "symbol"]
    )
    data = pd.DataFrame(
        {"open": 100.0, "close": 100.0, "eligible": True, "observed": True}, index=index
    )
    # 仅到期收盘有价格变化，下一开盘回到100；错误使用开盘会使结果完全不同。
    close_at = first + pd.Timedelta(minutes=21)
    for asset in ("甲", "乙"):
        for number in (0, 1, 38, 39):
            data.loc[(close_at, f"{asset}{number:03}"), "close"] = 90.0 if number < 2 else 110.0
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
                    "first_close_ms": begin,
                    "last_close_ms": last,
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
            "end": end.isoformat(),
        },
    )
    save(
        funding / "funding_manifest.json", {"api_time_range_coverage_verified": True, "leaves": []}
    )
    save(funding / "funding_events.json", [])
    save(
        source / "protocol.json",
        {
            "account_family": "settlement_unwind",
            "hypothesis_id": "phase3_settlement_unwind_01",
            "orders_enabled": False,
            "leader_labels_read": False,
            "features": FEATURES,
            "universe": contracts,
            "phases": {
                "retrospective_validation": {"start": first.isoformat(), "end": end.isoformat()}
            },
            "input_identities": {
                str(funding / name): sha(funding / name)
                for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json")
            },
        },
    )
    save(source / "receipts.json", {path.name: sha(path) for path in source.iterdir()})
    return source, market, funding, output, begin


def test_all_48_accounts_hold_through_refit_and_exit_on_exact_complete_close(tmp_path):
    source, market, funding, output, first = sources(tmp_path)
    result = run(source, market, funding, output)
    verify_receipts(output)
    assert result["cases"] == len(result["metrics"]) == len(result["financial_proofs"]) == 48
    assert not result["invalid_accounts"] and result["frames"] == 26
    assert result["real_time_forward_required"] is result["production_eligible"] is False
    for key, metrics in result["metrics"].items():
        assert metrics["net_return"] > 0 and metrics["closed_trades"] == 4
        assert metrics["complete_batches"] == 1
        assert metrics["open_positions"] == 0
        assert metrics["historical_blocks"][0]["ending_state"]["open_positions"] == 4
        assert sum(
            row["pnl_over_initial"] for row in metrics["historical_blocks"]
        ) == pytest.approx(metrics["net_return"])
        with gzip.open(output / (key.replace(":", "_") + ".json.gz"), "rt") as stream:
            ledger = json.load(stream)
        assert {trade["exit_ms"] for trade in ledger["trades"]} == {first + 21 * 60_000}
        assert {trade["reason"] for trade in ledger["trades"]} == {"time_close"}
        assert {trade["signal_ms"] for trade in ledger["trades"]} == {first}
    assert set(result["opportunity_clocks"].values()) == {2}


def test_decisions_ignore_future_target_and_reject_duplicate_symbol(tmp_path):
    source, _, _, _, first = sources(tmp_path)
    frame = pd.read_parquet(source / "hourly_samples.parquet")
    frame = frame[(frame.asset_class == "甲") & (frame.clock_ms == first)].copy()
    record = read(source / "coefficients.json")["refits"][str(first)]["groups"]["甲:20m"]
    before = decisions(frame, record, "expanded_h01")
    frame["target_20"] = np.nan
    frame["phase_20"] = "fabricated"
    assert decisions(frame, record, "expanded_h01") == before
    frame["target_20"] = np.arange(len(frame)) * 1e9
    assert decisions(frame, record, "expanded_h01") == before
    assert len(before) == 4 and sum(direction for _, direction, _ in before) == 0
    with pytest.raises(ValueError, match="重复"):
        decisions(pd.concat([frame, frame.iloc[:1]]), record, "expanded_h01")
