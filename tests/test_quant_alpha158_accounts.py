"""合成分钟验证24账户薄适配、连续持仓及完整失败证据，不运行真实账户。"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.backtesting import alpha158 as adapter
from tradingagents.research.quant.backtesting.paired_forecast import drive_accounts, select_pairs

MINUTE = 60_000


def case_grid():
    return {
        f"{method}:{asset}:20m:{model}:{cost}bps": (method, asset, model, cost)
        for method in adapter.METHODS
        for asset in ("甲", "乙")
        for model in adapter.MODELS
        for cost in adapter.COSTS
    }


def driver_fixture(*, missing_open=False, missing_held=False, risk=False):
    first, middle, last = 100 * MINUTE, 110 * MINUTE, 130 * MINUTE
    cases = case_grid()
    selected = [("L1", 1, 0.005), ("L2", 1, 0.004), ("S1", -1, 0.005), ("S2", -1, 0.004)]
    schedules = {
        (method, asset, model, clock): selected
        for method, asset, model, _ in cases.values()
        for clock in (first, middle)
    }
    funding = [
        {"symbol": symbol, "time_ms": stamp, "mark_price": 100.0, "rate": 0.001}
        for stamp in (first + MINUTE, first + MINUTE + 1, first + 21 * MINUTE)
        for symbol in ("L1", "L2", "S1", "S2")
    ]

    def frames():
        for clock in range(first, last + MINUTE, MINUTE):
            market = dict.fromkeys(("L1", "L2", "S1", "S2"), (100.0, 100.0, True, True))
            if clock == first + 21 * MINUTE:
                market.update(
                    {
                        symbol: (100.0, 110.0 if symbol.startswith("L") else 90.0, True, True)
                        for symbol in market
                    }
                )
            if missing_open and clock == first + 2 * MINUTE:
                market["L1"] = (np.nan, np.nan, True, False)
            if missing_held and clock == first + 5 * MINUTE:
                market["L1"] = (np.nan, np.nan, True, False)
            if risk and clock >= first + 5 * MINUTE:
                market["L1"] = market["L2"] = (100.0, 10.0, True, True)
            yield clock, market

    return cases, schedules, frames, funding, (first, middle, last)


def test_all_24_accounts_hold_through_block_and_charge_only_inventory_funding():
    cases, schedules, frames, funding, bounds = driver_fixture()
    accounts, result = drive_accounts(cases, schedules, frames(), funding, bounds)
    assert result["cases"] == len(result["metrics"]) == len(result["financial_proofs"]) == 24
    assert (
        result["frames"] == 31
        and not result["invalid_accounts"]
        and result["global_failure"] is None
    )
    first = bounds[0]
    for key, account in accounts.items():
        assert account.timed_exit_at_close and account.take_profit_fraction is None
        assert len(account.trades) == 4
        assert {trade["entry_ms"] for trade in account.trades} == {first + MINUTE}
        assert {trade["exit_ms"] for trade in account.trades} == {first + 21 * MINUTE}
        assert {trade["reason"] for trade in account.trades} == {"time_close"}
        assert {row["time_ms"] for row in account.funding_ledger} == {first + MINUTE + 1}
        assert len(account.funding_ledger) == 4
        metrics = result["metrics"][key]
        assert metrics["historical_blocks"][0]["ending_state"]["open_positions"] == 4
        assert metrics["historical_blocks"][1]["opening_equity"] == pytest.approx(
            metrics["historical_blocks"][0]["closing_equity"]
        )
        assert sum(
            row["pnl_over_initial"] for row in metrics["historical_blocks"]
        ) == pytest.approx(metrics["net_return"])
        assert result["opportunity_clocks"][key] == 2


def test_missing_open_cancels_whole_batch_and_never_selects_surviving_legs():
    cases, schedules, frames, funding, bounds = driver_fixture(missing_open=True)
    schedules = {key: value for key, value in schedules.items() if key[-1] == bounds[0]}
    accounts, result = drive_accounts(cases, schedules, frames(), funding, bounds)
    assert not result["invalid_accounts"]
    for account in accounts.values():
        assert not account.fills
        assert len(account.declines) == 4
        assert {row["reason"] for row in account.declines} == {"paired_batch_open_unavailable"}


def test_missing_held_price_invalidates_all_affected_accounts_but_retains_partial_ledgers():
    cases, schedules, frames, funding, bounds = driver_fixture(missing_held=True)
    accounts, result = drive_accounts(cases, schedules, frames(), funding, bounds)
    assert len(accounts) == len(result["invalid_accounts"]) == 24
    assert not result["metrics"] and not result["financial_proofs"]
    assert all(
        account.positions and account.fills and account.curve for account in accounts.values()
    )
    assert all(
        row["numeric_conclusions_allowed"] is False for row in result["invalid_accounts"].values()
    )


def test_risk_halt_persists_across_blocks_and_later_predictions_do_not_restart_capital():
    cases, schedules, frames, funding, bounds = driver_fixture(risk=True)
    accounts, result = drive_accounts(cases, schedules, frames(), funding, bounds)
    assert not result["invalid_accounts"]
    for key, account in accounts.items():
        assert account.halted and len(account.trades) == 4
        assert {trade["reason"] for trade in account.trades} == {"risk"}
        assert result["metrics"][key]["historical_blocks"][0]["ending_state"]["halted"]
        assert all(fill["clock"] < bounds[1] for fill in account.fills)


@pytest.mark.parametrize("failure", ["gap", "start", "short", "read_error"])
def test_market_axis_or_read_failure_keeps_all_cases_and_marks_numeric_results_invalid(failure):
    cases, schedules, frames, funding, bounds = driver_fixture()
    original = list(frames())
    if failure == "gap":
        source = iter(original[:5] + original[6:])
    elif failure == "start":
        source = iter(original[1:])
    elif failure == "short":
        source = iter(original[:-1])
    else:

        def broken():
            yield original[0]
            raise OSError("合成文件读取失败")

        source = broken()
    accounts, result = drive_accounts(cases, schedules, source, funding, bounds)
    assert len(accounts) == len(result["invalid_accounts"]) == 24
    assert result["global_failure"] and not result["metrics"]


def test_pair_selection_has_fixed_threshold_tie_order_and_requires_both_sides():
    frame = pd.DataFrame(
        {
            "symbol": ["B", "A", "D", "C", "weak", "unknown"],
            "score": [0.002, 0.002, -0.003, -0.003, 0.00159, np.nan],
        }
    )
    assert select_pairs(frame, "score") == [
        ("A", 1, 0.002),
        ("B", 1, 0.002),
        ("C", -1, 0.003),
        ("D", -1, 0.003),
    ]
    assert select_pairs(frame[frame.score > 0], "score") == []
    with pytest.raises(ValueError):
        select_pairs(pd.concat([frame, frame.iloc[:1]]), "score")


def prediction_fixture(monkeypatch, members=20):
    first = int(pd.Timestamp("2026-10-06T00:00:00Z").value // 1_000_000)
    last = first + 60 * MINUTE
    phases = {
        "training": {
            "start": pd.Timestamp(first - 60 * MINUTE, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
        },
        "retrospective_validation": {
            "start": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp(last, unit="ms", tz="UTC").isoformat(),
        },
        "reserved_later_day": {
            "start": pd.Timestamp(last, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp(last + 60 * MINUTE, unit="ms", tz="UTC").isoformat(),
        },
    }
    monkeypatch.setattr(adapter, "REFITS", (first, first + 10 * MINUTE))
    monkeypatch.setattr(adapter, "END_MS", last)
    monkeypatch.setattr(adapter, "TRAINING_START_MS", first - 60 * MINUTE)
    monkeypatch.setattr(adapter, "PHASES", phases)
    universe = {
        f"{asset}{number:03}": {"asset_class": asset}
        for asset in ("甲", "乙")
        for number in range(members)
    }
    records = [
        {
            "symbol": symbol,
            "asset_class": meta["asset_class"],
            "clock_ms": clock,
            "coefficient_mode": mode,
            "common_member_gate": True,
            **{
                f"prediction_{model}": 0.01 * (int(symbol[1:]) / (members - 1) - 0.5)
                for model in adapter.MODELS
            },
        }
        for mode in adapter.METHODS
        for clock in (first - 55 * MINUTE, first + 5 * MINUTE, first + 65 * MINUTE)
        for symbol, meta in universe.items()
    ]
    return pd.DataFrame(records), universe, first, last


def test_complete_schedules_exclude_training_and_later_rows_without_future_labels(monkeypatch):
    predictions, universe, first, _ = prediction_fixture(monkeypatch)
    cases, schedules, report = adapter.build_schedules(predictions, universe)
    assert len(cases) == 24 and len(schedules) == 8
    assert {key[-1] for key in schedules} == {first + 5 * MINUTE}
    assert (
        report["training_rows_excluded"]
        == report["later_day_rows_excluded"]
        == report["account_rows"]
        == 80
    )
    assert len(report["schedules"]) == 8
    assert all(len(selected) == 4 for selected in schedules.values())


def test_one_unknown_prediction_cancels_entire_common_model_clock(monkeypatch):
    predictions, universe, first, _ = prediction_fixture(monkeypatch)
    mask = (
        (predictions.coefficient_mode == "fixed")
        & (predictions.asset_class == "甲")
        & (predictions.clock_ms == first + 5 * MINUTE)
    )
    predictions.loc[predictions[mask].index[0], "prediction_baseline_alpha158_ridge_167"] = np.nan
    _, schedules, report = adapter.build_schedules(predictions, universe)
    assert schedules["fixed", "甲", "baseline_alpha158_ridge_167", first + 5 * MINUTE] == []
    assert len(schedules["fixed", "甲", "baseline_ridge_9", first + 5 * MINUTE]) == 4
    assert sum(row["status"] == "INCOMPLETE_COMMON_PREDICTIONS" for row in report["schedules"]) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "mode_gate",
        "small_gate",
        "float_clock",
        "outside_gate",
        "future_column",
    ],
)
def test_prediction_contract_rejects_population_or_identity_changes(monkeypatch, mutation):
    predictions, universe, _, _ = prediction_fixture(monkeypatch)
    if mutation == "missing":
        predictions = predictions.iloc[1:]
    elif mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif mutation == "mode_gate":
        predictions.loc[0, "common_member_gate"] = False
        predictions.loc[0, [f"prediction_{model}" for model in adapter.MODELS]] = np.nan
    elif mutation == "small_gate":
        mask = predictions.symbol.eq("甲000")
        predictions.loc[mask, "common_member_gate"] = False
        predictions.loc[mask, [f"prediction_{model}" for model in adapter.MODELS]] = np.nan
    elif mutation == "float_clock":
        predictions.clock_ms = predictions.clock_ms.astype(float)
    elif mutation == "outside_gate":
        predictions.common_member_gate = False
    else:
        predictions["target_20"] = 1e20
    with pytest.raises(ValueError):
        adapter.validate_predictions(predictions, universe)


def source_fixture(tmp_path, monkeypatch):
    predictions, universe, first, last = prediction_fixture(monkeypatch, members=374)
    source, market, funding, label, output = (
        tmp_path / name for name in ("source", "market", "funding", "label", "output")
    )
    for path in (source, market, funding, label):
        path.mkdir()
    # 故意放入未来列；账户读取器必须只选白名单，不读这三个列。
    predictions["target_20"], predictions["phase_20"], predictions["holding_path_observed"] = (
        1e100,
        "forbidden",
        False,
    )
    predictions.to_parquet(source / "predictions.parquet", index=False)
    index = pd.MultiIndex.from_product(
        [
            pd.date_range(
                pd.Timestamp(first, unit="ms", tz="UTC"),
                pd.Timestamp(last, unit="ms", tz="UTC"),
                freq="min",
            ),
            universe,
        ],
        names=["bar_time", "symbol"],
    )
    data = pd.DataFrame(
        {"open": 100.0, "close": 100.0, "eligible": True, "observed": True}, index=index
    )
    exit_clock = pd.Timestamp(first + 26 * MINUTE, unit="ms", tz="UTC")
    selected = [f"{asset}{number:03}" for asset in ("甲", "乙") for number in (0, 1, 372, 373)]
    for symbol in selected:
        data.loc[(exit_clock, symbol), "close"] = 110.0 if int(symbol[1:]) >= 372 else 90.0
    data.to_parquet(market / "day.parquet")
    manifest = {
        "contracts": universe,
        "orders_enabled": False,
        "leader_labels_read": False,
        "market_panel_contract": {"timestamp_semantics": "CLOSE_TIME"},
        "daily_files": [
            {
                "first_close_ms": first,
                "last_close_ms": last,
                "path": "day.parquet",
                "sha256": sha(market / "day.parquet"),
            }
        ],
    }
    save(market / "manifest.json", manifest)
    capture = {
        "panel": str(market),
        "manifest_sha256": sha(market / "manifest.json"),
        "start": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
        "end": pd.Timestamp(last, unit="ms", tz="UTC").isoformat(),
        "funding_symbols": list(universe),
        "orders_enabled": False,
        "leader_labels_read": False,
    }
    save(funding / "capture_plan.json", capture)
    save(
        funding / "funding_manifest.json", {"api_time_range_coverage_verified": True, "leaves": []}
    )
    save(funding / "funding_events.json", [])
    save(
        label / "protocol.json",
        {
            "schema": "ta-native-settlement-unwind-study.v1",
            "universe": universe,
            "orders_enabled": False,
            "leader_labels_read": False,
            "input_identities": {
                str(funding / name): sha(funding / name)
                for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json")
            },
        },
    )
    protocol = {
        "schema": "ta-alpha158-ridge-study.v1",
        "account_family": "alpha158_ridge",
        "alpha158_profile": adapter.PROFILE,
        "features": adapter.FEATURES,
        "models": adapter.MODELS,
        "phases": adapter.PHASES,
        "refit_times_ms": adapter.REFITS,
        "universe": universe,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
        "label_source": str(label),
        "input_identities": {str(label / "protocol.json"): sha(label / "protocol.json")},
    }
    save(source / "protocol.json", protocol)
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    return source, market, funding, output, first


def test_thin_adapter_runs_all24_with_only_prediction_whitelist_and_preserves_blocks(
    tmp_path, monkeypatch
):
    source, market, funding, output, first = source_fixture(tmp_path, monkeypatch)
    original_read = pd.read_parquet
    loaded = []

    def spy(path, **kwargs):
        loaded.append((str(path), kwargs.get("columns")))
        assert not {"target_20", "phase_20", "holding_path_observed"} & set(
            kwargs.get("columns", [])
        )
        return original_read(path, **kwargs)

    monkeypatch.setattr(adapter.pd, "read_parquet", spy)
    result = adapter.run(source, market, funding, output)
    verify_receipts(output)
    assert len(result["metrics"]) == len(result["financial_proofs"]) == result["cases"] == 24
    assert not result["invalid_accounts"] and not result["future_labels_loaded"]
    assert loaded[0][1] == list(adapter.PREDICTION_COLUMNS)
    assert len(list(output.glob("*.json.gz"))) == 24
    for key, metrics in result["metrics"].items():
        assert (
            metrics["closed_trades"] == 4
            and metrics["historical_blocks"][0]["ending_state"]["open_positions"] == 4
        )
        assert result["case_execution_status"][key] == "FINANCIALLY_VERIFIED"
        with gzip.open(output / (key.replace(":", "_") + ".json.gz"), "rt") as stream:
            ledger = json.load(stream)
        assert {trade["exit_ms"] for trade in ledger["trades"]} == {first + 26 * MINUTE}
        assert {trade["reason"] for trade in ledger["trades"]} == {"time_close"}


def test_adapter_failure_before_execution_still_seals_24_unstarted_ledgers(tmp_path, monkeypatch):
    source, market, funding, output, _ = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adapter, "build_schedules", lambda *_: (_ for _ in ()).throw(ValueError("合成调度失败"))
    )
    with pytest.raises(ValueError, match="合成调度失败"):
        adapter.run(source, market, funding, output)
    verify_receipts(output)
    assert len(list(output.glob("*.json.gz"))) == 24
    assert read(output / "failure.json")["numeric_conclusions_allowed"] is False
    for path in output.glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            ledger = json.load(stream)
        assert ledger["execution_started"] is False and ledger["metrics"] is None


@pytest.mark.parametrize("changed", ["market", "funding"])
def test_exact_market_or_funding_source_identity_changes_are_rejected(
    tmp_path, monkeypatch, changed
):
    source, market, funding, _, _ = source_fixture(tmp_path, monkeypatch)
    adapter._sources(source, market, funding)
    path = funding / "funding_events.json" if changed == "funding" else market / "day.parquet"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        adapter._sources(source, market, funding)


def test_adapter_read_failure_seals_all_partial_pending_ledgers_without_metrics(
    tmp_path, monkeypatch
):
    source, market, funding, output, first = source_fixture(tmp_path, monkeypatch)
    original = adapter.market_frames

    def broken(*args):
        for clock, quotes in original(*args):
            if clock > first + 6 * MINUTE:
                raise OSError("合成部分运行后读盘失败")
            yield clock, quotes

    monkeypatch.setattr(adapter, "market_frames", broken)
    with pytest.raises(ValueError, match="分钟输入失败"):
        adapter.run(source, market, funding, output)
    verify_receipts(output)
    partial = read(output / "partial_result.json")
    assert len(partial["invalid_accounts"]) == 24 and not partial["financial_proofs"]
    assert len(list(output.glob("*.json.gz"))) == 24
    for path in output.glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            ledger = json.load(stream)
        assert ledger["execution_started"] and ledger["metrics"] is None
        assert len(ledger["ending_state"]["pending"]) == 4
        assert not ledger["ending_state"]["positions"]
