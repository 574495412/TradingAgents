"""纯合成验证四小时48账户、显式现金、完整人口及有界分钟缓存。"""

import gzip
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.backtesting import extended_funding as adapter
from tradingagents.research.quant.backtesting.paired_forecast import drive_accounts

MINUTE = 60_000


@pytest.fixture
def small_calendar(monkeypatch):
    """缩短日历不改变48案例；保留两种明确资产及一个UNKNOWN。"""
    start = int(pd.Timestamp("2026-08-01T00:00:00Z").value // 1_000_000)
    first, second, end = start + 60 * MINUTE, start + 180 * MINUTE, start + 420 * MINUTE
    for name, value in {
        "START": start,
        "END": end,
        "REFITS": (first, second),
        "EXPECTED_SYMBOLS": 41,
        "ORIGINAL_COUNT": 40,
    }.items():
        monkeypatch.setattr(adapter, name, value)
    universe = {}
    for asset, prefix in zip(adapter.ASSETS, ("C", "T"), strict=True):
        for number in range(20):
            symbol = f"{prefix}{number:02}USDT"
            universe[symbol] = {
                "asset_class": asset,
                "onboard_ms": start - 86_400_000,
                "delivery_ms": end + 86_400_000,
                "qualification_interval_known": True,
                "snapshot_symbol_found": True,
                "original_748_member": True,
                "historical_point_in_time_metadata_proven": False,
            }
    universe["UNKNOWNUSDT"] = {
        "asset_class": "UNKNOWN",
        "onboard_ms": None,
        "delivery_ms": None,
        "qualification_interval_known": False,
        "snapshot_symbol_found": False,
        "original_748_member": False,
        "historical_point_in_time_metadata_proven": False,
    }
    rows = []
    for method in adapter.METHODS:
        for clock in range(start, end, adapter.HOUR):
            for symbol, contract in universe.items():
                known = contract["asset_class"] != "UNKNOWN"
                score = (int(symbol[1:3]) / 19 - 0.5) * 0.01 if known else np.nan
                rows.append(
                    {
                        "symbol": symbol,
                        "asset_class": contract["asset_class"],
                        "clock_ms": clock,
                        "coefficient_mode": method,
                        "common_member_gate": known,
                        **{
                            f"prediction_{name}": score if clock >= first else np.nan
                            for name in adapter.TRADING_MODELS
                        },
                    }
                )
    return start, first, second, end, universe, pd.DataFrame(rows)


def typed_row(symbol, stamp, *, kind="Regular", rate="0.001", trusted=True, ready=True):
    """同钟异类型保留原字段和精确十进制，不制造默认Regular。"""
    raw = {
        "symbol": symbol,
        "fundingTime": stamp,
        "fundingRate": rate,
        "markPrice": "100",
        "rateType": kind,
    }
    return {
        "symbol": symbol,
        "time_ms": stamp,
        "rate": rate,
        "mark": "100",
        "type": kind,
        "raw": raw,
        "source": {"event_index": stamp % 1000 + (kind == "Special")},
        "quality": {
            "explicit_type_verified": True,
            "trusted_typed_fields": trusted,
            "identity_conflict": not trusted,
            "month_ready_under_source_assumptions": ready,
        },
    }


def cash_fixture(root, calendar, *, unknown_symbol=None, untrusted=False):
    start, first, _, _, universe, _ = calendar
    events = [
        typed_row(symbol, start + 1)
        for symbol, contract in universe.items()
        if contract["asset_class"] != "UNKNOWN"
    ]
    for symbol in universe:
        if symbol == "UNKNOWNUSDT":
            continue
        for stamp, kind, rate in (
            (first + MINUTE, "Regular", "0.001"),
            (first + MINUTE + 1, "Regular", "0.001"),
            (first + MINUTE + 1, "Special", "-0.002"),
            (first + 241 * MINUTE, "Regular", "0.001"),
        ):
            events.append(typed_row(symbol, stamp, kind=kind, rate=rate))
    coverage = []
    for symbol in universe:
        for month in adapter.historical_panel.MONTHS:
            ready = month == "2026-08" and symbol not in {"UNKNOWNUSDT", unknown_symbol}
            first_ms, last_ms = adapter.historical_panel.month_bounds(month)
            matching = [
                row
                for row in events
                if row["symbol"] == symbol and first_ms <= row["time_ms"] < last_ms
            ]
            for row in matching:
                row["quality"]["month_ready_under_source_assumptions"] = ready
                if untrusted and symbol == unknown_symbol and row["type"] == "Special":
                    row["quality"]["trusted_typed_fields"] = False
                    row["quality"]["identity_conflict"] = True
            coverage.append(
                {
                    "symbol": symbol,
                    "month": month,
                    "start_ms": first_ms,
                    "end_exclusive_ms": last_ms,
                    "month_ready_under_source_assumptions": ready,
                    "observed_regular_events": sum(row["type"] == "Regular" for row in matching),
                    "observed_special_events": sum(row["type"] == "Special" for row in matching),
                }
            )
    event_path, coverage_path = root / "typed_cash_events.jsonl", root / "coverage.json"
    with event_path.open("x", encoding="utf-8") as stream:
        for row in events:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(coverage_path, coverage)
    return event_path, coverage_path


def synthetic_frames(calendar, *, missing=False):
    _, first, _, end, universe, _ = calendar
    for clock in range(first, end + MINUTE, MINUTE):
        market = {}
        for symbol, contract in universe.items():
            raw_open = raw_close = 100.0
            if clock == first + 242 * MINUTE and symbol != "UNKNOWNUSDT":
                raw_open = (
                    110.0 if int(symbol[1:3]) >= 18 else 90.0 if int(symbol[1:3]) <= 1 else 100.0
                )
                raw_close = (
                    105.0 if int(symbol[1:3]) >= 18 else 95.0 if int(symbol[1:3]) <= 1 else 100.0
                )
            observed = not (missing and symbol == "C00USDT" and clock == first + 10 * MINUTE)
            market[symbol] = (
                raw_open if observed else np.nan,
                raw_close if observed else np.nan,
                contract["asset_class"] != "UNKNOWN",
                observed,
            )
        yield clock, market


def test_48_grid_excludes_controls_and_preserves_unknown_and_training(small_calendar):
    _, first, _, end, universe, predictions = small_calendar
    schedules, report = adapter.build_schedules(predictions, universe)
    assert len(adapter.cases()) == 48 and "controls" not in adapter.TRADING_MODELS
    assert len(schedules) == 2 * 2 * 4 * ((end - first) // adapter.HOUR)
    assert all(first <= key[-1] < end for key in schedules)
    assert report["unknown_asset_rows_retained"] == 14 and report["training_rows_excluded"] == 82
    assert all(len(value) == 4 for value in schedules.values())
    assert not {"target_240", "phase_240", "prediction_controls"} & set(adapter.PREDICTION_COLUMNS)


@pytest.mark.parametrize(
    "bad",
    ["target", "controls", "missing", "unknown_gate", "training_prediction", "one_method_gate"],
)
def test_prediction_whitelist_and_full_population_reject_mutations(small_calendar, bad):
    _, _, _, _, universe, predictions = small_calendar
    predictions = predictions.copy()
    if bad == "target":
        predictions["target_240"] = 1
    elif bad == "controls":
        predictions["prediction_controls"] = 1
    elif bad == "missing":
        predictions = predictions.iloc[1:]
    elif bad == "unknown_gate":
        predictions.loc[predictions.symbol.eq("UNKNOWNUSDT"), "common_member_gate"] = True
    elif bad == "training_prediction":
        predictions.loc[0, f"prediction_{adapter.TRADING_MODELS[0]}"] = 0.01
    else:
        predictions.loc[0, "common_member_gate"] = False
    with pytest.raises(ValueError):
        adapter.validate_predictions(predictions, universe)


def test_missing_prediction_rejects_whole_model_clock(small_calendar):
    _, first, _, _, universe, predictions = small_calendar
    mask = (
        predictions.coefficient_mode.eq("fixed")
        & predictions.asset_class.eq(adapter.ASSETS[0])
        & predictions.clock_ms.eq(first)
    )
    predictions.loc[predictions[mask].index[0], f"prediction_{adapter.TRADING_MODELS[0]}"] = np.nan
    schedules, report = adapter.build_schedules(predictions, universe)
    assert schedules["fixed", adapter.ASSETS[0], adapter.TRADING_MODELS[0], first] == []
    assert sum(row["status"] == "INCOMPLETE_COMMON_PREDICTIONS" for row in report["schedules"]) == 1


def test_open_240_exit_and_regular_special_cash_hold_across_refit(tmp_path, small_calendar):
    _, first, second, end, universe, predictions = small_calendar
    events, coverage = cash_fixture(tmp_path, small_calendar)
    funding, ready, report = adapter.load_cash(events, coverage, universe)
    schedules, _ = adapter.build_schedules(predictions, universe)
    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        synthetic_frames(small_calendar),
        funding,
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    assert not result["invalid_accounts"] and len(result["financial_proofs"]) == 48
    assert not adapter.reject_unknown_cash(accounts, result, ready)
    assert report["observed_types"]["Special"] == 40
    for key, account in accounts.items():
        assert not account.timed_exit_at_close and account.hold == 240 * MINUTE
        initial = [trade for trade in account.trades if trade["signal_ms"] == first]
        assert len(initial) == 4
        assert {trade["exit_ms"] for trade in initial} == {first + 241 * MINUTE}
        assert {trade["reason"] for trade in initial} == {"time"}
        terminal = [trade for trade in account.trades if trade["reason"] == "scheduled_end"]
        assert terminal and all(
            trade["exit_ms"] == end and end - trade["entry_ms"] < 240 * MINUTE for trade in terminal
        )
        assert all(
            trade["exit_price"]
            == pytest.approx(
                (110.0 if trade["direction"] > 0 else 90.0)
                * (1 - trade["direction"] * account.slip)
            )
            for trade in initial
        )
        assert any(row["rate_type"] == "Special" for row in account.funding_ledger)
        assert all(row["time_ms"] != first + MINUTE for row in account.funding_ledger)
        assert any(row["time_ms"] == first + 241 * MINUTE for row in account.funding_ledger)
        blocks = result["metrics"][key]["historical_blocks"]
        assert blocks[0]["ending_state"]["open_positions"] == 4
        assert blocks[1]["opening_equity"] == blocks[0]["closing_equity"]
        assert sum(block["pnl_over_initial"] for block in blocks) == pytest.approx(
            result["metrics"][key]["net_return"]
        )


def test_unknown_cash_invalidates_affected_accounts_without_avoiding_orders(
    tmp_path, small_calendar
):
    _, first, second, end, universe, predictions = small_calendar
    events, coverage = cash_fixture(
        tmp_path, small_calendar, unknown_symbol="C00USDT", untrusted=True
    )
    funding, ready, _ = adapter.load_cash(events, coverage, universe)
    schedules, _ = adapter.build_schedules(predictions, universe)
    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        synthetic_frames(small_calendar),
        funding,
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    assert all(account.fills for account in accounts.values())
    failures = adapter.reject_unknown_cash(accounts, result, ready)
    assert len(failures) == 24 and len(result["metrics"]) == len(result["financial_proofs"]) == 24
    assert all(adapter.ASSETS[0] in key for key in failures)
    adapter.save_ledgers(tmp_path, adapter.cases(), accounts, result["invalid_accounts"])
    for key in failures:
        with gzip.open(tmp_path / (key.replace(":", "_") + ".json.gz"), "rt") as stream:
            ledger = json.load(stream)
        assert ledger["metrics"] is None and ledger["cash_unknown_suffix_discarded"]
        assert ledger["failure"]["held_interval"]["symbol"] == "C00USDT"
        assert all(row["clock"] < first + 2 * MINUTE for row in ledger["curve"])


def test_untrusted_event_overrides_declared_ready_without_losing_observed_special(
    tmp_path, small_calendar
):
    _, _, _, _, universe, _ = small_calendar
    events, coverage = cash_fixture(tmp_path, small_calendar)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    for row in rows:
        if row["symbol"] == "C00USDT" and row["type"] == "Special":
            row["quality"]["trusted_typed_fields"] = False
            row["quality"]["identity_conflict"] = True
    events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    funding, ready, report = adapter.load_cash(events, coverage, universe)
    assert not ready["C00USDT", "2026-08"]
    assert report["observed_types"]["Special"] == 40
    assert report["trusted_cash_types"]["Special"] == 39
    assert not any(row["symbol"] == "C00USDT" and row["rate_type"] == "Special" for row in funding)


def test_missing_held_price_invalidates_only_affected_24_accounts(tmp_path, small_calendar):
    _, first, second, end, universe, predictions = small_calendar
    events, coverage = cash_fixture(tmp_path, small_calendar)
    funding, _, _ = adapter.load_cash(events, coverage, universe)
    schedules, _ = adapter.build_schedules(predictions, universe)
    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        synthetic_frames(small_calendar, missing=True),
        funding,
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    assert len(result["invalid_accounts"]) == len(result["metrics"]) == 24
    assert all(adapter.ASSETS[0] in key for key in result["invalid_accounts"])
    adapter.save_ledgers(tmp_path, adapter.cases(), accounts, result["invalid_accounts"])
    for key in result["invalid_accounts"]:
        with gzip.open(tmp_path / (key.replace(":", "_") + ".json.gz"), "rt") as stream:
            ledger = json.load(stream)
        assert ledger["metrics"] is None and ledger["ending_state"]["positions"]
        assert ledger["fills"] and ledger["curve"]


def test_earlier_unknown_cash_trims_already_failed_unclosed_positions(tmp_path, small_calendar):
    _, first, second, end, universe, predictions = small_calendar
    events, coverage = cash_fixture(tmp_path, small_calendar, unknown_symbol="C00USDT")
    funding, ready, _ = adapter.load_cash(events, coverage, universe)
    schedules, _ = adapter.build_schedules(predictions, universe)
    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        synthetic_frames(small_calendar, missing=True),
        funding,
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    original_failures = dict(result["invalid_accounts"])
    assert len(original_failures) == 24
    failures = adapter.reject_unknown_cash(accounts, result, ready)
    assert len(failures) == 24
    adapter.save_ledgers(tmp_path, adapter.cases(), accounts, result["invalid_accounts"])
    for key, failure in failures.items():
        assert failure["prior_execution_failure"] == original_failures[key]
        assert failure["held_interval"]["exit_reason"] == "unresolved_position_at_failure"
        assert failure["at"] == first + 2 * MINUTE < original_failures[key]["at"]
        with gzip.open(tmp_path / (key.replace(":", "_") + ".json.gz"), "rt") as stream:
            ledger = json.load(stream)
        assert ledger["cash_unknown_suffix_discarded"] and ledger["ending_state"] is None
        assert ledger["last_valid_curve"]["clock"] < failure["at"]
        assert all(row["clock"] < failure["at"] for row in ledger["curve"])


def test_four_hour_risk_halt_and_capital_are_not_reset_at_refit(small_calendar):
    _, first, second, end, universe, predictions = small_calendar
    schedules, _ = adapter.build_schedules(predictions, universe)

    def frames():
        for clock, market in synthetic_frames(small_calendar):
            if clock >= first + 5 * MINUTE:
                market["C18USDT"] = market["C19USDT"] = (10.0, 10.0, True, True)
            yield clock, market

    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        frames(),
        [],
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    assert not result["invalid_accounts"]
    for key, account in accounts.items():
        if adapter.ASSETS[0] in key:
            assert account.halted and len(account.trades) == 4
            assert {trade["reason"] for trade in account.trades} == {"risk"}
            assert all(fill["clock"] < second for fill in account.fills)
            assert result["metrics"][key]["historical_blocks"][0]["ending_state"]["halted"]


def test_no_trade_keeps_48_complete_flat_accounts_even_if_cash_unknown(tmp_path, small_calendar):
    _, first, second, end, universe, predictions = small_calendar
    for name in adapter.TRADING_MODELS:
        mask = predictions.common_member_gate & predictions.clock_ms.ge(first)
        predictions.loc[mask, f"prediction_{name}"] = 0.0
    schedules, _ = adapter.build_schedules(predictions, universe)
    events, coverage = cash_fixture(tmp_path, small_calendar, unknown_symbol="C00USDT")
    funding, ready, _ = adapter.load_cash(events, coverage, universe)
    accounts, result = drive_accounts(
        adapter.cases(),
        schedules,
        synthetic_frames(small_calendar),
        funding,
        (first, second, end),
        holding_minutes=240,
        timed_exit_at_close=False,
    )
    assert not adapter.reject_unknown_cash(accounts, result, ready)
    assert len(result["metrics"]) == 48
    for account in accounts.values():
        assert not account.fills and len(account.curve) == (end - first) // MINUTE + 1
        assert account.curve[-1]["equity"] == 1000


@pytest.mark.parametrize(
    "holding,at_close",
    [(0, True), (-1, False), (1.0, True), (True, True), (240, 1), (240, np.bool_(False))],
)
def test_driver_options_are_strict(holding, at_close):
    with pytest.raises(ValueError):
        drive_accounts(
            {"a": ("m", "a", "f", 3)},
            {},
            iter(()),
            [],
            (0, MINUTE),
            holding_minutes=holding,
            timed_exit_at_close=at_close,
        )


def test_default_driver_matches_explicit_20_close_byte_semantics():
    definitions = {"case": ("fixed", "asset", "legacy", 3)}
    first, last = 100 * MINUTE, 130 * MINUTE
    schedules = {("fixed", "asset", "legacy", first): [("L", 1, 0.003), ("S", -1, 0.003)]}

    def frames():
        return (
            (clock, {"L": (100.0, 100.0, True, True), "S": (100.0, 100.0, True, True)})
            for clock in range(first, last + MINUTE, MINUTE)
        )

    a, x = drive_accounts(definitions, schedules, frames(), [], (first, last))
    b, y = drive_accounts(
        definitions,
        schedules,
        frames(),
        [],
        (first, last),
        holding_minutes=20,
        timed_exit_at_close=True,
    )
    assert x == y and a["case"].trades == b["case"].trades and a["case"].curve == b["case"].curve


def source_fixture(tmp_path, monkeypatch, calendar):
    start, first, _, end, universe, predictions = calendar
    source, panel, capture, typed, output = (
        tmp_path / name for name in ("study", "panel", "capture", "typed", "output")
    )
    for path in (source, panel, capture, typed):
        path.mkdir()
    (capture / "inputs").mkdir()
    original = [symbol for symbol, c in universe.items() if c["original_748_member"]]
    metadata = [
        {
            "symbol": symbol,
            "snapshot_symbol_found": c["snapshot_symbol_found"],
            "original_748_member": c["original_748_member"],
            "snapshot": None
            if not c["snapshot_symbol_found"]
            else {
                "symbol": symbol,
                "contractType": "PERPETUAL"
                if c["asset_class"] == adapter.ASSETS[0]
                else "TRADIFI_PERPETUAL",
                "onboardDate": c["onboard_ms"],
                "deliveryDate": c["delivery_ms"],
            },
        }
        for symbol, c in universe.items()
    ]
    save(capture / "inputs/eligibility_symbol_metadata.json", metadata)
    save(
        capture / "capture_plan.json",
        {
            "schema": "ta-fixed-historical-public-capture.v1",
            "start_ms": start,
            "end_exclusive_ms": end,
            "symbols": list(universe),
            "original_universe": original,
        },
    )
    save(capture / "result.json", {"capture_completed": True})
    save(
        capture / "receipts.json",
        {str(p.relative_to(capture)): sha(p) for p in capture.rglob("*") if p.is_file()},
    )
    events, coverage = cash_fixture(typed, calendar)
    save(
        typed / "protocol.json",
        {
            "schema": "ta-historical-typed-funding-view.v1",
            "source_root": str(capture),
            "input_identities": {str(capture / "receipts.json"): sha(capture / "receipts.json")},
        },
    )
    save(typed / "result.json", {"status": "COMPLETED_WITH_UNKNOWN"})
    save(typed / "receipts.json", {p.name: sha(p) for p in typed.iterdir()})
    save(
        panel / "protocol.json",
        {
            "schema": "ta-historical-funding-panel.v1",
            "start_ms": start,
            "end_ms": end,
            "study_plan_sha256": sha(adapter.study.PLAN),
            "universe": universe,
            "capture_root": str(capture),
            "funding_view_root": str(typed),
            "capture_receipts_sha256": sha(capture / "receipts.json"),
            "funding_view_receipts_sha256": sha(typed / "receipts.json"),
            "input_identities": {str(path): sha(path) for path in (events, coverage)},
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    save(
        panel / "result.json",
        {
            "status": "COMPLETED_WITH_UNKNOWN",
            "protocol_sha256": sha(panel / "protocol.json"),
        },
    )
    save(panel / "receipts.json", {p.name: sha(p) for p in panel.iterdir()})
    parent = {
        "schema": "ta-extended-funding-study.v1",
        "start_ms": start,
        "end_ms": end,
        "horizon_minutes": 240,
        "label_delay_ms": 241 * MINUTE,
        "refit_times_ms": adapter.REFITS,
        "features": adapter.study.FEATURES,
        "models": adapter.study.MODELS,
        "contrasts": adapter.study.CONTRASTS,
        "source_sha256": {str(adapter.study.PLAN.resolve()): sha(adapter.study.PLAN)},
        "source": str(panel),
        "universe": universe,
        "input_identities": {
            str(panel / name): sha(panel / name)
            for name in ("protocol.json", "result.json", "receipts.json")
        },
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
    }
    save(source / "protocol.json", parent)
    save(
        source / "result.json",
        {
            "schema": "ta-extended-funding-regression-result.v1",
            "protocol_sha256": sha(source / "protocol.json"),
            "declared_tests": 24,
            "prediction_rows": len(predictions),
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    predictions["target_240"] = 1e100
    predictions["prediction_controls"] = 1e100
    predictions["holding_path_observed_240"] = False
    predictions.to_parquet(source / "predictions.parquet", index=False)
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    monkeypatch.setattr(
        adapter.historical_panel,
        "minute_sources",
        lambda bound, plan: {symbol: [] for symbol in universe},
    )

    def load(symbol, sources):
        clocks = np.arange(start + MINUTE, end + MINUTE, MINUTE)
        rows = pd.DataFrame({"open": 100.0, "close": 100.0}, index=clocks)
        exit_frame = first + 242 * MINUTE
        if symbol != "UNKNOWNUSDT":
            rows.loc[exit_frame, "open"] = (
                110.0 if int(symbol[1:3]) >= 18 else 90.0 if int(symbol[1:3]) <= 1 else 100.0
            )
        return rows

    monkeypatch.setattr(adapter.historical_panel, "load_minutes", load)
    return source, output, universe


def test_adapter_runs_48_with_physical_whitelist_and_no_labels(
    tmp_path, monkeypatch, small_calendar
):
    source, output, universe = source_fixture(tmp_path, monkeypatch, small_calendar)
    original = pd.read_parquet
    calls = []

    def spy(path, **kwargs):
        columns = kwargs["columns"]
        calls.append(columns)
        assert not {"target_240", "prediction_controls", "holding_path_observed_240"} & set(columns)
        return original(path, **kwargs)

    monkeypatch.setattr(adapter.pd, "read_parquet", spy)
    result = adapter.run(source, output)
    assert calls == [list(adapter.PREDICTION_COLUMNS)]
    assert len(result["metrics"]) == len(result["financial_proofs"]) == 48
    assert not result["future_labels_loaded"] and not result["invalid_accounts"]
    assert len(list(output.glob("*.json.gz"))) == 48
    assert not list(output.glob(".minute-market-*"))
    assert set(read(output / "market_coverage.json")) == set(universe)
    verify_receipts(output)


def test_market_read_failure_seals_48_partial_accounts(tmp_path, monkeypatch, small_calendar):
    source, output, _ = source_fixture(tmp_path, monkeypatch, small_calendar)
    monkeypatch.setattr(
        adapter.historical_panel,
        "load_minutes",
        lambda *_: (_ for _ in ()).throw(OSError("合成规范表读盘失败")),
    )
    with pytest.raises(ValueError, match="分钟输入失败"):
        adapter.run(source, output)
    assert len(list(output.glob("*.json.gz"))) == 48 and not (output / "result.json").exists()
    assert len(read(output / "partial_result.json")["invalid_accounts"]) == 48
    verify_receipts(output)


def test_failure_before_execution_keeps_48_unstarted_ledgers(tmp_path, monkeypatch, small_calendar):
    source, output, _ = source_fixture(tmp_path, monkeypatch, small_calendar)
    monkeypatch.setattr(
        adapter, "build_schedules", lambda *_: (_ for _ in ()).throw(ValueError("合成调度失败"))
    )
    with pytest.raises(ValueError, match="合成调度失败"):
        adapter.run(source, output)
    for path in output.glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            ledger = json.load(stream)
        assert not ledger["execution_started"] and ledger["metrics"] is None
    assert len(list(output.glob("*.json.gz"))) == 48
    verify_receipts(output)


@pytest.mark.parametrize("source_kind", ["prediction", "typed", "capture"])
def test_changed_sealed_input_fails_before_execution_and_preserves_48_cases(
    tmp_path, monkeypatch, small_calendar, source_kind
):
    source, output, _ = source_fixture(tmp_path, monkeypatch, small_calendar)
    parent = read(source / "protocol.json")
    panel = read(parent["source"] + "/protocol.json")
    path = {
        "prediction": source / "predictions.parquet",
        "typed": tmp_path / "typed/typed_cash_events.jsonl",
        "capture": tmp_path / "capture/capture_plan.json",
    }[source_kind]
    with path.open("ab") as stream:
        stream.write(b" ")
    assert panel["capture_root"] == str(tmp_path / "capture")
    with pytest.raises(ValueError):
        adapter.run(source, output)
    assert len(list(output.glob("*.json.gz"))) == 48
    assert not (output / "result.json").exists()
    verify_receipts(output)


@pytest.mark.parametrize("failed_kind", ["study_failure", "study_result", "panel_result"])
def test_failed_upstream_is_rejected_even_when_resealed(
    tmp_path, monkeypatch, small_calendar, failed_kind
):
    source, output, _ = source_fixture(tmp_path, monkeypatch, small_calendar)
    if failed_kind == "study_failure":
        save(source / "failure.json", {"reason": "末验失败"})
    elif failed_kind == "study_result":
        value = read(source / "result.json")
        value["declared_tests"] = 23
        (source / "result.json").write_text(json.dumps(value))
    else:
        panel = tmp_path / "panel"
        value = read(panel / "result.json")
        value["status"] = "FAILED_INTEGRITY"
        (panel / "result.json").write_text(json.dumps(value))
        (panel / "receipts.json").write_text(
            json.dumps({p.name: sha(p) for p in panel.iterdir() if p.name != "receipts.json"})
        )
    (source / "receipts.json").write_text(
        json.dumps({p.name: sha(p) for p in source.iterdir() if p.name != "receipts.json"})
    )
    with pytest.raises(ValueError):
        adapter.run(source, output)
    assert len(list(output.glob("*.json.gz"))) == 48
    verify_receipts(output)


def test_final_input_drift_invalidates_all_48_numeric_ledgers(
    tmp_path, monkeypatch, small_calendar
):
    source, output, _ = source_fixture(tmp_path, monkeypatch, small_calendar)
    original = adapter.paired_forecast.drive_accounts

    def drift_after_accounts(*args, **kwargs):
        result = original(*args, **kwargs)
        with (source / "predictions.parquet").open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(adapter.paired_forecast, "drive_accounts", drift_after_accounts)
    with pytest.raises(ValueError, match="运行期间改变"):
        adapter.run(source, output)
    assert not (output / "result.json").exists()
    for path in output.glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            ledger = json.load(stream)
        assert ledger["metrics"] is None and not ledger["numeric_conclusions_allowed"]
        assert ledger["last_valid_curve"] is None
    assert len(list(output.glob("*.json.gz"))) == 48
    verify_receipts(output)


def test_market_maps_only_four_arrays_and_uses_open_stamp_qualification(
    tmp_path, monkeypatch, small_calendar
):
    start, first, _, end, universe, _ = small_calendar

    def load(symbol, sources):
        return pd.DataFrame(
            {"open": 100.0, "close": 101.0, "unused": 1e100},
            index=np.arange(start + MINUTE, end + MINUTE, MINUTE),
        )

    monkeypatch.setattr(adapter.historical_panel, "load_minutes", load)
    universe["C00USDT"]["delivery_ms"] = first
    universe["C01USDT"]["onboard_ms"] = first - MINUTE
    report = {}
    frames = adapter.market_frames(universe, {s: [] for s in universe}, tmp_path, report)
    clock, market = next(frames)
    assert clock == first and len(market) == len(universe)
    assert market["C00USDT"][2] and not market["C01USDT"][2]
    _, following = next(frames)
    assert not following["C00USDT"][2] and following["C01USDT"][2]
    assert {path.name for path in tmp_path.iterdir()} == {"prices.npy", "flags.npy"}
    frames.close()
    estimate = adapter.memory_estimate(909)
    assert estimate["market_fields"] == ["open", "close", "observed", "qualification"]


def test_full_run_memory_budget_is_under_three_gibibytes():
    estimate = adapter.memory_estimate(909)
    assert estimate["curve_points"] == 3_179_568
    assert estimate["market_mapped_bytes"] == 1_083_835_242
    assert estimate["working_set_estimate_bytes"] < 3 * 1024**3


def test_bounded_curve_scale_rss_smoke():
    code = """import gc,json,resource,sys
from tradingagents.research.quant.backtesting.extended_funding import cases
from tradingagents.research.quant.backtesting.paired_forecast import drive_accounts
gc.collect()
unit=1 if sys.platform=="darwin" else 1024
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
accounts,result=drive_accounts(cases(),{},((i*60000,{}) for i in range(2001)),[],(0,60000000,120000000),holding_minutes=240,timed_exit_at_close=False)
after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
points=sum(len(a.curve) for a in accounts.values())
print(json.dumps({"points":points,"delta":max(0,after-before),"per_point":max(0,after-before)/points,"valid":len(result["financial_proofs"])}))"""
    process = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    report = json.loads(process.stdout)
    assert report["points"] == 96_048 and report["valid"] == 48
    assert report["per_point"] <= 384
