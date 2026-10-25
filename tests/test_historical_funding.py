"""显式类型资金费视图只用合成封存原文，不发HTTP或计算收益。"""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tradingagents.dataflows.research import (
    historical_capture as capture,
    historical_funding as view,
)


def event(symbol, stamp, kind="Regular", rate="0.00010000", mark="100", **extra):
    return {
        "symbol": symbol,
        "fundingTime": stamp,
        "fundingRate": rate,
        "markPrice": mark,
        "rateType": kind,
        **extra,
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def reseal(source):
    path = source / "receipts.json"
    write(
        path,
        {
            str(p.relative_to(source)): view.sha(p)
            for p in source.rglob("*")
            if p.is_file() and p != path
        },
    )


def latest(source, symbol):
    return sorted((source / "rest_jobs" / symbol).glob("attempt_*.json"))[-1]


def replace_rest(source, symbol, events):
    """合成一份摘要自洽但语义可能错误的来源，核验不止依赖总SHA。"""
    path = latest(source, symbol)
    record = view.read(path)
    body = next(p for p in record["files"] if p.endswith(".body"))
    write(source / body, events)
    receipt_path = Path(body).with_suffix(".json").as_posix()
    receipt = view.read(source / receipt_path)
    receipt.update(sha256=view.sha(source / body), bytes=(source / body).stat().st_size)
    write(source / receipt_path, receipt)
    write(source / record["events"], events)
    record["pager"]["checkpoint"]["closed_events"] = events
    for name in (body, receipt_path, record["events"]):
        record["files"][name] = view.sha(source / name)
    write(path, record)
    reseal(source)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    root.mkdir()
    symbols = ["MATCHUSDT", "牛来USDT", "EMPTYUSDT"]
    monkeypatch.setattr(view, "EXPECTED_SYMBOLS", len(symbols))
    first, september = view.bounds("2026-08")[0], view.bounds("2026-09")[0]
    events = {
        symbols[0]: [event(symbols[0], first + 1), event(symbols[0], september + 1)],
        symbols[1]: [
            event(symbols[1], first + 1),
            event(symbols[1], first + 2, "Special", "-0.002", retained_extra="保留"),
            event(symbols[1], september + 1),
        ],
        symbols[2]: [],
    }
    tasks, responses = [], {}
    for symbol in symbols:
        for month in view.MONTHS:
            for kind in ("1m", "fundingRate"):
                key = capture.archive_key(kind, symbol, month)
                listed = kind == "fundingRate" and symbol != "EMPTYUSDT"
                task = {
                    "id": f"{kind}_{symbol}_{month}",
                    "symbol": symbol,
                    "month": month,
                    "kind": kind,
                    "key": key,
                    "listed": listed,
                    "checksum_key_listed": listed,
                    "initial_status": "PENDING" if listed else "NOT_LISTED",
                    "listed_bytes": None,
                }
                if listed:
                    start, end = view.bounds(month)
                    rows = [
                        [str(e["fundingTime"]), "8", e["fundingRate"]]
                        for e in events[symbol]
                        if start <= e["fundingTime"] < end
                    ]
                    text = io.StringIO()
                    writer = csv.writer(text)
                    writer.writerow(view.ARCHIVE_FIELDS)
                    writer.writerows(rows)
                    output = io.BytesIO()
                    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                        bundle.writestr(Path(key).name[:-4] + ".csv", text.getvalue())
                    raw = output.getvalue()
                    task["listed_bytes"] = len(raw)
                    responses[capture.archive_url(key)] = raw
                    responses[capture.archive_url(key + ".CHECKSUM")] = (
                        f"{hashlib.sha256(raw).hexdigest()}  {Path(key).name}".encode()
                    )
                tasks.append(task)
    plan = {
        "schema": "ta-fixed-historical-public-capture.v1",
        "symbols": symbols,
        "original_universe": symbols,
        "archive_tasks": tasks,
        "rest_symbols": symbols,
        "months": view.MONTHS,
        "start_ms": first,
        "end_exclusive_ms": view.bounds("2026-09")[1],
        "executed_source": {
            "historical_rest.py": view.sha(Path(view.capture_pages.__code__.co_filename))
        },
        "orders_enabled": False,
        "leader_labels_read": False,
        "returns_evaluated": False,
    }
    capture.save(root / "capture_plan.json", plan)
    client = capture.Client(root, plan, transport=lambda url, maximum: (200, {}, responses[url]))
    for task in tasks:
        assert capture.capture_archive(root, task, client)["status"] in {"VERIFIED", "NOT_LISTED"}
    for symbol in symbols:
        client = capture.Client(
            root,
            plan,
            rest=True,
            transport=lambda url, maximum, s=symbol: (200, {}, json.dumps(events[s]).encode()),
            sleep=lambda seconds: None,
        )
        assert capture.capture_rest(root, symbol, plan, client)["status"] in {
            "API_RANGE_VERIFIED",
            "API_EMPTY_RANGE",
        }
    capture.save(root / "funding_month_comparisons.json", capture.compare_all(root, plan))
    reseal(root)
    return root


def test_complete_population_and_special_explanation_keep_original_status(source, tmp_path):
    output = tmp_path / "view"
    result = view.run(source, output)
    assert result["status"] == "COMPLETED_WITH_UNKNOWN" and result["symbol_months"] == 6
    assert result["original_status_counts"] == {
        "MATCHED": 3,
        "MISMATCH": 1,
        "UNPROVEN_MISSING_SOURCE": 2,
    }
    assert result["regular_plus_special_status_counts"] == {
        "MATCHED": 4,
        "UNKNOWN_MISSING_SOURCE": 2,
    }
    assert (
        result["month_ready_under_source_assumptions"] == 4
        and result["original_regular_mismatch_explained_by_special"] == 1
    )
    coverage = view.read(output / "coverage.json")
    assert len(coverage) == 6
    assert all(
        not x["known_zero_cash"] and not x["independent_completeness_proven"] for x in coverage
    )
    empty = [x for x in coverage if x["symbol"] == "EMPTYUSDT"]
    assert all(
        x["api_empty_unknown"] and not x["month_ready_under_source_assumptions"] for x in empty
    )
    rows = [
        json.loads(line) for line in (output / "typed_cash_events.jsonl").read_text().splitlines()
    ]
    special = next(x for x in rows if x["type"] == "Special")
    assert (
        special["raw"]["retained_extra"] == "保留"
        and special["time_ms"] == view.bounds("2026-08")[0] + 2
    )
    assert special["quality"]["trusted_typed_fields"] is True
    assert result["typed_event_counts"] == {"Regular": 4, "Special": 1}
    manifest = view.read(output / "receipts.json")
    assert set(manifest) == {
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    } - {"receipts.json"}
    assert all(view.sha(output / name) == expected for name, expected in manifest.items())


@pytest.mark.parametrize(
    "changes",
    [
        {"rateType": "Unknown"},
        {"fundingRate": "NaN"},
        {"fundingRate": "Infinity"},
        {"markPrice": "NaN"},
        {"markPrice": "0"},
        {"markPrice": "-1"},
        {"symbol": "WRONGUSDT"},
    ],
)
def test_invalid_explicit_events_are_unknown_and_other_symbols_still_seal(
    source, tmp_path, changes
):
    record = view.read(latest(source, "MATCHUSDT"))
    rows = view.read(source / record["events"])
    rows[0].update(changes)
    replace_rest(source, "MATCHUSDT", rows)
    output = tmp_path / "view"
    result = view.run(source, output)
    assert result["status"] == "COMPLETED_WITH_UNKNOWN" and result["symbol_months"] == 6
    assert result["error_count"] == 1
    failed = [x for x in view.read(output / "coverage.json") if x["symbol"] == "MATCHUSDT"]
    assert all(
        not x["rest_source_consistent"] and not x["month_ready_under_source_assumptions"]
        for x in failed
    )
    assert (output / "rejected_events.jsonl").read_text() and (output / "receipts.json").exists()


def test_raw_rest_and_normalized_event_values_must_match(source, tmp_path):
    path = latest(source, "MATCHUSDT")
    record = view.read(path)
    rows = view.read(source / record["events"])
    rows[0]["markPrice"] = "101"
    write(source / record["events"], rows)
    record["files"][record["events"]] = view.sha(source / record["events"])
    write(path, record)
    reseal(source)
    output = tmp_path / "view"
    view.run(source, output)
    assert any(
        "REST_RAW_NORMALIZED_MISMATCH" in e["error"] for e in view.read(output / "errors.json")
    )


def test_raw_archive_and_normalized_funding_rows_must_match(source, tmp_path):
    path = source / "archive_jobs/fundingRate_MATCHUSDT_2026-08/attempt_0000.json"
    record = view.read(path)
    rows = view.read(source / record["funding_rows"])
    rows[0]["last_funding_rate"] = "0.02"
    write(source / record["funding_rows"], rows)
    record["files"][record["funding_rows"]] = view.sha(source / record["funding_rows"])
    write(path, record)
    reseal(source)
    output = tmp_path / "view"
    result = view.run(source, output)
    assert result["symbol_months"] == 6 and result["error_count"] == 1
    assert any(
        "ARCHIVE_RAW_NORMALIZED_MISMATCH" in e["error"] for e in view.read(output / "errors.json")
    )


@pytest.mark.parametrize("changes", [{"fundingRate": "0.02"}, {"markPrice": "101"}])
def test_same_clock_different_type_is_allowed_but_same_type_is_never_deduplicated(
    source, tmp_path, changes
):
    symbol = "MATCHUSDT"
    first = view.bounds("2026-08")[0]
    rows = [
        event(symbol, first + 1),
        {**event(symbol, first + 1), **changes},
        event(symbol, view.bounds("2026-09")[0] + 1),
    ]
    replace_rest(source, symbol, rows)
    output = tmp_path / "view"
    view.run(source, output)
    cash = [json.loads(x) for x in (output / "typed_cash_events.jsonl").read_text().splitlines()]
    conflicts = [x for x in cash if x["symbol"] == symbol and x["time_ms"] == first + 1]
    assert len(conflicts) == 2 and all(
        x["quality"]["identity_conflict"] and not x["quality"]["trusted_typed_fields"]
        for x in conflicts
    )
    coverage = next(
        x
        for x in view.read(output / "coverage.json")
        if x["symbol"] == symbol and x["month"] == "2026-08"
    )
    assert (
        not coverage["month_ready_under_source_assumptions"]
        and coverage["identity_conflict_count"] == 2
    )
    assert view.event_conflicts([event(symbol, first), event(symbol, first, "Special")]) == set()
    assert view.event_conflicts([event(symbol, first), event(symbol, first)]) == {0, 1}


def test_month_boundary_ms_and_decimal_value_multisets_are_exact():
    symbol = "EDGEUSDT"
    first, end = view.bounds("2026-08")
    rows = [
        event(symbol, first, rate="1e-4"),
        event(symbol, end - 1, "Special", "-0.002"),
        event(symbol, end, rate="0.5"),
    ]
    archive = {
        "source_consistent": True,
        "rows": [
            {"calc_time": str(first), "last_funding_rate": "0.00010000"},
            {"calc_time": str(end - 1), "last_funding_rate": "-0.0020"},
        ],
    }
    rest = {
        "events": rows,
        "source_consistent": True,
        "conflicts": set(),
        "record": {"status": "API_RANGE_VERIFIED"},
    }
    result = view.compare_month(symbol, "2026-08", {"status": "MISMATCH"}, archive, rest, [])
    assert (
        result["regular_plus_special_status"] == "MATCHED"
        and result["observed_regular_events"] == result["observed_special_events"] == 1
    )
    assert result["original_comparison"]["status"] == "MISMATCH"


def test_wrong_source_paths_and_source_byte_changes_fail_explicitly(source, tmp_path):
    bound = view.BoundSource(source)
    for name in ("../outside.json", "/outside.json", "rest_jobs\\outside.json"):
        with pytest.raises(ValueError, match="PATH"):
            bound.path(name)
    path = latest(source, "MATCHUSDT")
    record = view.read(path)
    record["files"]["../outside.json"] = "0" * 64
    write(path, record)
    reseal(source)
    output = tmp_path / "view"
    view.run(source, output)
    assert any("TASK_FILE_WRONG_SOURCE" in e["error"] for e in view.read(output / "errors.json"))
    (source / "capture_plan.json").write_text("changed")
    failed = view.run(source, tmp_path / "bad_view")
    assert failed["status"] == "FAILED_INTEGRITY" and (tmp_path / "bad_view/receipts.json").exists()


def test_source_change_at_finish_preserves_partial_failure_receipts(source, tmp_path, monkeypatch):
    def changed(paths, expected):
        raise ValueError("EXECUTED_SOURCE_CHANGED:synthetic")

    monkeypatch.setattr(view, "assert_stable", changed)
    output = tmp_path / "partial"
    result = view.run(source, output)
    assert result["status"] == "FAILED_INTEGRITY"
    assert not (output / "typed_cash_events.jsonl").exists()
    assert (output / "partial_typed_cash_events.jsonl").exists() and len(
        view.read(output / "partial_coverage.json")
    ) == 6
    assert (output / "receipts.json").exists()
    fake = tmp_path / "fake.py"
    fake.write_text("original")
    expected = {"fake": view.sha(fake)}
    fake.write_text("changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        # 使用原函数而非修改任何真实模块文件。
        monkeypatch.undo()
        view.assert_stable({"fake": fake}, expected)


def test_same_clock_regular_and_special_both_survive_without_type_inference(source, tmp_path):
    first = view.bounds("2026-08")[0]
    rows = [
        event("MATCHUSDT", first + 1),
        event("MATCHUSDT", first + 1, "Special", "-0.02"),
        event("MATCHUSDT", view.bounds("2026-09")[0] + 1),
    ]
    replace_rest(source, "MATCHUSDT", rows)
    output = tmp_path / "view"
    view.run(source, output)
    observed = [
        json.loads(x) for x in (output / "typed_cash_events.jsonl").read_text().splitlines()
    ]
    tied = [x for x in observed if x["symbol"] == "MATCHUSDT" and x["time_ms"] == first + 1]
    assert {x["type"] for x in tied} == {"Regular", "Special"}
    assert len(tied) == 2 and all(x["quality"]["trusted_typed_fields"] for x in tied)
    month = next(
        x
        for x in view.read(output / "coverage.json")
        if x["symbol"] == "MATCHUSDT" and x["month"] == "2026-08"
    )
    assert (
        month["regular_plus_special_status"] == "MISMATCH"
        and not month["month_ready_under_source_assumptions"]
    )


def test_exact_duplicate_rest_row_is_quarantined_not_deleted(source, tmp_path):
    path = latest(source, "MATCHUSDT")
    record = view.read(path)
    rows = view.read(source / record["events"])
    rows.insert(1, dict(rows[0]))
    replace_rest(source, "MATCHUSDT", rows)
    output = tmp_path / "view"
    view.run(source, output)
    rejected = [json.loads(x) for x in (output / "rejected_events.jsonl").read_text().splitlines()]
    assert len(rejected[0]["observed_normalized_events"]) == 3
    assert rejected[0]["trusted_typed_fields"] is False
    assert "UNRESOLVED_MULTIPLICITY" in rejected[0]["error"]


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example/funding?symbol=MATCHUSDT",
        "https://fapi.binance.com/fapi/v1/fundingRate?symbol=WRONGUSDT&startTime=1785542400000&endTime=1790812799999&limit=1000",
    ],
)
def test_request_url_must_match_frozen_public_source(source, tmp_path, url):
    path = latest(source, "MATCHUSDT")
    record = view.read(path)
    receipt = next(
        name
        for name in record["files"]
        if Path(name).parent.name == "requests" and name.endswith(".json")
    )
    value = view.read(source / receipt)
    value["url"] = url
    write(source / receipt, value)
    record["files"][receipt] = view.sha(source / receipt)
    write(path, record)
    reseal(source)
    output = tmp_path / "view"
    view.run(source, output)
    assert any("REST_QUERY_WRONG_SOURCE" in e["error"] for e in view.read(output / "errors.json"))


def test_empty_side_never_becomes_known_zero_or_month_ready():
    first = view.bounds("2026-09")[0]
    rest = {
        "events": [event("EMPTYMONTHUSDT", first + 1)],
        "record": {"status": "API_RANGE_VERIFIED"},
        "source_consistent": True,
        "conflicts": set(),
    }
    month = view.compare_month(
        "EMPTYMONTHUSDT",
        "2026-08",
        {"status": "UNKNOWN_EMPTY"},
        {"rows": [], "source_consistent": True},
        rest,
        [],
    )
    assert month["regular_plus_special_status"] == "UNKNOWN_EMPTY_SOURCE"
    assert (
        month["observed_regular_events"] == 0
        and not month["known_zero_cash"]
        and not month["month_ready_under_source_assumptions"]
    )
    rest.update(events=[], record={"status": "API_EMPTY_RANGE"})
    month = view.compare_month(
        "EMPTYMONTHUSDT",
        "2026-08",
        {"status": "UNKNOWN_EMPTY"},
        {"rows": [], "source_consistent": True},
        rest,
        [],
    )
    assert month["api_empty_unknown"] and not month["month_ready_under_source_assumptions"]


def test_source_byte_drift_observed_during_analysis_cannot_be_ignored(source):
    bound = view.BoundSource(source)
    bound.verify_all()
    name = "funding_month_comparisons.json"
    (source / name).write_text("changed")
    with pytest.raises(ValueError, match="BYTES_CHANGED"):
        bound.path(name)
    with pytest.raises(ValueError, match="SOURCE_RECEIPTS_CHANGED"):
        bound.verify_used()
