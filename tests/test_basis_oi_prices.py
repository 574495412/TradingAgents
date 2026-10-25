"""以合成公开分钟原文验证精度、完整人口、完成钟和来源封存。"""

import copy
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tradingagents.dataflows.research import basis_oi_prices as prices
from tradingagents.research.quant.artifacts import sha, verify_receipts

DAY_TEXT = "2026-09-28"
START = 1_790_553_600_000
SYMBOL = "牛来USDT"


def _row(stamp=START):
    return [
        stamp,
        "10.00",
        "12.000",
        "9",
        "11",
        "0",
        stamp + prices.MINUTE - 1,
        "0.000",
        0,
        "0",
        "0.0",
        "0",
    ]


def _payload(rows=None, symbol=SYMBOL, day=DAY_TEXT, **updates):
    left = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()) * 1000
    rows = [_row(left)] if rows is None else rows
    source = {
        "schema": prices.RAW_SCHEMA,
        "symbol": symbol,
        "start_ms": left,
        "end_ms": left + prices.DAY,
        "rows": rows,
        "rows_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
    }
    source.update(updates)
    return json.dumps(source, ensure_ascii=False).encode()


def _parse(rows=None, **updates):
    return prices.parse_raw(_payload(rows, **updates), SYMBOL, DAY_TEXT)


def _minutes(sources, clocks, metadata=None):
    return list(prices.minute_rows(SYMBOL, sources, metadata or {}, clocks))


@pytest.fixture
def bound_fixture(tmp_path, monkeypatch):
    """缩小纯合成工程格，不读任何真实协议或原数据。"""
    root = tmp_path / "synthetic_ta"
    root.mkdir()
    start = int(datetime(2026, 10, 13, tzinfo=UTC).timestamp()) * 1000
    end = start + 2 * prices.DAY
    symbols = ["BTCUSDT", SYMBOL]
    identities = {}
    for symbol in symbols:
        for day, base in (("2026-10-13", prices.DYNAMIC_ROOT), ("2026-10-14", prices.UNSEEN_ROOT)):
            left = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()) * 1000
            path = root / base / symbol / (day + ".json")
            path.parent.mkdir(parents=True)
            path.write_bytes(
                _payload(
                    [
                        _row(left),
                        _row(left + 5 * prices.MINUTE),
                        _row(left + prices.DAY - prices.MINUTE),
                    ],
                    symbol,
                    day,
                )
            )
            identities[str(path)] = sha(path)
    # 该额外身份故意无真实文件，证明不消费其他H01依赖。
    identities[str(root / "never_read_labels.parquet")] = "f" * 64
    protocol = root / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "ta-native-settlement-unwind-study.v1",
                "universe": {
                    symbol: {
                        "asset_class": "CRYPTO_PERPETUAL",
                        "captured_status": "SETTLING",
                        "onboard_ms": start + prices.MINUTE,
                        "delivery_ms": start + prices.DAY,
                    }
                    for symbol in symbols
                },
                "input_identities": identities,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prices, "ROOT", root)
    monkeypatch.setattr(prices, "START", start)
    monkeypatch.setattr(prices, "END", end)
    monkeypatch.setattr(prices, "EXPECTED_SYMBOLS", 2)
    monkeypatch.setattr(prices, "SOURCE_PROTOCOL", protocol)
    monkeypatch.setattr(prices, "SOURCE_PROTOCOL_SHA", sha(protocol))
    return root, protocol, start, end


def test_twelve_original_positions_and_completed_boundary_have_no_float_price():
    row = _row()
    row[1] = "10.123456789012345678901234567890123456789"
    parsed = _parse([row])
    assert list(parsed["records"]) == [START + prices.MINUTE]
    record = parsed["records"][START + prices.MINUTE]
    assert Decimal(record["values"]["contract_1m_open"]) == Decimal(row[1])
    assert record["values"]["contract_1m_high"] == "12"
    assert record["values"]["base_volume"] == "0"
    assert record["values"]["quote_volume"] == "0"
    assert record["values"]["trade_count"] == "0"
    assert set(record["values"]) == set(prices.FIELDS)
    assert record["open_ms"] == START
    assert record["source_refs"] == [DAY_TEXT + ":1"]
    assert all(isinstance(value, str) for value in record["values"].values())


@pytest.mark.parametrize("field", prices.FIELDS)
def test_missing_invalid_or_negative_fields_do_not_fill_zero(field):
    row = _row()
    position = prices.FIELD_POSITIONS[field]
    row[position] = -1 if field == "trade_count" else "-1"
    parsed = _parse([row])
    record = parsed["records"][START + prices.MINUTE]
    assert record["value_status"][field] == "INVALID"
    assert record["values"][field] is None
    assert parsed["anomalies"][0]["raw"][position] == row[position]
    if not field.startswith("contract_"):
        assert record["values"]["contract_1m_close"] == "11"


@pytest.mark.parametrize("bad", [1.2, True, "NaN", "Infinity", "1_000", "", None])
def test_price_float_boolean_or_unparseable_text_cannot_be_promoted_to_original_price(bad):
    row = _row()
    row[1] = bad
    record = _parse([row])["records"][START + prices.MINUTE]
    assert record["values"]["contract_1m_open"] is None
    assert record["value_status"]["contract_1m_open"] in {"EMPTY", "INVALID"}


def test_valid_original_zero_volume_and_count_are_not_missing():
    record = _parse()["records"][START + prices.MINUTE]
    assert all(record["values"][field] == "0" for field in prices.FIELDS[4:])
    assert all(record["value_status"][field] == "KNOWN_RAW_VALUE" for field in prices.FIELDS[4:])


@pytest.mark.parametrize(
    "values", [["13", "12", "9", "11"], ["10", "12", "11", "11"], ["10", "8", "9", "11"]]
)
def test_ohlc_relationship_conflict_keeps_clock_but_all_four_prices_unknown(values):
    row = _row()
    row[1:5] = values
    parsed = _parse([row])
    assert len(parsed["records"]) == 1
    record = parsed["records"][START + prices.MINUTE]
    assert all(record["values"][field] is None for field in prices.FIELDS[:4])
    assert record["values"]["base_volume"] == "0"
    assert parsed["anomalies"][0]["raw"][1:5] == values


def test_same_value_duplicates_keep_all_references_and_sort_stably():
    one, two = _row(), _row(START + prices.MINUTE)
    duplicate = copy.deepcopy(one)
    duplicate[1] = "1E+1"
    parsed = _parse([two, one, duplicate])
    assert list(parsed["records"]) == [START + prices.MINUTE, START + 2 * prices.MINUTE]
    record = parsed["records"][START + prices.MINUTE]
    assert record["values"]["contract_1m_open"] == "1E+1"
    assert record["duplicate_count"] == 2
    assert record["source_refs"] == [DAY_TEXT + ":2", DAY_TEXT + ":3"]
    assert parsed["profile"]["inverse_time_steps_in_input"] == 1
    assert parsed["profile"]["duplicate_extra_rows"] == 1


@pytest.mark.parametrize("bad", ["2", "", "bad"])
def test_duplicate_conflicts_affect_only_the_conflicting_field(bad):
    row = _row()
    row[7] = bad
    parsed = _parse([_row(), row])
    record = parsed["records"][START + prices.MINUTE]
    assert record["value_status"]["quote_volume"] == "DUPLICATE_CONFLICT"
    assert record["values"]["quote_volume"] is None
    assert record["values"]["base_volume"] == "0"
    assert record["values"]["contract_1m_close"] == "11"
    assert parsed["anomalies"][-1]["source_row_nos"] == [1, 2]
    assert parsed["anomalies"][-1]["reasons"] == ["DUPLICATE_CONFLICT"]


@pytest.mark.parametrize(
    "fault", ["offgrid", "close", "outside_day", "declared_range", "not_list", "no_clock"]
)
def test_identity_errors_are_preserved_as_anomalies_and_never_rounded(fault):
    row = _row()
    updates = {}
    if fault == "offgrid":
        row[0] += 1
    elif fault == "close":
        row[6] += 1
    elif fault == "outside_day":
        row = _row(START + prices.DAY)
    elif fault == "declared_range":
        updates["start_ms"] = START + prices.MINUTE
    elif fault == "not_list":
        row = {"not": "a row"}
    else:
        row = []
    parsed = _parse([row], **updates)
    assert parsed["records"] == {}
    assert parsed["profile"]["invalid_identity_rows"] == 1
    assert parsed["anomalies"][0]["raw"] == row
    assert parsed["profile"]["raw_rows"] == 1


def test_missing_trailing_field_keeps_other_economic_fields_and_raw_anomaly():
    parsed = _parse([_row()[:10]])
    record = parsed["records"][START + prices.MINUTE]
    assert record["value_status"]["taker_buy_quote_volume"] == "EMPTY"
    assert record["values"]["contract_1m_close"] == "11"
    assert "FIELD_COUNT_MISMATCH" in parsed["anomalies"][0]["reasons"]


@pytest.mark.parametrize("fault", ["symbol", "rows_sha", "interval"])
def test_source_identity_drift_is_fatal_not_a_missing_clock(fault):
    updates = (
        {"symbol": "OTHER"}
        if fault == "symbol"
        else {"rows_sha256": "f" * 64}
        if fault == "rows_sha"
        else {"end_ms": START}
    )
    with pytest.raises(ValueError) as caught:
        prices.parse_raw(_payload(**updates), SYMBOL, DAY_TEXT)
    assert not isinstance(caught.value, prices.RawFormatError)


@pytest.mark.parametrize(
    "payload", [b"not json", b'{"schema":1,"schema":2}', b'{"rows":NaN}', b"[]", b"{}"]
)
def test_file_structure_error_is_explicit(payload):
    with pytest.raises(prices.RawFormatError):
        prices.parse_raw(payload, SYMBOL, DAY_TEXT)


def test_exact_completed_clock_has_no_fill_and_keeps_end_boundary_in_source():
    parsed = _parse([_row(START + 4 * prices.MINUTE), _row(START + prices.DAY - prices.MINUTE)])
    clocks = [START, START + 5 * prices.MINUTE, START + 6 * prices.MINUTE]
    rows = _minutes([parsed], clocks)
    assert [row["observed"] for row in rows] == [False, True, False]
    assert rows[0]["price_source_day"] == "2026-09-27"
    assert rows[0]["price_source_status"] == "NO_BOUND_SOURCE"
    assert rows[1]["bar_open_ms"] == START + 4 * prices.MINUTE
    assert rows[2]["contract_1m_close"] is None
    assert rows[2]["contract_1m_close_value_status"] == "MISSING_CLOCK"
    assert START + prices.DAY in parsed["records"]


def test_cross_day_lookup_preserves_previous_file_and_does_not_reset_clock():
    previous = prices.parse_raw(
        _payload([_row(START - prices.MINUTE)], day="2026-09-27"), SYMBOL, "2026-09-27"
    )
    current = _parse([_row()])
    rows = _minutes([previous, current], [START, START + prices.MINUTE])
    assert all(row["observed"] for row in rows)
    assert rows[0]["price_source_day"] == "2026-09-27"
    assert rows[1]["price_source_day"] == DAY_TEXT


def test_static_qualification_does_not_remove_known_price_and_unknown_dates_stay_unknown():
    parsed = _parse([_row()])
    qualified = {
        "onboard_ms": START + prices.DAY,
        "delivery_ms": START + 2 * prices.DAY,
        "captured_status": "SETTLING",
    }
    row = _minutes([parsed], [START + prices.MINUTE], qualified)[0]
    assert row["late_static_qualified_at_open"] is False
    assert row["contract_1m_close"] == "11"
    assert row["observed"] is True
    assert row["historical_pti_proven"] is False
    unknown = _minutes([parsed], [START + prices.MINUTE])[0]
    assert unknown["late_static_qualified_at_open"] is None
    assert unknown["late_static_qualification_status"] == "UNKNOWN"


def test_future_perturbation_and_inverse_order_do_not_change_older_values():
    original = [_row(), _row(START + 5 * prices.MINUTE)]
    future = copy.deepcopy(original)
    future[1][1:5] = ["99", "100", "90", "95"]
    future[1][5] = "999999999999.000000000000000001"
    clocks = [START + prices.MINUTE, START + 6 * prices.MINUTE]
    before = _minutes([_parse(original)], clocks)
    after = _minutes([_parse(future)], clocks)
    assert before[0] == after[0]
    assert before[1] != after[1]
    reverse = _minutes([_parse(list(reversed(original)))], clocks)
    assert [{field: row[field] for field in prices.FIELDS} for row in before] == [
        {field: row[field] for field in prices.FIELDS} for row in reverse
    ]


def test_multiple_overlapping_sources_use_per_field_consensus_not_first_last():
    first, conflicting = _parse(), _row()
    conflicting[7] = "8"
    second = _parse([conflicting])
    row = _minutes([first, second], [START + prices.MINUTE])[0]
    assert row["price_duplicate_count"] == 2
    assert row["quote_volume"] is None
    assert row["quote_volume_value_status"] == "DUPLICATE_CONFLICT"
    assert row["contract_1m_close"] == "11"


@pytest.mark.parametrize(
    "clocks", [[START, START], [START + 1], [START + prices.MINUTE, START], [True], [float(START)]]
)
def test_invalid_snapshot_calendar_is_rejected(clocks):
    with pytest.raises(ValueError):
        _minutes([], clocks)


def test_source_status_and_symbol_conflicts_are_rejected():
    parsed = _parse()
    altered = copy.deepcopy(parsed)
    altered["symbol"] = "OTHER"
    with pytest.raises(ValueError):
        _minutes([altered], [START])
    altered = copy.deepcopy(parsed)
    altered["source_status"] = "PARSE_FAILED"
    with pytest.raises(ValueError):
        _minutes([altered], [START])


def test_binding_selects_only_complete_original_sources_and_does_not_follow_reuse(bound_fixture):
    _, protocol, _, _ = bound_fixture
    bound = prices.bind_inputs(protocol)
    assert len(bound.inputs) == 5
    assert len(bound.symbols) == 2
    assert all(len(days) == 2 for days in bound.sources.values())
    assert not any("never_read" in path for path in bound.inputs)
    source = bound.consume(SYMBOL, "2026-10-13")
    assert (
        source["source_file"] in source["records"][prices.START + prices.MINUTE]["source_refs"][0]
    )


def test_even_identical_protocol_copy_is_not_an_implicit_source_fallback(bound_fixture):
    root, protocol, _, _ = bound_fixture
    copied = root / "copied_protocol.json"
    copied.write_bytes(protocol.read_bytes())
    with pytest.raises(ValueError, match="固定位置"):
        prices.bind_inputs(copied)


def test_binding_rejects_missing_identity_even_if_other_sources_succeed(bound_fixture, monkeypatch):
    _, protocol, _, _ = bound_fixture
    value = json.loads(protocol.read_text())
    value["input_identities"].pop(
        next(key for key in value["input_identities"] if "/dynamic_market_panel/raw/" in key)
    )
    protocol.write_text(json.dumps(value))
    monkeypatch.setattr(prices, "SOURCE_PROTOCOL_SHA", sha(protocol))
    with pytest.raises(ValueError, match="完整"):
        prices.bind_inputs(protocol)


def test_binding_rejects_raw_protocol_and_final_byte_drift(bound_fixture):
    _, protocol, _, _ = bound_fixture
    bound = prices.bind_inputs(protocol)
    path = Path(next(iter(bound.sources[SYMBOL].values())))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="漂移"):
        bound.consume(SYMBOL, "2026-10-13")
    with pytest.raises(ValueError, match="漂移"):
        bound.verify_after()
    with pytest.raises(ValueError, match="漂移"):
        prices.bind_inputs(protocol)


def test_canonical_source_rejects_symlink_and_external_path(bound_fixture, tmp_path):
    root, _, _, _ = bound_fixture
    target = root / "target.json"
    target.write_text("{}")
    link = root / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="软链接"):
        prices._canonical(link)
    external = tmp_path / "outside.json"
    external.write_text("{}")
    with pytest.raises(ValueError, match="所有权树"):
        prices._canonical(external)


def test_full_synthetic_run_preserves_all_minutes_missingness_and_receipts(bound_fixture, tmp_path):
    _, protocol, start, end = bound_fixture
    output = tmp_path / "output"
    result = prices.run(protocol, output)
    assert result["rows"] == 2 * 2 * 1440
    assert result["input_raw_files"] == 4
    table = pq.read_table(output / "minutes.parquet")
    assert table.num_rows == result["rows"]
    rows = table.to_pylist()
    for symbol in ("BTCUSDT", SYMBOL):
        selected = [row for row in rows if row["symbol"] == symbol]
        assert [row["clock_ms"] for row in selected] == list(range(start, end, prices.MINUTE))
        assert sum(row["observed"] for row in selected) == 5
        assert selected[0]["contract_1m_close"] is None
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["raw_rows_in_parseable_files"] == 12
    assert coverage["observed"] == 10
    assert coverage["normalized_completed_keys_outside_panel"] == 2
    assert coverage["observed"] + coverage["unobserved"] == result["rows"]
    assert all(
        sum(counts.values()) == result["rows"]
        for counts in coverage["field_status_counts"].values()
    )
    assert all(day["rows"] == 2880 for day in coverage["by_clock_day"].values())
    assert all(result[key] is False for key in prices.FLAGS)
    assert not (output / "failure.json").exists()
    verify_receipts(output)
    with pytest.raises(FileExistsError):
        prices.run(protocol, output)


def test_parse_failed_file_has_full_unknown_rows_and_explicit_file_status(
    bound_fixture, tmp_path, monkeypatch
):
    _, protocol, _, _ = bound_fixture
    value = json.loads(protocol.read_text())
    path = next(Path(key) for key in value["input_identities"] if "/unseen_" in key)
    path.write_text("{}")
    value["input_identities"][str(path)] = sha(path)
    protocol.write_text(json.dumps(value))
    monkeypatch.setattr(prices, "SOURCE_PROTOCOL_SHA", sha(protocol))
    result = prices.run(protocol, tmp_path / "parse_failed_output")
    assert result["rows"] == 5760
    assert result["source_file_statuses"] == {"VERIFIED_RAW_BYTES": 3, "PARSE_FAILED": 1}
    rows = pq.read_table(tmp_path / "parse_failed_output/minutes.parquet").to_pylist()
    assert sum(row["price_source_status"] == "PARSE_FAILED" for row in rows) == 1439
    verify_receipts(tmp_path / "parse_failed_output")


def test_late_source_change_seals_failure_without_success_claim(
    bound_fixture, tmp_path, monkeypatch
):
    _, protocol, _, _ = bound_fixture
    original = prices.BoundPrices.verify_after
    calls = 0

    def fail_final(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("合成末验漂移")
        return original(self)

    monkeypatch.setattr(prices.BoundPrices, "verify_after", fail_final)
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="末验漂移"):
        prices.run(protocol, output)
    assert (output / "failure.json").exists()
    assert not (output / "result.json").exists()
    verify_receipts(output)


@pytest.mark.parametrize("place", ["dynamic", "unseen", "parent", "package", "ancestor"])
def test_output_cannot_mutate_frozen_sources_or_package(tmp_path, monkeypatch, place):
    """只用人工目录验证路径保护，不能让反证测试碰真实封存来源。"""
    root = tmp_path / "project"
    monkeypatch.setattr(prices, "ROOT", root)
    monkeypatch.setattr(prices, "SOURCE_PROTOCOL", root / "results/parent/protocol.json")
    choices = {
        "dynamic": (root / prices.DYNAMIC_ROOT).parent / "new",
        "unseen": (root / prices.UNSEEN_ROOT).parent / "new",
        "parent": prices.SOURCE_PROTOCOL.parent / "new",
        "package": root / "tradingagents/new",
        "ancestor": root,
    }
    target = choices[place]
    with pytest.raises(ValueError, match="输出"):
        prices.run(prices.SOURCE_PROTOCOL, target)
    assert not target.exists()
