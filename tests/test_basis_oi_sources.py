"""纯合成验证index／metrics原文解析与因果时钟连接，不读取真实市场。"""

import copy
import csv
import hashlib
import io
import struct
import zipfile
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tradingagents.dataflows.research.basis_oi_sources import (
    INDEX_FIELDS,
    METRICS_DELAY_MS,
    METRICS_FIELDS,
    MINUTE,
    ArchiveFormatError,
    parse_archive,
    source_values_for_clock,
)

DAY = "2026-09-28"
PREVIOUS = "2026-09-27"
NEXT = "2026-09-29"
SYMBOL = "牛来USDT"
START = int(datetime(2026, 9, 28, tzinfo=UTC).timestamp()) * 1000
HEADER = ["create_time", "symbol", *METRICS_FIELDS]
INDEX_HEADER = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "Ignore",
    "close_time",
    "Ignore",
    "Ignore",
    "Ignore",
    "Ignore",
    "Ignore",
]


def _zip(rows, kind="metrics", day=DAY, symbol=SYMBOL, header=True):
    """内存生成一个固定身份的UTF-8档案。"""
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    if header:
        writer.writerow(HEADER if kind == "metrics" else INDEX_HEADER)
    writer.writerows(rows)
    content = text.getvalue().encode("utf-8")
    name = f"{symbol}-{'metrics' if kind == 'metrics' else '1m'}-{day}.csv"
    return _zip_content(content, name), content


def _zip_content(content, name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def _metric(stamp=START, values=None, symbol=SYMBOL):
    text = datetime.fromtimestamp(stamp / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")
    return [text, symbol, *(values if values is not None else ["1", "2", "3", "4", "5", "6"])]


def _index(stamp=START, values=None):
    return [
        str(stamp),
        *(values if values is not None else ["10", "12", "9", "11"]),
        "987654321",
        str(stamp + MINUTE - 1),
        "99999999",
        "60",
        "123456789",
        "123456789",
        "17",
    ]


def _parse(rows, kind="metrics", day=DAY, **kwargs):
    payload, _ = _zip(rows, kind, day, **kwargs)
    return parse_archive(payload, kind, kwargs.get("symbol", SYMBOL), day)


def _status(*days, status="VERIFIED_BYTES_AND_PROFILE"):
    return {day: {"status": status, "profile": "G"} for day in days}


def _aligned(clocks, index=None, metrics=None, index_status=None, metrics_status=None):
    return source_values_for_clock(
        clocks,
        index or {},
        metrics or {},
        _status(DAY) if index_status is None else index_status,
        _status(DAY) if metrics_status is None else metrics_status,
    )


@pytest.mark.parametrize("header", [False, True])
def test_index_completion_only_price_fields_and_exact_raw_bytes(header):
    payload, content = _zip([_index()], "indexPriceKlines", header=header)
    result = parse_archive(payload, "indexPriceKlines", SYMBOL, DAY)
    assert result["member"] == f"{SYMBOL}-1m-{DAY}.csv"
    assert result["member_sha256"] == hashlib.sha256(content).hexdigest()
    assert list(result["normalized"]) == [START + MINUTE]
    row = result["normalized"][START + MINUTE]
    assert row["values"] == dict(zip(INDEX_FIELDS, ["1E+1", "12", "9", "11"], strict=True))
    assert row["index_bar_open_ms"] == START
    assert row["index_completed_ms"] == START + MINUTE
    assert result["raw_records"][0]["fields"][8] == "60"
    assert result["raw_records"][0]["raw_text"].endswith("\r\n")
    assert result["raw_records"][0]["source_line_no"] == (2 if header else 1)
    assert result["profile"]["index_placeholder_fields_excluded"] is True
    assert not any("volume" in name or "taker" in name or "count" in name for name in row["values"])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.infolist()[0].flag_bits & 2048


def test_precision_zero_and_missing_invalid_are_per_field():
    high_precision = "123456789012345678901234567890123456789.12000"
    result = _parse([_metric(values=[high_precision, "-0.000", "", "NaN", "bad", "-2.50"])])
    row = result["normalized"][START]
    assert row["values"][METRICS_FIELDS[0]] == high_precision[:-3]
    assert Decimal(row["values"][METRICS_FIELDS[0]]) == Decimal(high_precision)
    assert row["values"][METRICS_FIELDS[1]] == "0"
    assert row["value_status"][METRICS_FIELDS[1]] == "KNOWN_RAW_VALUE"
    assert [row["value_status"][name] for name in METRICS_FIELDS[2:5]] == [
        "EMPTY",
        "INVALID",
        "INVALID",
    ]
    assert row["values"][METRICS_FIELDS[5]] is None
    assert row["value_reasons"][METRICS_FIELDS[5]] == ["NEGATIVE_METRICS_VALUE"]
    assert result["raw_records"][0]["fields"][2] == high_precision


@pytest.mark.parametrize("bad", ["0", "-1", "Infinity", "NaN", "text", ""])
def test_index_invalid_price_does_not_destroy_other_price_fields(bad):
    result = _parse([_index(values=[bad, "12", "9", "11"])], "indexPriceKlines")
    row = result["normalized"][START + MINUTE]
    assert row["values"][INDEX_FIELDS[0]] is None
    assert row["value_status"][INDEX_FIELDS[0]] in {"EMPTY", "INVALID"}
    assert row["value_status"][INDEX_FIELDS[1]] == "KNOWN_RAW_VALUE"
    assert row["values"][INDEX_FIELDS[3]] == "11"


def test_duplicates_are_exact_decimal_consensus_and_retain_every_source_line():
    values = ["1.00", "2", "3", "4", "5", "6"]
    result = _parse([_metric(values=values), _metric()])
    row = result["normalized"][START]
    assert row["duplicate_count"] == 2
    assert row["source_line_nos"] == [2, 3]
    assert row["raw_timestamp_texts"] == ["2026-09-28 00:00:00"] * 2
    assert row["values"][METRICS_FIELDS[0]] == "1"
    assert all(status == "KNOWN_RAW_VALUE" for status in row["value_status"].values())
    assert result["profile"]["duplicate_extra_rows"] == 1


@pytest.mark.parametrize("conflict", ["7", "", "NaN", "unreadable"])
def test_duplicate_conflict_is_unknown_only_in_the_affected_field(conflict):
    values = ["1", conflict, "3", "4", "5", "6"]
    result = _parse([_metric(), _metric(values=values)])
    row = result["normalized"][START]
    assert row["values"][METRICS_FIELDS[1]] is None
    assert row["value_status"][METRICS_FIELDS[1]] == "DUPLICATE_CONFLICT"
    assert row["source_line_nos"] == [2, 3]
    assert row["values"][METRICS_FIELDS[0]] == "1"
    assert row["values"][METRICS_FIELDS[-1]] == "6"
    assert "DUPLICATE_FIELD_NOT_UNANIMOUS" in row["value_reasons"][METRICS_FIELDS[1]]


def test_reverse_order_is_sorted_without_changing_semantic_values():
    rows = [_metric(START + offset * 5 * MINUTE) for offset in range(4)]
    forward = _parse(rows)
    reverse = _parse(list(reversed(rows)))
    assert list(forward["normalized"]) == list(reverse["normalized"])
    assert [row["values"] for row in forward["normalized"].values()] == [
        row["values"] for row in reverse["normalized"].values()
    ]
    assert reverse["profile"]["inverse_time_steps_in_input"] == 3
    assert reverse["normalized"][START]["source_line_nos"] == [5]
    assert [row["parsed_timestamp_ms"] for row in reverse["raw_records"]] == list(
        reversed(list(forward["normalized"]))
    )


def test_short_trailing_metric_field_is_empty_and_other_values_survive():
    result = _parse([_metric()[:-1]])
    row = result["normalized"][START]
    assert row["value_status"][METRICS_FIELDS[-1]] == "EMPTY"
    assert row["values"][METRICS_FIELDS[0]] == "1"
    assert result["raw_records"][0]["identity_status"] == "ON_GRID"
    assert result["anomalies"][0]["reasons"] == ["FIELD_COUNT_MISMATCH"]


@pytest.mark.parametrize(
    ("time_text", "symbol", "reason"),
    [
        ("2026-09-28 00:00:00", "OTHERUSDT", "WRONG_SYMBOL"),
        ("2026-09-28 00:01:00", SYMBOL, "OFFGRID_TIMESTAMP"),
        ("2026-09-29 00:00:00", SYMBOL, "OUTSIDE_FILE_DAY"),
        ("2026-09-27 23:55:00", SYMBOL, "OUTSIDE_FILE_DAY"),
        ("2026-09-28 00:00:00.0001", SYMBOL, "NON_INTEGER_MILLISECOND"),
        ("2026-09-28 00:00:00.001", SYMBOL, "OFFGRID_TIMESTAMP"),
        ("2026-09-28 00:00:00-00:00:00.0001", SYMBOL, "NON_INTEGER_MILLISECOND"),
        ("2026-09-28 00:00:00.000-00:00:00.0000001", SYMBOL, "NON_INTEGER_MILLISECOND"),
        ("not a time", SYMBOL, "INVALID_TIMESTAMP"),
    ],
)
def test_metrics_bad_identity_remains_raw_and_anomalous(time_text, symbol, reason):
    row = _metric(symbol=symbol)
    row[0] = time_text
    result = _parse([row])
    assert result["normalized"] == {}
    assert len(result["raw_records"]) == 1
    assert result["raw_records"][0]["fields"] == row
    assert result["raw_records"][0]["raw_timestamp_text"] == time_text
    assert result["raw_records"][0]["source_line_no"] == 2
    assert result["raw_records"][0]["identity_status"] == "INVALID_IDENTITY"
    assert reason in result["anomalies"][0]["reasons"]


def test_timezone_is_explicit_or_a_separate_utc_assumption():
    rows = [_metric(), _metric(START + 5 * MINUTE)]
    rows[1][0] = "2026-09-28T08:05:00+08:00"
    result = _parse(rows)
    assert result["normalized"][START]["timezone_assumption"] == "UTC"
    assert result["normalized"][START + 5 * MINUTE]["timezone_assumption"] is None
    rows[0][0] = "2026-09-28 00:00:00.0000000"
    assert list(_parse(rows)["normalized"]) == [START, START + 5 * MINUTE]


@pytest.mark.parametrize("change", ["bad_close", "offgrid", "outside", "huge_timestamp"])
def test_index_identity_errors_remain_raw(change):
    row = _index()
    if change == "bad_close":
        row[6] = str(START + MINUTE)
    elif change == "offgrid":
        row = _index(START + 1)
    elif change == "outside":
        row = _index(START + 86_400_000)
    else:
        row[0] = "9" * 5000
    result = _parse([row], "indexPriceKlines")
    assert result["normalized"] == {}
    assert result["profile"]["invalid_identity_rows"] == 1
    assert result["raw_records"][0]["fields"] == row


def test_original_multiline_text_and_physical_line_numbers_are_preserved():
    rows = [_metric(), _metric(START + 5 * MINUTE)]
    rows[0][-1] = "not\na decimal"
    result = _parse(rows)
    assert result["raw_records"][0]["source_line_no"] == 2
    assert result["raw_records"][0]["source_line_end_no"] == 3
    assert result["raw_records"][1]["source_line_no"] == 4
    assert '"not\na decimal"' in result["raw_records"][0]["raw_text"]
    assert result["normalized"][START]["value_status"][METRICS_FIELDS[-1]] == "INVALID"


def test_unflagged_utf8_member_accepts_only_exact_expected_original_bytes():
    payload, content = _zip([_metric()])
    unflagged = bytearray(payload)
    # 原样保留UTF-8名字字节，只清除local／central的UTF-8声明位。
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        position = unflagged.index(signature) + offset
        flags = struct.unpack_from("<H", unflagged, position)[0]
        struct.pack_into("<H", unflagged, position, flags & ~2048)
    result = parse_archive(bytes(unflagged), "metrics", SYMBOL, DAY)
    member = f"{SYMBOL}-metrics-{DAY}.csv"
    profile = result["profile"]
    assert result["member"] == member
    assert result["member_sha256"] == hashlib.sha256(content).hexdigest()
    assert profile["member_original_filename"] == member.encode().decode("cp437")
    assert profile["member_original_name_bytes_hex"] == member.encode().hex()
    assert profile["member_name_interpretation"] == "UTF8_BYTES_WITHOUT_FLAG_EXACT_EXPECTED"
    assert profile["member_utf8_flag"] is False
    assert result["normalized"][START]["values"][METRICS_FIELDS[0]] == "1"
    with pytest.raises(ArchiveFormatError):
        parse_archive(bytes(unflagged), "metrics", "OTHERUSDT", DAY)


@pytest.mark.parametrize("offset", range(6))
def test_negative_metrics_are_invalid_only_for_the_corresponding_raw_field(offset):
    values = ["1"] * 6
    values[offset] = "-0.00001"
    result = _parse([_metric(values=values)])
    row = result["normalized"][START]
    assert row["values"][METRICS_FIELDS[offset]] is None
    assert row["value_reasons"][METRICS_FIELDS[offset]] == ["NEGATIVE_METRICS_VALUE"]
    assert all(
        row["values"][name] == "1" for index, name in enumerate(METRICS_FIELDS) if index != offset
    )
    assert result["raw_records"][0]["fields"][offset + 2] == "-0.00001"


@pytest.mark.parametrize("value", ["1_000", "１２", "1,000", "1 0"])
def test_non_decimal_numeric_spellings_are_not_silently_reinterpreted(value):
    result = _parse([_metric(values=[value, "2", "3", "4", "5", "6"])])
    row = result["normalized"][START]
    assert row["value_status"][METRICS_FIELDS[0]] == "INVALID"
    assert row["values"][METRICS_FIELDS[0]] is None
    assert result["raw_records"][0]["fields"][2] == value


@pytest.mark.parametrize(
    "values",
    [
        ["13", "12", "9", "11"],
        ["10", "8", "9", "11"],
        ["10", "12", "11", "11"],
        ["10", "12", "9", "13"],
    ],
)
def test_ohlc_relationship_conflicts_preserve_key_but_make_all_prices_unknown(values):
    result = _parse([_index(values=values)], "indexPriceKlines")
    row = result["normalized"][START + MINUTE]
    assert set(row["values"].values()) == {None}
    assert set(row["value_status"].values()) == {"INVALID"}
    assert result["raw_records"][0]["fields"][1:5] == values
    assert result["raw_records"][0]["identity_status"] == "ON_GRID"
    assert result["anomalies"][0]["reasons"] == ["INVALID_OHLC_RELATIONSHIP"]


@pytest.mark.parametrize(
    "content",
    [b"", b"\r\n", b"\xff", b'create_time,symbol,"unterminated\n', b"create_time,symbol\n"],
)
def test_csv_structure_failures_have_explicit_archive_error_and_member_identity(content):
    member = f"{SYMBOL}-metrics-{DAY}.csv"
    with pytest.raises(ArchiveFormatError) as caught:
        parse_archive(_zip_content(content, member), "metrics", SYMBOL, DAY)
    assert caught.value.member == member
    assert caught.value.member_sha256 == hashlib.sha256(content).hexdigest()


def test_header_duplicates_unknown_index_header_and_wrong_member_are_rejected():
    text = ",".join([*HEADER, METRICS_FIELDS[0]]) + "\n"
    with pytest.raises(ArchiveFormatError):
        parse_archive(
            _zip_content(text.encode(), f"{SYMBOL}-metrics-{DAY}.csv"), "metrics", SYMBOL, DAY
        )
    bad_index = (
        "timestamp,open,high,low,close,ignore,close_time,ignore,ignore,ignore,ignore,ignore\n"
    )
    with pytest.raises(ArchiveFormatError):
        parse_archive(
            _zip_content(bad_index.encode(), f"{SYMBOL}-1m-{DAY}.csv"),
            "indexPriceKlines",
            SYMBOL,
            DAY,
        )
    payload, _ = _zip([_metric()], symbol="OTHERUSDT")
    with pytest.raises(ArchiveFormatError):
        parse_archive(payload, "metrics", SYMBOL, DAY)


def test_bad_zip_and_multiple_members_are_format_failures():
    with pytest.raises(ArchiveFormatError):
        parse_archive(b"not zip", "metrics", SYMBOL, DAY)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("one.csv", "one")
        archive.writestr("two.csv", "two")
    with pytest.raises(ArchiveFormatError):
        parse_archive(buffer.getvalue(), "metrics", SYMBOL, DAY)


@pytest.mark.parametrize(
    "member", ["../escape.csv", "/absolute.csv", "nested/../escape.csv", "bad\\path.csv"]
)
def test_path_escape_is_integrity_failure_not_missing_data(member):
    with pytest.raises(ValueError) as caught:
        parse_archive(_zip_content(b"irrelevant", member), "metrics", SYMBOL, DAY)
    assert not isinstance(caught.value, ArchiveFormatError)


@pytest.mark.parametrize(
    ("kind", "symbol", "day"),
    [
        ("premiumIndex", SYMBOL, DAY),
        ("metrics", "../x", DAY),
        ("metrics", "", DAY),
        ("metrics", SYMBOL, "2026-9-28"),
    ],
)
def test_bad_input_identity_fails_before_guessing(kind, symbol, day):
    with pytest.raises(ValueError):
        parse_archive(b"", kind, symbol, day)


def test_exact_clock_delay_and_first_two_missing_metrics_without_forward_fill():
    index = _parse([_index(START + 4 * MINUTE), _index(START + 9 * MINUTE)], "indexPriceKlines")[
        "normalized"
    ]
    metrics = _parse([_metric(), _metric(START + 10 * MINUTE)])["normalized"]
    clocks = [START + offset * 5 * MINUTE for offset in range(5)]
    rows = _aligned(clocks, index, metrics)
    assert len(rows) == 5
    assert [row["sum_open_interest"] for row in rows] == [None, None, "1", None, "1"]
    assert rows[0]["index_source_day"] == PREVIOUS
    assert rows[0]["index_source_status"] == "OUTSIDE_SOURCE_RANGE"
    assert rows[0]["index_completed_ms"] is None
    assert rows[1]["index_bar_open_ms"] == START + 4 * MINUTE
    assert rows[2]["index_completed_ms"] == START + 10 * MINUTE
    assert rows[3]["index_1m_close"] is None
    assert rows[3]["sum_open_interest_value_status"] == "MISSING_CLOCK"
    assert rows[2]["raw_T_ms"] == START
    assert rows[2]["research_visible_at_ms"] == START + METRICS_DELAY_MS
    assert rows[2]["requested_metrics_raw_T_ms"] == START
    assert rows[2]["metrics_timezone_assumption"] == "UTC"
    assert rows[2]["metrics_economic_time_status"] == "UNKNOWN"
    assert rows[2]["first_publication_time_status"] == "UNKNOWN"
    assert not any("volume" in key or "taker_quote" in key for key in rows[2])


def test_cross_day_exact_lookup_uses_previous_file_and_preserves_end_records():
    index = _parse([_index(START - MINUTE)], "indexPriceKlines", PREVIOUS)["normalized"]
    metrics = _parse([_metric(START - 10 * MINUTE), _metric(START - 5 * MINUTE)], day=PREVIOUS)[
        "normalized"
    ]
    rows = _aligned(
        [START, START + 5 * MINUTE], index, metrics, _status(PREVIOUS, DAY), _status(PREVIOUS, DAY)
    )
    assert rows[0]["index_bar_open_ms"] == START - MINUTE
    assert rows[0]["index_source_day"] == PREVIOUS
    assert [row["raw_T_ms"] for row in rows] == [START - 10 * MINUTE, START - 5 * MINUTE]
    assert all(row["metrics_source_day"] == PREVIOUS for row in rows)
    final = _parse(
        [_metric(START + 86_400_000 - 10 * MINUTE), _metric(START + 86_400_000 - 5 * MINUTE)]
    )
    assert len(final["normalized"]) == 2
    assert [row["research_visible_at_ms"] for row in final["normalized"].values()] == [
        START + 86_400_000,
        START + 86_400_000 + 5 * MINUTE,
    ]
    earlier = _aligned([START + 86_400_000 - 5 * MINUTE], metrics=final["normalized"])[0]
    assert earlier["sum_open_interest"] is None


@pytest.mark.parametrize("status", ["PUBLIC_404", "FAILED_OR_UNKNOWN", "PARSE_FAILED"])
def test_missing_source_status_and_reason_remain_distinct_from_numeric_zero(status):
    source = {DAY: {"status": status, "error": "原始失败证据"}}
    row = _aligned([START + 10 * MINUTE], index_status=source, metrics_status=source)[0]
    assert row["index_source_status"] == status
    assert row["metrics_source_status"] == status
    assert row["metrics_source_reason"] == "原始失败证据"
    assert all(row[name] is None for name in (*INDEX_FIELDS, *METRICS_FIELDS))
    assert all(
        row[name + "_value_status"] == "MISSING_CLOCK" for name in (*INDEX_FIELDS, *METRICS_FIELDS)
    )
    assert row["metrics_source_line_nos"] == []
    assert row["metrics_duplicate_count"] == 0


def test_zero_observation_survives_alignment_and_keeps_all_duplicate_refs():
    metrics = _parse([_metric(values=["0"] * 6), _metric(values=["0.000"] * 6)])["normalized"]
    row = _aligned([START + 10 * MINUTE], metrics=metrics)[0]
    assert row["sum_open_interest"] == "0"
    assert row["sum_open_interest_value_status"] == "KNOWN_RAW_VALUE"
    assert row["sum_open_interest_source_line_nos"] == [2, 3]
    assert row["metrics_duplicate_count"] == 2


def test_future_raw_mutations_do_not_change_older_snapshot_and_hourly_is_projection():
    old_index = [_index(START + 9 * MINUTE), _index(START + 59 * MINUTE)]
    old_metrics = [_metric(), _metric(START + 50 * MINUTE)]
    later_index = copy.deepcopy(old_index)
    later_metrics = copy.deepcopy(old_metrics)
    later_index[-1][1:5] = ["999"] * 4
    later_metrics[-1][2:] = ["999"] * 6
    clocks = [START + offset * 5 * MINUTE for offset in range(13)]
    before = _aligned(
        clocks,
        _parse(old_index, "indexPriceKlines")["normalized"],
        _parse(old_metrics)["normalized"],
    )
    after = _aligned(
        clocks,
        _parse(later_index, "indexPriceKlines")["normalized"],
        _parse(later_metrics)["normalized"],
    )
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]
    hourly = _aligned(
        [clocks[0], clocks[-1]],
        _parse(old_index, "indexPriceKlines")["normalized"],
        _parse(old_metrics)["normalized"],
    )
    assert hourly == [row for row in before if row["clock_ms"] % (60 * MINUTE) == 0]


@pytest.mark.parametrize("fault", ["status", "day", "kind", "key", "symbol"])
def test_join_rejects_source_contract_conflicts(fault):
    index = _parse([_index(START + 9 * MINUTE)], "indexPriceKlines")["normalized"]
    metrics = _parse([_metric()])["normalized"]
    statuses = _status(DAY)
    if fault == "status":
        statuses[DAY]["status"] = "PUBLIC_404"
    elif fault == "day":
        index[START + 10 * MINUTE]["day"] = NEXT
    elif fault == "kind":
        index[START + 10 * MINUTE]["kind"] = "metrics"
    elif fault == "key":
        index[START + 10 * MINUTE]["key_ms"] += MINUTE
    else:
        metrics[START]["symbol"] = "OTHERUSDT"
    with pytest.raises(ValueError):
        _aligned([START + 10 * MINUTE], index, metrics, statuses)


@pytest.mark.parametrize(
    "clocks",
    [
        [START + 1],
        [START, START],
        [START + 5 * MINUTE, START],
        [-5 * MINUTE],
        [float(START)],
        [True],
    ],
)
def test_join_rejects_invalid_calendar_without_rounding(clocks):
    with pytest.raises(ValueError):
        _aligned(clocks)


def test_empty_clock_calendar_is_empty_and_original_inputs_are_not_mutated():
    result = _parse([_metric()])
    frozen = copy.deepcopy(result)
    assert _aligned([], metrics=result["normalized"]) == []
    _aligned([START + 10 * MINUTE], metrics=result["normalized"])
    assert result == frozen
