"""现货原微秒与月度完整性测试，不使用网络或合成利润。"""

import csv
import io
import zipfile
from datetime import UTC, datetime

import pytest

from tradingagents.dataflows.research.spot_archives import HEADERS, MINUTE_US, key, parse_archive


def rows():
    start = int(datetime(2026, 8, 1, tzinfo=UTC).timestamp()) * 1_000_000
    return [
        [
            str(start + i * MINUTE_US),
            "100",
            "102",
            "99",
            "101",
            "2",
            str(start + (i + 1) * MINUTE_US - 1),
            "202",
            "3",
            "1",
            "101",
            "0",
        ]
        for i in range(2)
    ]


def archive(data, name="BTCUSDT-1m-2026-08.csv"):
    content = io.StringIO()
    csv.writer(content).writerows(data)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr(name, content.getvalue())
    return stream.getvalue()


def test_microseconds_are_exactly_normalized_and_calendar_gaps_retained():
    profile = parse_archive(archive(rows()), "BTCUSDT", "2026-08")
    assert profile["rows"] == 2
    assert profile["expected_minutes"] == 31 * 1440
    assert profile["unknown_minutes"] == 31 * 1440 - 2
    assert profile["normalized_first_completed_ms"] == int(rows()[0][0]) // 1000 + 60000
    assert profile["declared_time_unit"] == "microseconds"


@pytest.mark.parametrize("header", list(HEADERS))
def test_only_exact_declared_headers_are_skipped(header):
    profile = parse_archive(archive([header, *rows()]), "BTCUSDT", "2026-08")
    assert profile["header_present"]


def test_zero_quote_is_observed_zero_and_original_order_need_not_be_sorted():
    data = rows()
    for p in (5, 7, 9, 10):
        data[0][p] = "0"
    profile = parse_archive(archive(data[::-1]), "BTCUSDT", "2026-08")
    assert profile["zero_quote_rows"] == 1
    assert profile["rows"] == 2


@pytest.mark.parametrize(
    "damage",
    [
        "milliseconds",
        "close_boundary",
        "duplicate",
        "wrong_month",
        "nonfinite",
        "negative_volume",
        "taker",
        "ohlc",
        "trades",
        "width",
    ],
)
def test_corrupt_or_wrong_unit_rows_rejected(damage):
    data = rows()
    if damage == "milliseconds":
        data[0][0] = str(int(data[0][0]) // 1000)
        data[0][6] = str(int(data[0][6]) // 1000)
    elif damage == "close_boundary":
        data[0][6] = str(int(data[0][6]) - 1)
    elif damage == "duplicate":
        data[1] = data[0][:]
    elif damage == "wrong_month":
        data[0][0] = str(int(data[0][0]) - MINUTE_US)
        data[0][6] = str(int(data[0][6]) - MINUTE_US)
    elif damage == "nonfinite":
        data[0][4] = "NaN"
    elif damage == "negative_volume":
        data[0][5] = "-1"
    elif damage == "taker":
        data[0][10] = "203"
    elif damage == "ohlc":
        data[0][2] = "99"
    elif damage == "trades":
        data[0][8] = "3.0"
    else:
        data[0] = data[0][:-1]
    with pytest.raises(ValueError):
        parse_archive(archive(data), "BTCUSDT", "2026-08")


@pytest.mark.parametrize(
    "name", ["../BTCUSDT-1m-2026-08.csv", "ETHUSDT-1m-2026-08.csv", "BTCUSDT-1m-2026-09.csv"]
)
def test_member_identity_not_inferred_from_values(name):
    with pytest.raises(ValueError, match="成员"):
        parse_archive(archive(rows(), name), "BTCUSDT", "2026-08")


def test_task_population_and_empty_file_are_not_silently_changed():
    with pytest.raises(ValueError, match="人口"):
        key("SOLUSDT", "2026-08")
    with pytest.raises(ValueError, match="人口"):
        key("BTCUSDT", "2026-10")
    with pytest.raises(ValueError, match="没有"):
        parse_archive(archive([]), "BTCUSDT", "2026-08")
