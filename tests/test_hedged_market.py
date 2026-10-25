"""两真实市场的单位及结算原毫秒边界；不构造策略利润。"""

import copy
import csv
import io
import zipfile

import pytest

from tradingagents.dataflows.research.hedged_market import START, funding_rows, parse_market_zip


def zipped(market, *, minute=0, duplicate=False, month="2026-08"):
    scale = 1000 if market == "spot" else 1
    opening = (START + minute * 60000) * scale
    row = [
        str(opening),
        "100.0000",
        "101",
        "99",
        "100.5",
        "2",
        str(opening + 60000 * scale - 1),
        "201",
        "3",
        "1",
        "100.5",
        "0",
    ]
    stream, text = io.BytesIO(), io.StringIO()
    csv.writer(text).writerows([row, row] if duplicate else [row])
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(f"BTCUSDT-1m-{month}.csv", text.getvalue())
    return stream.getvalue()


def test_both_market_units_join_at_identical_open_without_losing_original_strings():
    spot = parse_market_zip(zipped("spot"), "BTCUSDT", "2026-08", "spot")
    perp = parse_market_zip(zipped("perp"), "BTCUSDT", "2026-08", "perp")
    assert list(spot) == list(perp) == [START]
    assert spot[START]["raw_open"] == perp[START]["raw_open"] * 1000
    assert (spot[START]["raw_close"] + 1) // 1000 == perp[START]["raw_close"] + 1
    assert spot[START]["open"] == perp[START]["open"] == "100.0000"


@pytest.mark.parametrize("actual,declared", [("spot", "perp"), ("perp", "spot")])
def test_wrong_declared_unit_never_guessed(actual, declared):
    with pytest.raises(ValueError):
        parse_market_zip(zipped(actual), "BTCUSDT", "2026-08", declared)


def test_month_last_minute_is_kept_with_next_month_completed_boundary():
    values = parse_market_zip(zipped("spot", minute=31 * 1440 - 1), "BTCUSDT", "2026-08", "spot")
    opening = max(values)
    assert opening == START + 31 * 86400000 - 60000
    assert (values[opening]["raw_close"] + 1) // 1000 == START + 31 * 86400000


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_duplicate_bar_is_not_silently_overwritten(market):
    with pytest.raises(ValueError, match="重复"):
        parse_market_zip(zipped(market, duplicate=True), "BTCUSDT", "2026-08", market)


def event(symbol="BTCUSDT", kind="Regular", offset=1):
    raw = {
        "symbol": symbol,
        "fundingTime": START + offset,
        "fundingRate": "-0.00001234",
        "markPrice": "100.0000",
        "rateType": kind,
    }
    return {
        "symbol": symbol,
        "time_ms": START + offset,
        "rate": raw["fundingRate"],
        "mark": raw["markPrice"],
        "type": kind,
        "raw": raw,
        "source": {"event_index": 0},
        "quality": {
            "trusted_typed_fields": True,
            "identity_conflict": False,
            "month_ready_under_source_assumptions": True,
        },
    }


def replay(*events):
    return {
        symbol: {"events": [r["raw"] for r in events if r["symbol"] == symbol]}
        for symbol in ("BTCUSDT", "ETHUSDT")
    }


def test_original_settlement_millisecond_signed_rate_and_special_are_preserved():
    rows = [event(), event("ETHUSDT", "Special", 2)]
    actual = funding_rows(rows, replay(*rows))
    assert actual[0]["time_ms"] == START + 1
    assert actual[0]["rate"] == "-0.00001234"
    assert actual[1]["type"] == "Special"


@pytest.mark.parametrize(
    "damage",
    [
        "rounded_time",
        "mark_changed",
        "rate_changed",
        "unknown_type",
        "untrusted",
        "missing_event",
        "duplicate",
    ],
)
def test_cash_source_contradiction_and_missing_history_rejected(damage):
    rows = [event(), event("ETHUSDT")]
    original = replay(*copy.deepcopy(rows))
    if damage == "rounded_time":
        rows[0]["time_ms"] = START
    elif damage == "mark_changed":
        rows[0]["mark"] = "999"
    elif damage == "rate_changed":
        rows[0]["rate"] = "0.5"
    elif damage == "unknown_type":
        rows[0]["type"] = "Other"
    elif damage == "untrusted":
        rows[0]["quality"]["trusted_typed_fields"] = False
    elif damage == "missing_event":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        funding_rows(rows, original)
