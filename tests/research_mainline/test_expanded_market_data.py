"""Source preservation, sparse calendars and unusable execution inputs."""

import csv
import io
import zipfile

import pandas as pd
import pytest

from research_tools.expanded_market.ingest import parse, profile


def payload(minutes=(0, 5), quote=10, price=10, micros=False):
    stream = io.StringIO()
    writer = csv.writer(stream)
    for minute in minutes:
        t = pd.Timestamp("2025-01-01T00:00Z") + pd.Timedelta(minutes=minute)
        scale = 1000 if micros else 1
        start = t.value // 1_000_000 * scale
        end = (t + pd.Timedelta(minutes=5)).value // 1_000_000 * scale - 1
        writer.writerow([start, price, 11, 9, 10, 1, end, quote, 3, 0.6, 6, 0])
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as z:
        z.writestr("TESTUSDT-5m-2025-01.csv", stream.getvalue())
    return body.getvalue()


def test_gap_is_retained_and_never_creates_a_zero_price_bar():
    f, _ = parse(payload((0, 10)), "TESTUSDT", "2025-01")
    p = profile("TESTUSDT", f, {"bars_start": "2025-01-01", "end_exclusive": "2025-01-02"})
    assert len(f) == 2 and p["interior_missing_bars"] == 1 and not p["missing_calendars_filled"]
    assert p["rows"] + p["interior_missing_bars"] + p["outside_source_range_bars"] == 288
    assert f.close.eq(10).all()


def test_liquidity_conflict_is_preserved_but_never_marked_executable():
    f, q = parse(payload(quote=1e6), "TESTUSDT", "2025-01")
    assert f.quote_volume.eq(1e6).all() and f.price_usable.all()
    assert not f.execution_liquidity_usable.any() and q["unusable_execution_bars"] == 2


def test_bad_price_is_preserved_with_an_explicit_unusable_flag():
    f, _ = parse(payload(price=0), "TESTUSDT", "2025-01")
    assert f.open.eq(0).all() and not f.price_usable.any()
    assert not f.execution_liquidity_usable.any()


def test_duplicate_source_timestamp_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        parse(payload((0, 0)), "TESTUSDT", "2025-01")


def test_microsecond_source_close_precision_is_not_a_partial_bar():
    f, _ = parse(payload(micros=True), "TESTUSDT", "2025-01")
    assert f.full_source_candle_duration.all() and f.execution_liquidity_usable.all()
    assert f.index[0] == pd.Timestamp("2025-01-01T00:00Z")
