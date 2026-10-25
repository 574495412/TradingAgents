"""Temporal and source-quality counterexamples for historical universe ingestion."""

import io
import zipfile

import pandas as pd
import pytest

from research_tools.historical_universe.seal_daily import independent_parse


def archive(days, quote=1000, taker=5, partial=False):
    stream = io.BytesIO()
    lines = [
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    ]
    for day in days:
        t = pd.Timestamp("2025-01-" + str(day).zfill(2), tz="UTC")
        opening = t.value // 1_000_000
        closing = opening + (86_400_000 if not partial else 43_200_000) - 1
        lines.append(f"{opening},100,110,90,105,10,{closing},{quote},20,{taker},500,0")
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("XUSDT-1d-2025-01.csv", "\n".join(lines))
    return stream.getvalue()


def test_quote_volume_outside_price_bounds_is_retained_but_unusable():
    f = independent_parse(archive([1], quote=9000), "XUSDT", "2025-01")
    assert f.quote_volume.iloc[0] == 9000
    assert not f.quote_volume_valid.iloc[0] and not f.price_liquidity_feature_usable.iloc[0]


def test_missing_source_day_is_not_invented():
    f = independent_parse(archive([1, 3]), "XUSDT", "2025-01")
    assert len(f) == 2 and f.index.day.tolist() == [1, 3]
    assert f.price_liquidity_feature_usable.all()


def test_auxiliary_defect_does_not_rewrite_valid_price_or_volume():
    f = independent_parse(archive([1], taker=50), "XUSDT", "2025-01")
    assert not f.auxiliary_valid.iloc[0]
    assert f.price_liquidity_feature_usable.iloc[0] and f.taker_base_volume.iloc[0] == 50


def test_duplicate_dates_and_wrong_month_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        independent_parse(archive([1, 1]), "XUSDT", "2025-01")
    with pytest.raises(ValueError, match="ZIP"):
        independent_parse(archive([1]), "XUSDT", "2025-02")


def test_partial_delisting_day_stays_visible_and_unusable_for_full_day_features():
    f = independent_parse(archive([1], partial=True), "XUSDT", "2025-01")
    assert len(f) == 1 and not f.full_daily_duration.iloc[0]
    assert not f.price_liquidity_feature_usable.iloc[0]
