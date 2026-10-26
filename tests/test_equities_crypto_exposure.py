"""Clock and raw-data rejection checks; not evidence of strategy profitability."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase
from zoneinfo import ZoneInfo

SOURCE = Path(__file__).resolve().parents[1] / "tradingagents/research/equities/crypto_exposure.py"
SPEC = importlib.util.spec_from_file_location("crypto_exposure_test_subject", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def entry(start):
    return {"candle": [int(start.timestamp()), 90, 110, 95, 100, 2], "lineage": []}


class CryptoExposureClockChecks(TestCase):
    def test_dst_weekend_keeps_both_original_clocks(self):
        ny = ZoneInfo("America/New_York")
        friday = datetime(2023, 3, 10, 16, tzinfo=ny)
        monday = datetime(2023, 3, 13, 16, tzinfo=ny)
        self.assertEqual(friday.astimezone(UTC).hour, 21)
        self.assertEqual(monday.astimezone(UTC).hour, 20)
        self.assertEqual(
            (monday.astimezone(UTC) - friday.astimezone(UTC)).total_seconds(), 71 * 3600
        )
        for close in [friday, monday]:
            start = close.astimezone(UTC) - timedelta(hours=1)
            selected = MODULE.closing_bucket({int(start.timestamp()): entry(start)}, close)
            self.assertEqual(selected["candle"][0], int(start.timestamp()))

    def test_half_day_uses_1300_new_york_not_1600(self):
        close = datetime(2023, 7, 3, 13, tzinfo=ZoneInfo("America/New_York"))
        needed = datetime(2023, 7, 3, 16, tzinfo=UTC)
        wrong = datetime(2023, 7, 3, 19, tzinfo=UTC)
        selected = MODULE.closing_bucket(
            {int(needed.timestamp()): entry(needed), int(wrong.timestamp()): entry(wrong)}, close
        )
        self.assertEqual(selected["candle"][0], int(needed.timestamp()))

    def test_no_stale_or_future_fill_when_required_hour_absent(self):
        close = datetime(2024, 6, 26, 20, tzinfo=UTC)
        stale, future = close - timedelta(hours=2), close
        with self.assertRaises(ValueError):
            MODULE.closing_bucket(
                {int(stale.timestamp()): entry(stale), int(future.timestamp()): entry(future)},
                close,
            )

    def test_key_cannot_hide_a_future_original_candle(self):
        close = datetime(2024, 6, 26, 20, tzinfo=UTC)
        with self.assertRaises(ValueError):
            MODULE.closing_bucket(
                {int((close - timedelta(hours=1)).timestamp()): entry(close)}, close
            )

    def test_naive_and_partial_hour_clocks_rejected(self):
        for close in [datetime(2024, 6, 26, 20), datetime(2024, 6, 26, 20, 30, tzinfo=UTC)]:
            with self.assertRaises(ValueError):
                MODULE.closing_bucket({}, close)

    def test_raw_candle_does_not_accept_invalid_numbers_or_ohlcv(self):
        row = entry(datetime(2024, 6, 26, 19, tzinfo=UTC))["candle"]
        for index, value in [
            (0, True),
            (0, row[0] * 1000),
            (1, float("nan")),
            (2, float("inf")),
            (3, 111),
            (4, 89),
            (5, -1),
        ]:
            altered = row.copy()
            altered[index] = value
            # A raw millisecond epoch is numeric and hour-divisible, but cannot
            # match an expected seconds clock. Other malformed values fail here.
            if index == 0 and value == row[0] * 1000:
                with self.assertRaises(ValueError):
                    MODULE.closing_bucket(
                        {row[0]: {"candle": altered}}, datetime(2024, 6, 26, 20, tzinfo=UTC)
                    )
            else:
                with self.assertRaises(ValueError):
                    MODULE.valid_candle(altered)

    def test_coinbase_six_fields_required(self):
        with self.assertRaises(ValueError):
            MODULE.valid_candle([1, 2, 3, 4, 5])
