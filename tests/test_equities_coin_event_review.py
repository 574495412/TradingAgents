"""Reject unsafe event clocks; these checks do not certify investment results."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

SOURCE = (
    Path(__file__).resolve().parents[1] / "tradingagents/research/equities/coin_event_review.py"
)
SPEC = importlib.util.spec_from_file_location("coin_event_review_test_subject", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def at(value):
    return datetime.fromisoformat(value)


CALENDAR = [
    ("2023-03-10", at("2023-03-10T14:30:00+00:00"), at("2023-03-10T21:00:00+00:00")),
    ("2023-03-13", at("2023-03-13T13:30:00+00:00"), at("2023-03-13T20:00:00+00:00")),
]


class CoinEventClocks(TestCase):
    def test_publisher_clock_retains_minute_precision_and_dst(self):
        lo, hi = MODULE.wire_minute_interval("May 4, 2023 4:05 PM Eastern Daylight Time")
        self.assertEqual(lo, at("2023-05-04T20:05:00+00:00"))
        self.assertEqual((hi - lo).total_seconds(), 60)
        # SEC accepted at :52 falls inside the displayed minute; ordering is unknown.
        self.assertTrue(lo <= at("2023-05-04T20:05:52+00:00") < hi)
        lo, _ = MODULE.wire_minute_interval("Feb 21, 2023 4:11 PM Eastern Standard Time")
        self.assertEqual(lo, at("2023-02-21T21:11:00+00:00"))

    def test_inconsistent_or_missing_publisher_timezone_is_rejected(self):
        for value in ["May 4, 2023 4:05 PM Eastern Standard Time", "May 4, 2023 4:05 PM"]:
            with self.assertRaises(ValueError):
                MODULE.wire_minute_interval(value)

    def test_after_close_uses_next_real_session_across_dst_weekend(self):
        self.assertEqual(
            MODULE.first_reaction(at("2023-03-10T21:05:00+00:00"), CALENDAR),
            ("2023-03-13", "before_open"),
        )

    def test_preopen_stays_on_current_session(self):
        self.assertEqual(
            MODULE.first_reaction(at("2023-03-10T11:36:17+00:00"), CALENDAR),
            ("2023-03-10", "before_open"),
        )

    def test_close_boundary_is_not_misclassified_as_earlier_session(self):
        self.assertEqual(
            MODULE.first_reaction(at("2023-03-10T21:00:00+00:00"), CALENDAR)[0], "2023-03-13"
        )

    def test_intraday_disclosure_is_explicit(self):
        self.assertEqual(
            MODULE.first_reaction(at("2023-03-10T15:05:00+00:00"), CALENDAR),
            ("2023-03-10", "during_regular_session"),
        )

    def test_naive_and_out_of_window_disclosures_fail(self):
        for stamp in [datetime(2023, 3, 10, 21, 5), datetime(2023, 3, 14, tzinfo=UTC)]:
            with self.assertRaises(ValueError):
                MODULE.first_reaction(stamp, CALENDAR)

    def test_source_clock_is_retained_without_automatic_timezone_shift(self):
        obj = {
            "@type": "NewsArticle",
            "headline": "Quarterly release",
            "datePublished": "2024-05-02T15:11:00Z",
            "description": "Q&A at 2:30 p.m. PT",
        }
        raw = ('<script type="application/ld+json">' + json.dumps(obj) + "</script>").encode()
        self.assertEqual(MODULE.article_metadata(raw)["datePublished"], "2024-05-02T15:11:00Z")

    def test_multiple_article_clocks_are_not_silently_selected(self):
        raw = b'<script type="application/ld+json">{"@type":"NewsArticle"}</script>'
        with self.assertRaises(ValueError):
            MODULE.article_metadata(raw + raw)

    def test_webcast_text_is_not_a_publication_timestamp(self):
        with self.assertRaises(ValueError):
            MODULE.article_metadata(b"<p>Q&A starts at 2:30 p.m. PT.</p>")
