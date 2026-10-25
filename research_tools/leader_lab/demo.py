"""Synthetic known-policy recovery fixture; never a leader performance dataset."""

import csv
import math
from datetime import UTC, datetime, timedelta


def make_panel(path, minutes=7200):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fields = [
        "leader_id",
        "data_kind",
        "timestamp",
        "available_at",
        "action_end",
        "observed",
        "campaign_id",
        "symbol",
        "action",
        "open",
        "mark",
        "funding_rate",
        "x_signal",
        "x_long_open",
        "x_short_open",
    ]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for minute in range(minutes):
            phase = minute % 20
            when = start + timedelta(minutes=minute)
            action = {0: "OPEN_LONG", 5: "CLOSE_LONG", 10: "OPEN_SHORT", 15: "CLOSE_SHORT"}.get(
                phase, "HOLD"
            )
            price = 100 + math.sin(phase * math.pi / 10)
            writer.writerow(
                {
                    "leader_id": "999999",
                    "data_kind": "SYNTHETIC_KNOWN_POLICY",
                    "timestamp": when.isoformat(),
                    "available_at": (when - timedelta(minutes=1)).isoformat(),
                    "action_end": (when + timedelta(minutes=19 - phase)).isoformat(),
                    "observed": True,
                    "campaign_id": f"synthetic-{minute // 20}",
                    "symbol": "SYNTHETICUSDT",
                    "action": action,
                    "open": price,
                    "mark": price,
                    "funding_rate": 0,
                    "x_signal": phase,
                    "x_long_open": int(1 <= phase <= 5),
                    "x_short_open": int(11 <= phase <= 15),
                }
            )
