"""原始公开分钟的研究读取器，独立于 CZ 运行时及其账户。"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from tradingagents.research.quant.artifacts import sha as _sha
from tradingagents.research.quant.reference import market_features

MINUTE_MS = 60_000
RAW_KLINE_COLUMNS = 12
RAW_SCHEMA = "damorne-dynamic-public-kline-day.v1"


def _rows_sha(rows: list) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _merge_rows(target: dict[int, list], rows: list, *, start: int, end: int) -> None:
    previous = None
    for row in rows:
        if not isinstance(row, list) or len(row) != RAW_KLINE_COLUMNS:
            raise ValueError("invalid original Binance raw kline schema")
        clock = row[0]
        if (
            isinstance(clock, bool)
            or not isinstance(clock, int)
            or clock % MINUTE_MS
            or (previous is not None and clock <= previous)
        ):
            raise ValueError("raw candle clocks must be sorted unique whole minutes")
        previous = clock
        if row[6] != clock + MINUTE_MS - 1:
            raise ValueError("raw candle close clock does not match its open")
        if start <= clock < end:
            if clock in target and target[clock] != row:
                raise ValueError("conflicting same-symbol same-clock public raw observations")
            target[clock] = row


def _read_source(path: Path, *, symbol: str, expected_sha: str | None = None) -> dict:
    if expected_sha is not None and _sha(path) != expected_sha:
        raise ValueError("reusable raw source file hash drift")
    value = json.loads(path.read_text())
    if value.get("symbol") != symbol or value.get("rows_sha256") != _rows_sha(value["rows"]):
        raise ValueError("raw source identity or row-content drift")
    left, right = value["start_ms"], value["end_ms"]
    if left >= right or left % MINUTE_MS or right % MINUTE_MS:
        raise ValueError("raw source interval drift")
    if any(not left <= row[0] < right for row in value["rows"]):
        raise ValueError("raw source contains out-of-range rows")
    return value


def _raw_frame(rows: list, symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        [[row[1], row[2], row[3], row[4], row[5], row[7], row[10]] for row in rows],
        columns=market_features.RAW_COLUMNS,
        index=pd.DatetimeIndex(
            pd.to_datetime([row[0] + MINUTE_MS for row in rows], unit="ms", utc=True),
            name="bar_time",
        ),
    )
    return market_features._normalize_market_frame(frame, symbol=symbol)
