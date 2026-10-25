"""Production orchestration for the low-complexity non-ETH factor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from factors.crypto.code.weijie.later_non_eth_sequential_factor import (
    build_later_sequential_market,
)
from factors.crypto.code.weijie.non_eth_low_complexity_factor import (
    POLICY_ID,
    generate_non_eth_low_complexity_signals,
)


def generate_later_non_eth_live_signals(
    raw_five_minute_frames: dict[str, pd.DataFrame],
    raw_one_minute_frames: dict[str, pd.DataFrame],
    contract_metadata: dict[str, dict[str, Any]],
    *,
    activation_time: object,
    decision_end: object,
    symbol_activation_ms: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate cross-sectional reversal candidates from closed bars only.

    ``raw_one_minute_frames`` remains in the adapter signature for compatibility
    but is deliberately ignored.  Leader data is not part of this API.
    """

    del raw_one_minute_frames
    activation = _utc_timestamp(activation_time)
    endpoint = _utc_timestamp(decision_end)
    market = build_later_sequential_market(
        raw_five_minute_frames,
        contract_metadata,
    )
    if symbol_activation_ms:
        activation_by_symbol = {
            _clean_symbol(symbol): pd.Timestamp(
                int(timestamp_ms),
                unit="ms",
                tz="UTC",
            )
            for symbol, timestamp_ms in symbol_activation_ms.items()
        }
        market["symbol_activation_time"] = market["symbol"].map(
            activation_by_symbol,
        )
        market = market.loc[
            market["symbol_activation_time"].notna()
            & (market["time"] + pd.Timedelta(minutes=5)).ge(
                market["symbol_activation_time"],
            )
        ].drop(columns="symbol_activation_time")
    signals = generate_non_eth_low_complexity_signals(
        market,
        activation_time=activation,
        decision_end=endpoint,
    )
    return signals, market, pd.DataFrame()


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()


__all__ = ["POLICY_ID", "generate_later_non_eth_live_signals"]
