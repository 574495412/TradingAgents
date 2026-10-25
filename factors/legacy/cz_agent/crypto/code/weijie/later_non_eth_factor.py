"""Later-regime autonomous non-ETH market-state factor for CZ Live.

The factor uses only closed Binance market bars, public contract taxonomy and
replica inventory.  It never reads leader symbols, timestamps, directions,
prices, quantities or PnL.  The policy is the production form of the frozen
later-regime research thresholds; leader rows remain an offline evaluator.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

POLICY_ID = "weijie_non_eth_later_unified_public_market_state_v1"

_REQUIRED_COLUMNS = frozenset(
    {
        "bar_time",
        "symbol",
        "close",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_288",
        "quote_volume_ratio",
        "signed_seller_pressure",
        "subtype",
    },
)


@dataclass(frozen=True, slots=True)
class LaterNonEthFactorPolicy:
    """Frozen round-number thresholds from the later public-market chain."""

    direction_rearm_hours: float = 12.0
    minimum_observation_bars_after_activation: int = 2
    pow_minimum_quote_volume_24h: float = 50_000_000.0
    pow_maximum_nr12: float = -1.2
    pow_maximum_nr36: float = -1.2
    pow_minimum_activity_ratio12: float = 1.5
    pow_minimum_nr3: float = 0.0
    pow_minimum_negative_breadth12: float = 0.90
    layer1_minimum_quote_volume_24h: float = 100_000_000.0
    layer1_maximum_nr36: float = -1.0
    layer1_maximum_nr12: float = -0.2
    layer1_minimum_seller_pressure: float = 0.02
    layer1_minimum_negative_breadth3: float = 0.75
    layer1_minimum_negative_breadth12: float = 0.80
    low_volume_minimum_quote_volume_24h: float = 500_000.0
    low_volume_maximum_quote_volume_24h: float = 5_000_000.0
    low_volume_maximum_nr288: float = -1.15
    low_volume_minimum_nr12: float = 0.75
    low_volume_minimum_nr3: float = 0.35
    low_volume_minimum_nr36: float = 0.05
    low_volume_maximum_nr36: float = 0.40
    low_volume_minimum_activity_ratio12: float = 0.50
    low_volume_maximum_activity_ratio12: float = 1.50
    low_volume_maximum_seller_pressure: float = 0.0
    low_volume_minimum_group_median_nr12: float = 0.50
    low_volume_minimum_group_positive12: float = 0.85
    low_volume_maximum_group_median_nr288: float = -0.50
    oil_root_maximum_median_nr36: float = -0.30
    oil_root_minimum_seller_pressure: float = 0.05
    oil_reentry_maximum_median_nr12: float = -0.30
    oil_reentry_minimum_median_nr36: float = 0.0
    oil_reentry_maximum_median_nr288: float = -0.70
    oil_reentry_minimum_seller_pressure: float = 0.20
    tradfi_initial_maximum_median_nr36: float = -1.50
    tradfi_initial_minimum_negative_breadth: float = 0.95
    tradfi_initial_maximum_candidate_nr36: float = -2.00
    tradfi_recovery_minimum_median_nr12: float = 1.00
    tradfi_recovery_maximum_negative_breadth: float = 0.20
    tradfi_recovery_minimum_candidate_nr12: float = 1.00


def build_later_non_eth_market_states(
    feature_rows: pd.DataFrame,
    *,
    policy: LaterNonEthFactorPolicy | None = None,
) -> pd.DataFrame:
    """Attach the frozen later-regime family states to closed market rows."""

    cfg = policy or LaterNonEthFactorPolicy()
    missing = sorted(_REQUIRED_COLUMNS.difference(feature_rows.columns))
    if missing:
        raise ValueError(
            "later non-ETH feature rows missing: " + ",".join(missing),
        )
    market = feature_rows.copy()
    market["bar_time"] = pd.to_datetime(
        market["bar_time"],
        utc=True,
        errors="raise",
    )
    market["symbol"] = market["symbol"].map(_clean_symbol)
    if "rolling_quote_volume_24h" not in market:
        market["rolling_quote_volume_24h"] = 100_000_000.0
    if "contract_type" not in market:
        market["contract_type"] = np.where(
            market["subtype"].astype(str).eq("TradFi"),
            "TRADIFI_PERPETUAL",
            "PERPETUAL",
        )
    numeric = [
        "close",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_288",
        "quote_volume_ratio",
        "rolling_quote_volume_24h",
        "signed_seller_pressure",
    ]
    for column in numeric:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.replace([np.inf, -np.inf], np.nan)
    market["_subtypes"] = market.apply(_subtype_values, axis=1)
    market["_pow"] = market["_subtypes"].map(lambda values: "PoW" in values)
    market["_layer1"] = market["_subtypes"].map(
        lambda values: "Layer-1" in values,
    )
    market["_tradfi"] = market["contract_type"].astype(str).str.upper().eq(
        "TRADIFI_PERPETUAL",
    )
    market["_oil"] = market["symbol"].isin({"BZUSDT", "CLUSDT"})

    layer1_group = _group_state(market.loc[market["_layer1"]], "layer1")
    pow_group = _group_state(market.loc[market["_pow"]], "pow")
    oil_group = _group_state(market.loc[market["_oil"]], "oil")
    tradfi_group = _group_state(market.loc[market["_tradfi"]], "tradfi")
    for group in (layer1_group, pow_group, oil_group, tradfi_group):
        market = market.merge(group, on="bar_time", how="left")

    market["pow_short_state"] = (
        market["_pow"]
        & ~market["symbol"].eq("BTCUSDT")
        & market["rolling_quote_volume_24h"].ge(
            cfg.pow_minimum_quote_volume_24h,
        )
        & market["normalized_return_12"].le(cfg.pow_maximum_nr12)
        & market["normalized_return_36"].le(cfg.pow_maximum_nr36)
        & market["quote_volume_ratio"].ge(
            cfg.pow_minimum_activity_ratio12,
        )
        & market["normalized_return_3"].ge(cfg.pow_minimum_nr3)
        & market["pow_negative12"].ge(
            cfg.pow_minimum_negative_breadth12,
        )
    )
    market["layer1_short_state"] = (
        market["_layer1"]
        & ~market["symbol"].eq("SOLUSDT")
        & market["rolling_quote_volume_24h"].ge(
            cfg.layer1_minimum_quote_volume_24h,
        )
        & market["normalized_return_36"].le(cfg.layer1_maximum_nr36)
        & market["normalized_return_12"].le(cfg.layer1_maximum_nr12)
        & market["signed_seller_pressure"].ge(
            cfg.layer1_minimum_seller_pressure,
        )
        & market["layer1_negative3"].ge(
            cfg.layer1_minimum_negative_breadth3,
        )
        & market["layer1_negative12"].ge(
            cfg.layer1_minimum_negative_breadth12,
        )
    )
    market["low_volume_long_state"] = (
        market["_layer1"]
        & ~market["symbol"].eq("SOLUSDT")
        & market["rolling_quote_volume_24h"].between(
            cfg.low_volume_minimum_quote_volume_24h,
            cfg.low_volume_maximum_quote_volume_24h,
        )
        & market["normalized_return_288"].le(
            cfg.low_volume_maximum_nr288,
        )
        & market["normalized_return_12"].ge(
            cfg.low_volume_minimum_nr12,
        )
        & market["normalized_return_3"].ge(
            cfg.low_volume_minimum_nr3,
        )
        & market["normalized_return_36"].between(
            cfg.low_volume_minimum_nr36,
            cfg.low_volume_maximum_nr36,
        )
        & market["quote_volume_ratio"].between(
            cfg.low_volume_minimum_activity_ratio12,
            cfg.low_volume_maximum_activity_ratio12,
        )
        & market["signed_seller_pressure"].le(
            cfg.low_volume_maximum_seller_pressure,
        )
        & market["layer1_median12"].ge(
            cfg.low_volume_minimum_group_median_nr12,
        )
        & market["layer1_positive12"].ge(
            cfg.low_volume_minimum_group_positive12,
        )
        & market["layer1_median288"].le(
            cfg.low_volume_maximum_group_median_nr288,
        )
    )
    market["oil_short_state"] = (
        market["_oil"]
        & market["oil_median36"].le(cfg.oil_root_maximum_median_nr36)
        & market["normalized_return_12"].lt(0.0)
        & market["oil_negative3"].ge(0.5)
        & market["signed_seller_pressure"].ge(
            cfg.oil_root_minimum_seller_pressure,
        )
    )
    market["oil_long_state"] = (
        market["_oil"]
        & market["oil_median12"].le(
            cfg.oil_reentry_maximum_median_nr12,
        )
        & market["oil_median36"].ge(
            cfg.oil_reentry_minimum_median_nr36,
        )
        & market["oil_median288"].le(
            cfg.oil_reentry_maximum_median_nr288,
        )
        & market["normalized_return_12"].lt(0.0)
        & market["signed_seller_pressure"].ge(
            cfg.oil_reentry_minimum_seller_pressure,
        )
    )
    market["tradfi_short_state"] = (
        market["_tradfi"]
        & market["tradfi_median36"].le(
            cfg.tradfi_initial_maximum_median_nr36,
        )
        & market["tradfi_negative36"].ge(
            cfg.tradfi_initial_minimum_negative_breadth,
        )
        & market["normalized_return_36"].le(
            cfg.tradfi_initial_maximum_candidate_nr36,
        )
    )
    market["tradfi_long_state"] = (
        market["_tradfi"]
        & market["tradfi_median12"].ge(
            cfg.tradfi_recovery_minimum_median_nr12,
        )
        & market["tradfi_negative12"].le(
            cfg.tradfi_recovery_maximum_negative_breadth,
        )
        & market["normalized_return_12"].ge(
            cfg.tradfi_recovery_minimum_candidate_nr12,
        )
    )
    market["short_state"] = market[
        [
            "pow_short_state",
            "layer1_short_state",
            "oil_short_state",
            "tradfi_short_state",
        ]
    ].any(axis=1)
    market["long_state"] = market[
        [
            "low_volume_long_state",
            "oil_long_state",
            "tradfi_long_state",
        ]
    ].any(axis=1)
    market["strong_recovery_state"] = market["long_state"]
    return market


def generate_later_non_eth_market_signals(  # noqa: C901
    feature_rows: pd.DataFrame,
    *,
    symbol_activation_ms: Mapping[str, int],
    seeded_symbols: Collection[str] = (),
    policy: LaterNonEthFactorPolicy | None = None,
) -> pd.DataFrame:
    """Generate activation-safe root transitions from the later state panel."""

    del seeded_symbols
    cfg = policy or LaterNonEthFactorPolicy()
    if feature_rows.empty or not symbol_activation_ms:
        return _empty_signals()
    market = build_later_non_eth_market_states(feature_rows, policy=cfg)
    activation = {
        _clean_symbol(symbol): pd.Timestamp(int(value), unit="ms", tz="UTC")
        for symbol, value in symbol_activation_ms.items()
    }
    market["activation_time"] = market["symbol"].map(activation)
    market = market.loc[
        market["activation_time"].notna()
        & market["bar_time"].ge(market["activation_time"])
    ].copy()
    if market.empty:
        return _empty_signals()

    families = (
        (
            "pow_short_state",
            "OPEN_SHORT",
            "dynamic_pow_breadth_break",
            "POW_ROOT_SHORT",
            "PoW 板块广度下破中的高流动性根仓",
            "rolling_quote_volume_24h",
        ),
        (
            "layer1_short_state",
            "OPEN_SHORT",
            "dynamic_layer1_liquidity_follower",
            "LAYER1_ROOT_SHORT",
            "Layer-1 板块卖压扩散中的流动性跟随仓",
            "rolling_quote_volume_24h",
        ),
        (
            "low_volume_long_state",
            "OPEN_LONG",
            "dynamic_low_volume_layer1_rotation",
            "LOW_VOLUME_ROOT_LONG",
            "Layer-1 深跌后的低成交量恢复轮动",
            "normalized_return_3",
        ),
        (
            "oil_short_state",
            "OPEN_SHORT",
            "dynamic_oil_pair_break",
            "OIL_ROOT_SHORT",
            "公开原油合约组同步走弱",
            "quote_volume_ratio",
        ),
        (
            "oil_long_state",
            "OPEN_LONG",
            "dynamic_oil_pair_reentry",
            "OIL_ROOT_LONG",
            "公开原油合约组深跌后的结构恢复",
            "quote_volume_ratio",
        ),
        (
            "tradfi_short_state",
            "OPEN_SHORT",
            "initial_direct_equity_group_break",
            "TRADFI_ROOT_SHORT",
            "TradFi 相关组同步下破",
            "rolling_quote_volume_24h",
        ),
        (
            "tradfi_long_state",
            "OPEN_LONG",
            "discounted_post_campaign_group_recovery",
            "TRADFI_ROOT_LONG",
            "TradFi 相关组折价后的同步恢复",
            "rolling_quote_volume_24h",
        ),
    )
    records: list[dict[str, object]] = []
    minimum_bars = max(2, int(cfg.minimum_observation_bars_after_activation))
    rearm = pd.Timedelta(hours=float(cfg.direction_rearm_hours))
    for state, action, family, profile, reason, rank_column in families:
        candidates: list[pd.Series] = []
        for _symbol, group in market.groupby("symbol", sort=True):
            ordered = group.sort_values("bar_time", kind="stable")
            prior = False
            observations = 0
            last_signal: pd.Timestamp | None = None
            for _, row in ordered.iterrows():
                observations += 1
                active = bool(row[state])
                onset = active and not prior and observations >= minimum_bars
                prior = active
                timestamp = pd.Timestamp(row["bar_time"])
                if not onset:
                    continue
                if last_signal is not None and timestamp - last_signal < rearm:
                    continue
                candidates.append(row)
                last_signal = timestamp
        if not candidates:
            continue
        candidate_frame = pd.DataFrame(candidates)
        for timestamp, rows in candidate_frame.groupby("bar_time", sort=True):
            ranked = rows.sort_values(
                [rank_column, "symbol"],
                ascending=[False, True],
                kind="stable",
            )
            row = ranked.iloc[0]
            symbol = str(row["symbol"])
            signal_id = (
                f"{POLICY_ID}|{pd.Timestamp(timestamp).value}|"
                f"{symbol}|{action}|{family}"
            )
            records.append(
                {
                    "signal_id": signal_id,
                    "signal_time": pd.Timestamp(timestamp),
                    "symbol": symbol,
                    "action": action,
                    "operation_type": "OPEN",
                    "signal_price": float(row["close"]),
                    "factor_family": family,
                    "lifecycle_profile": profile,
                    "reason": reason,
                    "policy_id": POLICY_ID,
                    "market_only": True,
                    "leader_direction_used": False,
                },
            )
    if not records:
        return _empty_signals()
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["signal_time", "symbol", "action"], kind="stable")
        .drop_duplicates("signal_id", keep="first")
        .reset_index(drop=True)
    )


def _group_state(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "bar_time",
        f"{prefix}_negative3",
        f"{prefix}_negative12",
        f"{prefix}_negative36",
        f"{prefix}_positive12",
        f"{prefix}_median12",
        f"{prefix}_median36",
        f"{prefix}_median288",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return (
        frame.groupby("bar_time", sort=True)
        .agg(
            **{
                f"{prefix}_negative3": (
                    "normalized_return_3",
                    lambda value: value.lt(0.0).mean(),
                ),
                f"{prefix}_negative12": (
                    "normalized_return_12",
                    lambda value: value.lt(0.0).mean(),
                ),
                f"{prefix}_negative36": (
                    "normalized_return_36",
                    lambda value: value.lt(0.0).mean(),
                ),
                f"{prefix}_positive12": (
                    "normalized_return_12",
                    lambda value: value.gt(0.0).mean(),
                ),
                f"{prefix}_median12": ("normalized_return_12", "median"),
                f"{prefix}_median36": ("normalized_return_36", "median"),
                f"{prefix}_median288": ("normalized_return_288", "median"),
            },
        )
        .reset_index()
    )


def _subtype_values(row: pd.Series) -> frozenset[str]:
    raw = row.get("underlying_subtypes")
    if isinstance(raw, (list, tuple, set, frozenset)):
        values = {str(value) for value in raw}
    else:
        values = {str(row.get("subtype") or "Other")}
    return frozenset(values)


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").strip()


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "signal_time",
            "symbol",
            "action",
            "operation_type",
            "signal_price",
            "factor_family",
            "lifecycle_profile",
            "reason",
            "policy_id",
            "market_only",
            "leader_direction_used",
        ],
    )


__all__ = [
    "POLICY_ID",
    "LaterNonEthFactorPolicy",
    "build_later_non_eth_market_states",
    "generate_later_non_eth_market_signals",
]
