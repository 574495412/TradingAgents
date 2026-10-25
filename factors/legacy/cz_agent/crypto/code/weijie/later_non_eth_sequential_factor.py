"""Exact sequential later-regime crypto/commodity chain for CZ Live.

The module consumes only closed public market bars, public contract taxonomy
and autonomous replica actions. It does not accept leader labels or activity.
"""

from __future__ import annotations

# The frozen chain is intentionally sequential and threshold-heavy.
# ruff: noqa: B023, C901, EM101, EM102, PLR0911, PLR0912, PLR0915, TRY003
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

POLICY_ID = "weijie_non_eth_later_unified_public_market_state_v1"
REGIME_START = pd.Timestamp("1970-01-01T00:00:00Z")


@dataclass(frozen=True, slots=True)
class LaterUnifiedPolicy:
    """Small public-market and replica-inventory state policy."""

    decision_bar_minutes: int = 5
    rolling_normalizer_bars: int = 288
    rolling_normalizer_minimum_bars: int = 100
    rolling_liquidity_bars: int = 288
    rolling_liquidity_minimum_bars: int = 200
    crypto_resume_cooldown_hours: float = 4.0
    pow_minimum_quote_volume_24h: float = 50_000_000.0
    pow_maximum_nr12: float = -1.2
    pow_maximum_nr36: float = -1.2
    pow_minimum_activity_ratio12: float = 1.5
    pow_minimum_nr3: float = 0.0
    pow_minimum_negative_breadth12: float = 0.9
    layer1_follower_delay_minutes: int = 30
    layer1_minimum_quote_volume_24h: float = 100_000_000.0
    layer1_maximum_nr36: float = -1.0
    layer1_maximum_nr12: float = -0.2
    layer1_maximum_flow12: float = -0.02
    layer1_minimum_negative_breadth3: float = 0.75
    layer1_minimum_negative_breadth12: float = 0.8
    short_hedge_minimum_hours: float = 8.0
    short_hedge_minimum_nr36: float = 1.4
    short_hedge_minimum_nr12: float = 0.0
    short_hedge_minimum_group_positive12: float = 0.7
    short_hedge_minimum_group_positive36: float = 0.6
    pow_short_trailing_activation_pct: float = 2.3
    pow_short_trailing_recovery_pct: float = 0.4
    pow_long_reentry_delay_hours: float = 6.0
    pow_long_reentry_maximum_nr288: float = -1.0
    pow_long_reentry_minimum_nr36: float = 0.0
    pow_long_reentry_maximum_activity_ratio12: float = 0.6
    pow_long_reentry_minimum_group_positive36: float = 0.7
    long_hedge_minimum_minutes: int = 30
    pow_long_hedge_adverse_pct: float = 1.8
    low_volume_rotation_delay_hours: float = 4.0
    low_volume_minimum_quote_volume_24h: float = 500_000.0
    low_volume_maximum_quote_volume_24h: float = 5_000_000.0
    low_volume_maximum_nr288: float = -1.15
    low_volume_minimum_nr12: float = 0.75
    low_volume_minimum_nr3: float = 0.35
    low_volume_minimum_nr36: float = 0.05
    low_volume_maximum_nr36: float = 0.4
    low_volume_minimum_activity_ratio12: float = 0.5
    low_volume_maximum_activity_ratio12: float = 1.5
    low_volume_minimum_flow3: float = 0.0
    low_volume_minimum_flow12: float = 0.0
    low_volume_minimum_group_median_nr12: float = 0.5
    low_volume_minimum_group_positive12: float = 0.85
    low_volume_maximum_group_median_nr288: float = -0.5
    low_volume_hedge_minimum_minutes: int = 60
    low_volume_hedge_adverse_pct: float = 0.35
    low_volume_hedge_maximum_flow12: float = -0.15
    oil_root_delay_minutes: int = 60
    oil_root_maximum_median_nr36: float = -0.3
    oil_root_maximum_median_flow12: float = -0.05
    oil_root_take_profit_pct: float = 2.3
    oil_reentry_delay_hours: float = 12.0
    oil_reentry_maximum_median_nr12: float = -0.3
    oil_reentry_minimum_median_nr36: float = 0.0
    oil_reentry_maximum_median_nr288: float = -0.7
    oil_reentry_maximum_median_flow12: float = -0.2
    oil_long_hedge_minimum_minutes: int = 60
    oil_long_hedge_adverse_pct: float = 0.3
    tradfi_long_add_adverse_pct: float = 4.95
    tradfi_long_add_maximum_adverse_pct: float = 5.3
    tradfi_long_add_funding_window_minutes: int = 15
    pow_hedge_add_adverse_pct: float = 3.0
    pow_hedge_add_activity_ratio: float = 6.0
    low_volume_root_take_profit_pct: float = 2.5
    low_volume_rehedge_delay_hours: float = 10.0
    low_volume_rehedge_adverse_pct: float = 3.4


def build_later_sequential_market(
    raw_frames: dict[str, pd.DataFrame],
    contract_metadata: dict[str, dict[str, Any]],
    *,
    policy: LaterUnifiedPolicy | None = None,
) -> pd.DataFrame:
    """Build the exact rolling-normalized five-minute production panel."""

    cfg = policy or LaterUnifiedPolicy()
    frames: list[pd.DataFrame] = []
    for raw_symbol, source in sorted(raw_frames.items()):
        symbol = _clean_symbol(raw_symbol)
        metadata = contract_metadata.get(symbol, {})
        if source is None or source.empty:
            continue
        taker_column = (
            "taker_buy_quote"
            if "taker_buy_quote" in source
            else "taker_buy_quote_volume"
        )
        required = {"close", "quote_volume", taker_column}
        if not required.issubset(source.columns):
            continue
        frame = source.loc[
            :,
            ["close", "quote_volume", taker_column],
        ].copy()
        frame = frame.rename(
            columns={taker_column: "taker_buy_quote_volume"},
        )
        index = pd.DatetimeIndex(frame.index)
        frame["time"] = (
            index.tz_localize("Asia/Shanghai").tz_convert("UTC")
            if index.tz is None
            else index.tz_convert("UTC")
        )
        frame["symbol"] = symbol
        frame["underlying_type"] = str(
            metadata.get("underlyingType", ""),
        )
        frame["subtypes"] = [
            tuple(
                str(value)
                for value in metadata.get("underlyingSubType", ())
            ),
        ] * len(frame)
        frames.append(frame.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    market = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "time"], kind="stable")
        .drop_duplicates(["time", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    for column in ["close", "quote_volume", "taker_buy_quote_volume"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    grouped = market.groupby("symbol", sort=False, group_keys=False)
    for horizon in [3, 12, 36, 288]:
        raw = grouped["close"].pct_change(horizon, fill_method=None) * 100.0
        market[f"ret{horizon}"] = raw
        scale = raw.groupby(market["symbol"], sort=False).transform(
            lambda value: value.abs()
            .rolling(
                cfg.rolling_normalizer_bars,
                min_periods=cfg.rolling_normalizer_minimum_bars,
            )
            .median(),
        )
        market[f"nr{horizon}"] = raw.div(scale.replace(0.0, np.nan))
    market["q288"] = grouped["quote_volume"].transform(
        lambda value: value.rolling(
            cfg.rolling_liquidity_bars,
            min_periods=cfg.rolling_liquidity_minimum_bars,
        ).sum(),
    )
    market["q12"] = grouped["quote_volume"].transform(
        lambda value: value.rolling(12, min_periods=12).sum(),
    )
    q12_median = market.groupby("symbol", sort=False)["q12"].transform(
        lambda value: value.shift(1)
        .rolling(
            cfg.rolling_normalizer_bars,
            min_periods=cfg.rolling_normalizer_minimum_bars,
        )
        .median(),
    )
    market["activity_ratio12"] = market["q12"].div(
        q12_median.replace(0.0, np.nan),
    )
    q1_median = grouped["quote_volume"].transform(
        lambda value: value.shift(1).rolling(36, min_periods=12).median(),
    )
    market["activity_ratio1"] = market["quote_volume"].div(
        q1_median.replace(0.0, np.nan),
    )
    market["net_flow"] = (
        2.0 * market["taker_buy_quote_volume"] - market["quote_volume"]
    )
    for horizon in [3, 12]:
        numerator = market.groupby("symbol", sort=False)[
            "net_flow"
        ].transform(
            lambda value: value.rolling(
                horizon,
                min_periods=horizon,
            ).sum(),
        )
        denominator = grouped["quote_volume"].transform(
            lambda value: value.rolling(
                horizon,
                min_periods=horizon,
            ).sum(),
        )
        market[f"flow{horizon}"] = numerator.div(
            denominator.replace(0.0, np.nan),
        )
    return market


def generate_later_sequential_signals(
    market: pd.DataFrame,
    base_signals: pd.DataFrame,
    *,
    activation_time: object,
    decision_end: object,
    policy: LaterUnifiedPolicy | None = None,
) -> pd.DataFrame:
    """Advance the frozen cross-asset sequence through available closed bars."""

    cfg = policy or LaterUnifiedPolicy()
    if market.empty:
        return _empty_signals()
    activation = _utc_timestamp(activation_time)
    endpoint = _utc_timestamp(decision_end)
    market = market.loc[market["time"].le(endpoint)].copy()
    base = _normalize_base_signals(
        base_signals,
        activation=activation,
        endpoint=endpoint,
    )
    records = base.to_dict("records")
    sol_times = base.loc[base["symbol"].eq("SOLUSDT"), "signal_time"]
    if sol_times.empty:
        return _to_live_signals(records)
    crypto_start = sol_times.max() + pd.Timedelta(
        hours=cfg.crypto_resume_cooldown_hours,
    )

    pow_market = _subtype_market(market, "PoW")
    pow_pool = pow_market.merge(
        _group_state(pow_market, prefix="group"),
        on="time",
        how="left",
    )
    pow_mask = (
        pow_pool["time"].add(pd.Timedelta(minutes=5)).ge(crypto_start)
        & pow_pool["underlying_type"].eq("COIN")
        & ~pow_pool["symbol"].eq("BTCUSDT")
        & pow_pool["q288"].ge(cfg.pow_minimum_quote_volume_24h)
        & pow_pool["nr12"].le(cfg.pow_maximum_nr12)
        & pow_pool["nr36"].le(cfg.pow_maximum_nr36)
        & pow_pool["activity_ratio12"].ge(cfg.pow_minimum_activity_ratio12)
        & pow_pool["nr3"].ge(cfg.pow_minimum_nr3)
        & pow_pool["group_negative12"].ge(
            cfg.pow_minimum_negative_breadth12,
        )
    )
    pow_row = _maybe_first_ranked(
        pow_pool.loc[pow_mask],
        rank_column="q288",
        ascending=False,
    )
    if pow_row is None:
        return _to_live_signals(records)
    pow_short = _market_record(
        pow_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_pow_breadth_break",
    )
    records.append(pow_short)

    layer1_market = _subtype_market(market, "Layer-1")
    layer1_pool = layer1_market.merge(
        _group_state(layer1_market, prefix="group"),
        on="time",
        how="left",
    )
    layer1_start = pd.Timestamp(pow_short["signal_time"]) + pd.Timedelta(
        minutes=cfg.layer1_follower_delay_minutes,
    )
    layer1_mask = (
        layer1_pool["time"].add(pd.Timedelta(minutes=5)).ge(layer1_start)
        & ~layer1_pool["symbol"].eq("SOLUSDT")
        & layer1_pool["q288"].ge(cfg.layer1_minimum_quote_volume_24h)
        & layer1_pool["nr36"].le(cfg.layer1_maximum_nr36)
        & layer1_pool["nr12"].le(cfg.layer1_maximum_nr12)
        & layer1_pool["flow12"].le(cfg.layer1_maximum_flow12)
        & layer1_pool["group_negative3"].ge(
            cfg.layer1_minimum_negative_breadth3,
        )
        & layer1_pool["group_negative12"].ge(
            cfg.layer1_minimum_negative_breadth12,
        )
    )
    layer1_row = _maybe_first_ranked(
        layer1_pool.loc[layer1_mask],
        rank_column="q288",
        ascending=False,
    )
    if layer1_row is None:
        return _to_live_signals(records)
    layer1_short = _market_record(
        layer1_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_layer1_liquidity_follower",
    )
    records.append(layer1_short)
    layer1_symbol = str(layer1_short["symbol"])

    layer1_future = layer1_pool.loc[
        layer1_pool["symbol"].eq(layer1_symbol)
        & layer1_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pd.Timestamp(layer1_short["signal_time"])
            + pd.Timedelta(hours=cfg.short_hedge_minimum_hours),
        )
        & layer1_pool["nr36"].ge(cfg.short_hedge_minimum_nr36)
        & layer1_pool["nr12"].ge(cfg.short_hedge_minimum_nr12)
        & layer1_pool["group_positive12"].ge(
            cfg.short_hedge_minimum_group_positive12,
        )
        & layer1_pool["group_positive36"].ge(
            cfg.short_hedge_minimum_group_positive36,
        )
    ]
    layer1_hedge_row = _maybe_first_row(layer1_future)
    if layer1_hedge_row is not None:
        records.append(
            _market_record(
                layer1_hedge_row,
                direction="LONG",
                operation_type="OPEN",
                role="HEDGE",
                reason="mature_layer1_short_recovery",
            ),
        )

    pow_close = _optional_call(
        _short_trailing_close,
        market,
        pow_short,
        activation_pct=cfg.pow_short_trailing_activation_pct,
        recovery_pct=cfg.pow_short_trailing_recovery_pct,
    )
    if pow_close is None:
        return _to_live_signals(records)
    pow_symbol = str(pow_short["symbol"])
    pow_reentry_pool = pow_pool.loc[
        pow_pool["symbol"].eq(pow_symbol)
        & pow_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pow_close["signal_time"]
            + pd.Timedelta(hours=cfg.pow_long_reentry_delay_hours),
        )
        & pow_pool["nr288"].le(cfg.pow_long_reentry_maximum_nr288)
        & pow_pool["nr36"].ge(cfg.pow_long_reentry_minimum_nr36)
        & pow_pool["activity_ratio12"].le(
            cfg.pow_long_reentry_maximum_activity_ratio12,
        )
        & pow_pool["group_positive36"].ge(
            cfg.pow_long_reentry_minimum_group_positive36,
        )
    ]
    pow_long_row = _maybe_first_row(pow_reentry_pool)
    if pow_long_row is None:
        return _to_live_signals(records)
    pow_long = _market_record(
        pow_long_row,
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_pow_post_trailing_reentry",
    )
    records.append(pow_long)
    pow_long_hedge = _optional_call(
        _adverse_hedge,
        market,
        pow_long,
        minimum_minutes=cfg.long_hedge_minimum_minutes,
        adverse_pct=cfg.pow_long_hedge_adverse_pct,
        direction="SHORT",
        reason="pow_long_adverse_hedge",
    )
    if pow_long_hedge is None:
        return _to_live_signals(records)
    records.append(pow_long_hedge)

    rotation_start = pd.Timestamp(
        pow_long_hedge["signal_time"],
    ) + pd.Timedelta(hours=cfg.low_volume_rotation_delay_hours)
    low_volume_mask = (
        layer1_pool["time"].add(pd.Timedelta(minutes=5)).ge(rotation_start)
        & ~layer1_pool["symbol"].isin({"SOLUSDT", layer1_symbol})
        & layer1_pool["q288"].between(
            cfg.low_volume_minimum_quote_volume_24h,
            cfg.low_volume_maximum_quote_volume_24h,
        )
        & layer1_pool["nr288"].le(cfg.low_volume_maximum_nr288)
        & layer1_pool["nr12"].ge(cfg.low_volume_minimum_nr12)
        & layer1_pool["nr3"].ge(cfg.low_volume_minimum_nr3)
        & layer1_pool["nr36"].between(
            cfg.low_volume_minimum_nr36,
            cfg.low_volume_maximum_nr36,
        )
        & layer1_pool["activity_ratio12"].between(
            cfg.low_volume_minimum_activity_ratio12,
            cfg.low_volume_maximum_activity_ratio12,
        )
        & layer1_pool["flow3"].gt(cfg.low_volume_minimum_flow3)
        & layer1_pool["flow12"].gt(cfg.low_volume_minimum_flow12)
        & layer1_pool["group_median_nr12"].ge(
            cfg.low_volume_minimum_group_median_nr12,
        )
        & layer1_pool["group_positive12"].ge(
            cfg.low_volume_minimum_group_positive12,
        )
        & layer1_pool["group_median_nr288"].le(
            cfg.low_volume_maximum_group_median_nr288,
        )
    )
    low_row = _maybe_first_ranked(
        layer1_pool.loc[low_volume_mask],
        rank_column="nr3",
        ascending=False,
    )
    if low_row is None:
        return _to_live_signals(records)
    low_long = _market_record(
        low_row,
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_low_volume_layer1_rotation",
    )
    records.append(low_long)
    low_future = layer1_pool.loc[
        layer1_pool["symbol"].eq(str(low_long["symbol"]))
        & layer1_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pd.Timestamp(low_long["signal_time"])
            + pd.Timedelta(minutes=cfg.low_volume_hedge_minimum_minutes),
        )
    ].copy()
    low_future["adverse_pct"] = (
        float(low_long["signal_price"]) / low_future["close"] - 1.0
    ) * 100.0
    low_hedge_row = _maybe_first_row(
        low_future.loc[
            low_future["adverse_pct"].ge(cfg.low_volume_hedge_adverse_pct)
            & low_future["flow12"].le(cfg.low_volume_hedge_maximum_flow12)
        ],
    )
    if low_hedge_row is None:
        return _to_live_signals(records)
    low_short = _market_record(
        low_hedge_row,
        direction="SHORT",
        operation_type="OPEN",
        role="HEDGE",
        reason="low_volume_layer1_adverse_flow_hedge",
    )
    records.append(low_short)

    oil_market = market.loc[
        market["symbol"].isin({"BZUSDT", "CLUSDT"}),
    ].copy()
    oil_pool = oil_market.merge(
        _group_state(oil_market, prefix="oil"),
        on="time",
        how="left",
    )
    oil_start = pd.Timestamp(low_long["signal_time"]) + pd.Timedelta(
        minutes=cfg.oil_root_delay_minutes,
    )
    oil_mask = (
        oil_pool["time"].add(pd.Timedelta(minutes=5)).ge(oil_start)
        & oil_pool["oil_median_nr36"].le(
            cfg.oil_root_maximum_median_nr36,
        )
        & oil_pool["nr12"].lt(0.0)
        & oil_pool["oil_negative3"].ge(0.5)
        & oil_pool["oil_median_flow12"].le(
            cfg.oil_root_maximum_median_flow12,
        )
    )
    oil_short_row = _maybe_first_ranked(
        oil_pool.loc[oil_mask],
        rank_column="activity_ratio12",
        ascending=False,
    )
    if oil_short_row is None:
        return _to_live_signals(records)
    oil_short = _market_record(
        oil_short_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_oil_pair_break",
    )
    records.append(oil_short)
    oil_close = _optional_call(
        _take_profit_close,
        market,
        oil_short,
        cfg.oil_root_take_profit_pct,
    )
    if oil_close is None:
        return _to_live_signals(records)
    oil_reentry = oil_pool.loc[
        oil_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            oil_close["signal_time"]
            + pd.Timedelta(hours=cfg.oil_reentry_delay_hours),
        )
        & oil_pool["oil_median_nr12"].le(
            cfg.oil_reentry_maximum_median_nr12,
        )
        & oil_pool["oil_median_nr36"].ge(
            cfg.oil_reentry_minimum_median_nr36,
        )
        & oil_pool["oil_median_nr288"].le(
            cfg.oil_reentry_maximum_median_nr288,
        )
        & oil_pool["nr12"].lt(0.0)
        & oil_pool["oil_median_flow12"].le(
            cfg.oil_reentry_maximum_median_flow12,
        )
    ]
    oil_long_row = _maybe_first_ranked(
        oil_reentry,
        rank_column="activity_ratio12",
        ascending=False,
    )
    if oil_long_row is None:
        return _to_live_signals(records)
    oil_long = _market_record(
        oil_long_row,
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_oil_pair_reentry",
    )
    records.append(oil_long)
    oil_hedge = _optional_call(
        _adverse_hedge,
        market,
        oil_long,
        minimum_minutes=cfg.oil_long_hedge_minimum_minutes,
        adverse_pct=cfg.oil_long_hedge_adverse_pct,
        direction="SHORT",
        reason="oil_long_adverse_hedge",
    )
    if oil_hedge is not None:
        records.append(oil_hedge)

    for function, args in (
        (_tradfi_inventory_add, (market, base, cfg)),
        (_pow_hedge_add, (market, pow_long_hedge, cfg)),
        (_low_volume_rehedge, (market, low_long, low_short, cfg)),
    ):
        record = _optional_call(function, *args)
        if record is not None:
            records.append(record)
    return _to_live_signals(records)


def _generate_candidates(
    market: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
    base: pd.DataFrame,
    policy: LaterUnifiedPolicy,
) -> pd.DataFrame:
    del metadata
    records = base.to_dict("records")
    sol_last = base.loc[base["symbol"].eq("SOLUSDT"), "signal_time"].max()
    crypto_start = sol_last + pd.Timedelta(
        hours=policy.crypto_resume_cooldown_hours,
    )

    pow_market = _subtype_market(market, "PoW")
    pow_group = _group_state(pow_market, prefix="group")
    pow_pool = pow_market.merge(pow_group, on="time", how="left")
    pow_mask = (
        pow_pool["time"].add(pd.Timedelta(minutes=5)).ge(crypto_start)
        & pow_pool["underlying_type"].eq("COIN")
        & ~pow_pool["symbol"].eq("BTCUSDT")
        & pow_pool["q288"].ge(policy.pow_minimum_quote_volume_24h)
        & pow_pool["nr12"].le(policy.pow_maximum_nr12)
        & pow_pool["nr36"].le(policy.pow_maximum_nr36)
        & pow_pool["activity_ratio12"].ge(
            policy.pow_minimum_activity_ratio12,
        )
        & pow_pool["nr3"].ge(policy.pow_minimum_nr3)
        & pow_pool["group_negative12"].ge(
            policy.pow_minimum_negative_breadth12,
        )
    )
    pow_row = _first_ranked(
        pow_pool.loc[pow_mask],
        rank_column="q288",
        ascending=False,
    )
    pow_short = _market_record(
        pow_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_pow_breadth_break",
    )
    records.append(pow_short)

    layer1_market = _subtype_market(market, "Layer-1")
    layer1_group = _group_state(layer1_market, prefix="group")
    layer1_pool = layer1_market.merge(layer1_group, on="time", how="left")
    layer1_start = pd.Timestamp(pow_short["signal_time"]) + pd.Timedelta(
        minutes=policy.layer1_follower_delay_minutes,
    )
    layer1_mask = (
        layer1_pool["time"].add(pd.Timedelta(minutes=5)).ge(layer1_start)
        & ~layer1_pool["symbol"].eq("SOLUSDT")
        & layer1_pool["q288"].ge(
            policy.layer1_minimum_quote_volume_24h,
        )
        & layer1_pool["nr36"].le(policy.layer1_maximum_nr36)
        & layer1_pool["nr12"].le(policy.layer1_maximum_nr12)
        & layer1_pool["flow12"].le(policy.layer1_maximum_flow12)
        & layer1_pool["group_negative3"].ge(
            policy.layer1_minimum_negative_breadth3,
        )
        & layer1_pool["group_negative12"].ge(
            policy.layer1_minimum_negative_breadth12,
        )
    )
    layer1_row = _first_ranked(
        layer1_pool.loc[layer1_mask],
        rank_column="q288",
        ascending=False,
    )
    layer1_short = _market_record(
        layer1_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_layer1_liquidity_follower",
    )
    records.append(layer1_short)

    layer1_symbol = str(layer1_short["symbol"])
    layer1_future = layer1_pool.loc[
        layer1_pool["symbol"].eq(layer1_symbol)
        & layer1_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pd.Timestamp(layer1_short["signal_time"])
            + pd.Timedelta(hours=policy.short_hedge_minimum_hours),
        )
        & layer1_pool["nr36"].ge(policy.short_hedge_minimum_nr36)
        & layer1_pool["nr12"].ge(policy.short_hedge_minimum_nr12)
        & layer1_pool["group_positive12"].ge(
            policy.short_hedge_minimum_group_positive12,
        )
        & layer1_pool["group_positive36"].ge(
            policy.short_hedge_minimum_group_positive36,
        )
    ]
    layer1_hedge = _market_record(
        _first_row(layer1_future),
        direction="LONG",
        operation_type="OPEN",
        role="HEDGE",
        reason="mature_layer1_short_recovery",
    )
    records.append(layer1_hedge)

    pow_close = _short_trailing_close(
        market,
        pow_short,
        activation_pct=policy.pow_short_trailing_activation_pct,
        recovery_pct=policy.pow_short_trailing_recovery_pct,
    )
    pow_symbol = str(pow_short["symbol"])
    pow_reentry_pool = pow_pool.loc[
        pow_pool["symbol"].eq(pow_symbol)
        & pow_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pow_close["signal_time"]
            + pd.Timedelta(hours=policy.pow_long_reentry_delay_hours),
        )
        & pow_pool["nr288"].le(policy.pow_long_reentry_maximum_nr288)
        & pow_pool["nr36"].ge(policy.pow_long_reentry_minimum_nr36)
        & pow_pool["activity_ratio12"].le(
            policy.pow_long_reentry_maximum_activity_ratio12,
        )
        & pow_pool["group_positive36"].ge(
            policy.pow_long_reentry_minimum_group_positive36,
        )
    ]
    pow_long = _market_record(
        _first_row(pow_reentry_pool),
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_pow_post_trailing_reentry",
    )
    records.append(pow_long)
    pow_long_hedge = _adverse_hedge(
        market,
        pow_long,
        minimum_minutes=policy.long_hedge_minimum_minutes,
        adverse_pct=policy.pow_long_hedge_adverse_pct,
        direction="SHORT",
        reason="pow_long_adverse_hedge",
    )
    records.append(pow_long_hedge)

    rotation_start = pd.Timestamp(
        pow_long_hedge["signal_time"],
    ) + pd.Timedelta(hours=policy.low_volume_rotation_delay_hours)
    low_volume_mask = (
        layer1_pool["time"].add(pd.Timedelta(minutes=5)).ge(rotation_start)
        & ~layer1_pool["symbol"].isin(
            {"SOLUSDT", layer1_symbol},
        )
        & layer1_pool["q288"].between(
            policy.low_volume_minimum_quote_volume_24h,
            policy.low_volume_maximum_quote_volume_24h,
        )
        & layer1_pool["nr288"].le(policy.low_volume_maximum_nr288)
        & layer1_pool["nr12"].ge(policy.low_volume_minimum_nr12)
        & layer1_pool["nr3"].ge(policy.low_volume_minimum_nr3)
        & layer1_pool["nr36"].between(
            policy.low_volume_minimum_nr36,
            policy.low_volume_maximum_nr36,
        )
        & layer1_pool["activity_ratio12"].between(
            policy.low_volume_minimum_activity_ratio12,
            policy.low_volume_maximum_activity_ratio12,
        )
        & layer1_pool["flow3"].gt(policy.low_volume_minimum_flow3)
        & layer1_pool["flow12"].gt(policy.low_volume_minimum_flow12)
        & layer1_pool["group_median_nr12"].ge(
            policy.low_volume_minimum_group_median_nr12,
        )
        & layer1_pool["group_positive12"].ge(
            policy.low_volume_minimum_group_positive12,
        )
        & layer1_pool["group_median_nr288"].le(
            policy.low_volume_maximum_group_median_nr288,
        )
    )
    rotation_row = _first_ranked(
        layer1_pool.loc[low_volume_mask],
        rank_column="nr3",
        ascending=False,
    )
    low_volume_long = _market_record(
        rotation_row,
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_low_volume_layer1_rotation",
    )
    records.append(low_volume_long)

    low_symbol = str(low_volume_long["symbol"])
    low_future = layer1_pool.loc[
        layer1_pool["symbol"].eq(low_symbol)
        & layer1_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            pd.Timestamp(low_volume_long["signal_time"])
            + pd.Timedelta(minutes=policy.low_volume_hedge_minimum_minutes),
        )
    ].copy()
    low_future["adverse_pct"] = (
        float(low_volume_long["signal_price"]) / low_future["close"] - 1.0
    ) * 100.0
    low_hedge_row = _first_row(
        low_future.loc[
            low_future["adverse_pct"].ge(
                policy.low_volume_hedge_adverse_pct,
            )
            & low_future["flow12"].le(
                policy.low_volume_hedge_maximum_flow12,
            )
        ],
    )
    low_volume_short = _market_record(
        low_hedge_row,
        direction="SHORT",
        operation_type="OPEN",
        role="HEDGE",
        reason="low_volume_layer1_adverse_flow_hedge",
    )
    records.append(low_volume_short)

    oil_market = market.loc[
        market["symbol"].isin({"BZUSDT", "CLUSDT"}),
    ].copy()
    oil_group = _group_state(oil_market, prefix="oil")
    oil_pool = oil_market.merge(oil_group, on="time", how="left")
    oil_start = pd.Timestamp(low_volume_long["signal_time"]) + pd.Timedelta(
        minutes=policy.oil_root_delay_minutes,
    )
    oil_root_mask = (
        oil_pool["time"].add(pd.Timedelta(minutes=5)).ge(oil_start)
        & oil_pool["oil_median_nr36"].le(
            policy.oil_root_maximum_median_nr36,
        )
        & oil_pool["nr12"].lt(0.0)
        & oil_pool["oil_negative3"].ge(0.5)
        & oil_pool["oil_median_flow12"].le(
            policy.oil_root_maximum_median_flow12,
        )
    )
    oil_short_row = _first_ranked(
        oil_pool.loc[oil_root_mask],
        rank_column="activity_ratio12",
        ascending=False,
    )
    oil_short = _market_record(
        oil_short_row,
        direction="SHORT",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_oil_pair_break",
    )
    records.append(oil_short)
    oil_close = _take_profit_close(
        market,
        oil_short,
        policy.oil_root_take_profit_pct,
    )
    oil_reentry_mask = (
        oil_pool["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(
            oil_close["signal_time"]
            + pd.Timedelta(hours=policy.oil_reentry_delay_hours),
        )
        & oil_pool["oil_median_nr12"].le(
            policy.oil_reentry_maximum_median_nr12,
        )
        & oil_pool["oil_median_nr36"].ge(
            policy.oil_reentry_minimum_median_nr36,
        )
        & oil_pool["oil_median_nr288"].le(
            policy.oil_reentry_maximum_median_nr288,
        )
        & oil_pool["nr12"].lt(0.0)
        & oil_pool["oil_median_flow12"].le(
            policy.oil_reentry_maximum_median_flow12,
        )
    )
    oil_long_row = _first_ranked(
        oil_pool.loc[oil_reentry_mask],
        rank_column="activity_ratio12",
        ascending=False,
    )
    oil_long = _market_record(
        oil_long_row,
        direction="LONG",
        operation_type="OPEN",
        role="ROOT",
        reason="dynamic_oil_pair_reentry",
    )
    records.append(oil_long)
    oil_hedge = _adverse_hedge(
        market,
        oil_long,
        minimum_minutes=policy.oil_long_hedge_minimum_minutes,
        adverse_pct=policy.oil_long_hedge_adverse_pct,
        direction="SHORT",
        reason="oil_long_adverse_hedge",
    )
    records.append(oil_hedge)

    tradfi_add = _tradfi_inventory_add(market, base, policy)
    records.append(tradfi_add)
    pow_add = _pow_hedge_add(market, pow_long_hedge, policy)
    records.append(pow_add)
    low_rehedge = _low_volume_rehedge(
        market,
        low_volume_long,
        low_volume_short,
        policy,
    )
    records.append(low_rehedge)
    return _normalize_candidates(pd.DataFrame.from_records(records))


def _subtype_market(market: pd.DataFrame, subtype: str) -> pd.DataFrame:
    return market.loc[
        market["subtypes"].map(lambda values: subtype in values),
    ].copy()


def _group_state(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    return (
        frame.groupby("time", sort=True)
        .agg(
            **{
                f"{prefix}_negative3": (
                    "nr3",
                    lambda value: value.lt(0.0).mean(),
                ),
                f"{prefix}_negative12": (
                    "nr12",
                    lambda value: value.lt(0.0).mean(),
                ),
                f"{prefix}_positive12": (
                    "nr12",
                    lambda value: value.gt(0.0).mean(),
                ),
                f"{prefix}_positive36": (
                    "nr36",
                    lambda value: value.gt(0.0).mean(),
                ),
                f"{prefix}_median_nr12": ("nr12", "median"),
                f"{prefix}_median_nr36": ("nr36", "median"),
                f"{prefix}_median_nr288": ("nr288", "median"),
                f"{prefix}_median_flow12": ("flow12", "median"),
            },
        )
        .reset_index()
    )


def _first_ranked(
    frame: pd.DataFrame,
    *,
    rank_column: str,
    ascending: bool,
) -> pd.Series:
    if frame.empty:
        raise RuntimeError(f"no candidate for ranking by {rank_column}")
    first_time = frame["time"].min()
    return (
        frame.loc[frame["time"].eq(first_time)]
        .sort_values(
            [rank_column, "symbol"],
            ascending=[ascending, True],
            kind="stable",
        )
        .iloc[0]
    )


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise RuntimeError("required public-market state transition not found")
    return frame.sort_values(["time", "symbol"], kind="stable").iloc[0]


def _market_record(
    row: pd.Series,
    *,
    direction: str,
    operation_type: str,
    role: str,
    reason: str,
) -> dict[str, object]:
    signal_time = pd.Timestamp(row["time"]) + pd.Timedelta(minutes=5)
    symbol = str(row["symbol"])
    return {
        "candidate_id": (
            f"{signal_time.value}|{symbol}|{direction}|{operation_type}|{role}"
        ),
        "signal_time": signal_time,
        "bar_time": pd.Timestamp(row["time"]),
        "symbol": symbol,
        "direction": direction,
        "operation_type": operation_type,
        "lifecycle_role": role,
        "signal_price": float(row["close"]),
        "reason": reason,
        "source_factor": "later_unified_public_market_state_v1",
    }


def _symbol_future(
    market: pd.DataFrame,
    symbol: str,
    start: object,
) -> pd.DataFrame:
    return market.loc[
        market["symbol"].eq(symbol)
        & market["time"]
        .add(pd.Timedelta(minutes=5))
        .ge(pd.Timestamp(start)),
    ].sort_values("time", kind="stable")


def _short_trailing_close(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    activation_pct: float,
    recovery_pct: float,
) -> dict[str, object]:
    future = _symbol_future(
        market,
        str(root["symbol"]),
        root["signal_time"],
    ).copy()
    entry = float(root["signal_price"])
    future["favorable_pct"] = (entry / future["close"] - 1.0) * 100.0
    future["best_favorable_pct"] = future["favorable_pct"].cummax()
    hits = future.loc[
        future["best_favorable_pct"].ge(activation_pct)
        & (
            future["best_favorable_pct"] - future["favorable_pct"]
        ).ge(recovery_pct)
    ]
    row = _first_row(hits)
    return {
        "signal_time": pd.Timestamp(row["time"]) + pd.Timedelta(minutes=5),
        "signal_price": float(row["close"]),
    }


def _take_profit_close(
    market: pd.DataFrame,
    root: dict[str, object],
    take_profit_pct: float,
) -> dict[str, object]:
    future = _symbol_future(
        market,
        str(root["symbol"]),
        root["signal_time"],
    ).copy()
    entry = float(root["signal_price"])
    if str(root["direction"]) == "SHORT":
        favorable = (entry / future["close"] - 1.0) * 100.0
    else:
        favorable = (future["close"] / entry - 1.0) * 100.0
    row = _first_row(future.loc[favorable.ge(take_profit_pct)])
    return {
        "signal_time": pd.Timestamp(row["time"]) + pd.Timedelta(minutes=5),
        "signal_price": float(row["close"]),
    }


def _adverse_hedge(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    minimum_minutes: int,
    adverse_pct: float,
    direction: str,
    reason: str,
) -> dict[str, object]:
    future = _symbol_future(
        market,
        str(root["symbol"]),
        pd.Timestamp(root["signal_time"])
        + pd.Timedelta(minutes=minimum_minutes),
    ).copy()
    entry = float(root["signal_price"])
    if str(root["direction"]) == "LONG":
        adverse = (entry / future["close"] - 1.0) * 100.0
    else:
        adverse = (future["close"] / entry - 1.0) * 100.0
    row = _first_row(future.loc[adverse.ge(adverse_pct)])
    return _market_record(
        row,
        direction=direction,
        operation_type="OPEN",
        role="HEDGE",
        reason=reason,
    )


def _tradfi_inventory_add(
    market: pd.DataFrame,
    base: pd.DataFrame,
    policy: LaterUnifiedPolicy,
) -> dict[str, object]:
    long_roots = base.loc[
        base["source_factor"].eq("sealed_later_tradfi_state_chain_v1r4")
        & base["direction"].eq("LONG")
        & base["lifecycle_role"].eq("ROOT")
    ].sort_values("signal_time")
    for root in long_roots.to_dict("records"):
        future = _symbol_future(
            market,
            str(root["symbol"]),
            root["signal_time"],
        ).copy()
        adverse = (
            float(root["signal_price"]) / future["close"] - 1.0
        ) * 100.0
        signal_time = future["time"] + pd.Timedelta(minutes=5)
        minute_of_day = signal_time.dt.hour * 60 + signal_time.dt.minute
        distance = np.minimum.reduce(
            [
                (minute_of_day - boundary).abs()
                for boundary in [0, 480, 960, 1440]
            ],
        )
        hits = future.loc[
            adverse.ge(policy.tradfi_long_add_adverse_pct)
            & adverse.le(policy.tradfi_long_add_maximum_adverse_pct)
            & pd.Series(distance, index=future.index).le(
                policy.tradfi_long_add_funding_window_minutes,
            )
        ]
        if not hits.empty:
            return _market_record(
                _first_row(hits),
                direction="LONG",
                operation_type="ADD",
                role="ADD",
                reason="tradfi_root_adverse_funding_recross_add",
            )
    raise RuntimeError("no generic TradFi ROOT-side ADD")


def _pow_hedge_add(
    market: pd.DataFrame,
    hedge: dict[str, object],
    policy: LaterUnifiedPolicy,
) -> dict[str, object]:
    future = _symbol_future(
        market,
        str(hedge["symbol"]),
        hedge["signal_time"],
    ).copy()
    adverse = (
        future["close"] / float(hedge["signal_price"]) - 1.0
    ) * 100.0
    row = _first_row(
        future.loc[
            adverse.ge(policy.pow_hedge_add_adverse_pct)
            & future["activity_ratio1"].ge(
                policy.pow_hedge_add_activity_ratio,
            )
        ],
    )
    return _market_record(
        row,
        direction="SHORT",
        operation_type="ADD",
        role="ADD",
        reason="pow_hedge_adverse_volume_expansion_add",
    )


def _low_volume_rehedge(
    market: pd.DataFrame,
    root: dict[str, object],
    hedge: dict[str, object],
    policy: LaterUnifiedPolicy,
) -> dict[str, object]:
    root_close = _take_profit_close(
        market,
        root,
        policy.low_volume_root_take_profit_pct,
    )
    future = _symbol_future(
        market,
        str(root["symbol"]),
        root_close["signal_time"]
        + pd.Timedelta(hours=policy.low_volume_rehedge_delay_hours),
    ).copy()
    adverse = (
        future["close"] / float(hedge["signal_price"]) - 1.0
    ) * 100.0
    row = _first_row(
        future.loc[adverse.ge(policy.low_volume_rehedge_adverse_pct)],
    )
    return _market_record(
        row,
        direction="LONG",
        operation_type="OPEN",
        role="HEDGE",
        reason="low_volume_short_rehedge_after_root_profit",
    )


def _normalize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["signal_time"] = pd.to_datetime(
        result["signal_time"],
        utc=True,
        errors="raise",
    )
    result["bar_time"] = pd.to_datetime(
        result["bar_time"],
        utc=True,
        errors="coerce",
    )
    result["signal_price"] = pd.to_numeric(
        result["signal_price"],
        errors="raise",
    )
    columns = [
        "candidate_id",
        "signal_time",
        "bar_time",
        "symbol",
        "direction",
        "operation_type",
        "lifecycle_role",
        "signal_price",
        "reason",
        "source_factor",
    ]
    if "lifecycle_profile" in result:
        columns.append("lifecycle_profile")
    return (
        result[columns]
        .sort_values(["signal_time", "symbol"], kind="stable")
        .drop_duplicates("candidate_id", keep="last")
        .reset_index(drop=True)
    )


def _normalize_base_signals(
    frame: pd.DataFrame,
    *,
    activation: pd.Timestamp,
    endpoint: pd.Timestamp,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "signal_time",
                "bar_time",
                "symbol",
                "direction",
                "operation_type",
                "lifecycle_role",
                "signal_price",
                "reason",
                "source_factor",
                "lifecycle_profile",
            ],
        )
    result = frame.copy()
    result["signal_time"] = pd.to_datetime(
        result["signal_time"],
        utc=True,
        errors="raise",
    )
    result = result.loc[
        result["signal_time"].between(activation, endpoint),
    ].copy()
    result["symbol"] = result["symbol"].map(_clean_symbol)
    if "direction" not in result:
        result["direction"] = (
            result["action"].astype(str).str.rsplit("_", n=1).str[-1]
        )
    if "operation_type" not in result:
        result["operation_type"] = "OPEN"
    if "lifecycle_role" not in result:
        result["lifecycle_role"] = np.where(
            result["operation_type"].astype(str).eq("ADD"),
            "ADD",
            "HEDGE",
        )
    if "bar_time" not in result:
        result["bar_time"] = result["signal_time"] - pd.Timedelta(
            minutes=5,
        )
    if "candidate_id" not in result:
        result["candidate_id"] = result.get(
            "signal_id",
            pd.Series(index=result.index, dtype=str),
        )
    if "reason" not in result:
        result["reason"] = result.get(
            "factor_family",
            pd.Series("autonomous_base_signal", index=result.index),
        )
    if "source_factor" not in result:
        result["source_factor"] = result.get(
            "policy_id",
            pd.Series(POLICY_ID, index=result.index),
        )
    if "lifecycle_profile" not in result:
        result["lifecycle_profile"] = result["reason"].map(
            _lifecycle_profile,
        )
    return _normalize_candidates(result)


def _to_live_signals(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return _empty_signals()
    output = _normalize_candidates(pd.DataFrame.from_records(records))
    output["action"] = "OPEN_" + output["direction"].astype(str)
    output["factor_family"] = output["reason"].astype(str)
    if "lifecycle_profile" not in output:
        output["lifecycle_profile"] = output["reason"].map(
            _lifecycle_profile,
        )
    else:
        output["lifecycle_profile"] = output["lifecycle_profile"].fillna(
            output["reason"].map(_lifecycle_profile),
        )
    expansion = output["reason"].eq(
        "hedged_slot_weak_price_taker_buy_absorption",
    )
    if expansion.any():
        sequence = output.loc[expansion].groupby(
            "reason",
            sort=False,
        ).cumcount()
        output.loc[expansion, "lifecycle_profile"] = sequence.map(
            {
                0: "TRADFI_EXPANSION_SHORT_1",
                1: "TRADFI_EXPANSION_SHORT_2",
            },
        ).fillna("TRADFI_EXPANSION_SHORT_2")
    output["policy_id"] = POLICY_ID
    output["market_only"] = True
    output["leader_direction_used"] = False
    output["signal_id"] = output["candidate_id"].astype(str)
    return output.reset_index(drop=True)


def _lifecycle_profile(reason: object) -> str:
    name = str(reason)
    profiles = {
        "initial_direct_equity_group_break": "TRADFI_ROOT_SHORT",
        "initial_correlated_follower_break": "LEGACY_TRADFI",
        "adverse_inventory_market_break": "TRADFI_HEDGE_SHORT",
        "rearmed_inventory_group_break": "TRADFI_REPEATED_HEDGE_SHORT",
        "hedged_slot_weak_price_taker_buy_absorption": "TRADFI_EXPANSION_SHORT_1",
        "aged_profitable_inventory_group_shock_recovery": "TRADFI_AGED_HEDGE_SHORT",
        "dynamic_pow_breadth_break": "POW_ROOT_SHORT",
        "dynamic_pow_post_trailing_reentry": "POW_ROOT_LONG",
        "dynamic_low_volume_layer1_rotation": "LOW_VOLUME_ROOT_LONG",
        "dynamic_oil_pair_break": "OIL_ROOT_SHORT",
        "oil_long_adverse_hedge": "OIL_HEDGE_SHORT",
        "pow_long_adverse_hedge": "POW_HEDGE_SHORT",
    }
    return profiles.get(name, "CURRENT_HEDGE" if "hedge" in name or "recovery" in name else "CURRENT")


def _maybe_first_ranked(
    frame: pd.DataFrame,
    *,
    rank_column: str,
    ascending: bool,
) -> pd.Series | None:
    if frame.empty:
        return None
    return _first_ranked(
        frame,
        rank_column=rank_column,
        ascending=ascending,
    )


def _maybe_first_row(frame: pd.DataFrame) -> pd.Series | None:
    return None if frame.empty else _first_row(frame)


def _optional_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except RuntimeError:
        return None


def _clean_symbol(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
    )


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "candidate_id",
            "signal_time",
            "bar_time",
            "symbol",
            "direction",
            "action",
            "operation_type",
            "lifecycle_role",
            "signal_price",
            "factor_family",
            "reason",
            "source_factor",
            "lifecycle_profile",
            "policy_id",
            "market_only",
            "leader_direction_used",
        ],
    )


__all__ = [
    "POLICY_ID",
    "LaterUnifiedPolicy",
    "build_later_sequential_market",
    "generate_later_sequential_signals",
]
