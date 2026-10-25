"""Exact later-regime TradFi state chain for CZ Live.

Only closed public market bars, exchange metadata and replica inventory enter
this module. Leader history remains an offline evaluator and is never accepted
by the production API.
"""

from __future__ import annotations

# The frozen factor is deliberately sequential and threshold-heavy.
# ruff: noqa: C901, EM101, PLR0912, PLR0915, TRY003
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

POLICY_ID = "weijie_non_eth_later_tradfi_state_chain_v1r4"
REGIME_START = pd.Timestamp("1970-01-01T00:00:00Z")
MARKET_END = pd.Timestamp("2262-01-01T00:00:00Z")

ETF_OR_ETP_BASE_ASSETS = frozenset({
    "DRAM", "EWJ", "EWT", "EWY", "EWZ", "IWM", "KORU", "KSTR",
    "MUU", "MVLL", "QQQ", "SHAZ", "SKHY", "SNXX", "SOXL", "SOXS",
    "SPCX", "SPY", "SQQQ", "STXX", "TQQQ", "TZA", "URNM", "UVXY",
    "XBI", "XLE",
})

@dataclass(frozen=True, slots=True)
class LaterTradFiStatePolicy:
    """Low-complexity inventory-conditioned market rules."""

    robust_window_minutes: int = 1440
    robust_minimum_minutes: int = 720
    correlation_window_minutes: int = 2880
    correlation_minimum_minutes: int = 1440
    minimum_anchor_correlation: float = 0.45
    maximum_expansion_correlation: float = 0.60
    initial_group_z: float = -1.50
    initial_negative_breadth: float = 0.95
    initial_return_strength: float = 2.00
    initial_follower_minutes: int = 60
    tactical_short_take_profit_pct: float = 2.75
    legacy_short_minimum_holding_hours: float = 36.0
    legacy_short_session_hour_utc: int = 13
    legacy_short_session_minute_utc: int = 30
    legacy_short_minimum_profit_pct: float = 0.50
    quick_campaign_hours: float = 8.0
    quick_campaign_reentry_cooldown_hours: float = 36.0
    swing_campaign_reentry_cooldown_hours: float = 8.0
    reentry_discount_pct: float = 3.0
    reentry_group_z: float = 1.0
    reentry_group_activity_z: float = 1.0
    reentry_maximum_negative_breadth: float = 0.20
    reentry_return_strength: float = 1.0
    quick_reentry_maximum_absolute_flow_z: float = 0.50
    swing_reentry_maximum_absolute_flow_z: float = 1.50
    volatility_window_minutes: int = 360
    volatility_horizon_minutes: int = 60
    minimum_hedge_scale_pct: float = 0.10
    maximum_hedge_scale_pct: float = 3.0
    initial_hedge_minimum_minutes: int = 30
    defensive_return_strength: float = 1.0
    defensive_flow_z: float = 1.5
    defensive_recent_group_z: float = 1.0
    defensive_recent_group_minutes: int = 180
    aged_profit_lock_hours: float = 36.0
    aged_profit_lock_recent_negative_group_z: float = 2.0
    aged_profit_lock_return_strength: float = 1.25
    aged_profit_lock_positive_flow_z: float = 2.0
    repeated_hedge_cooldown_hours: float = 36.0
    repeated_hedge_current_group_z: float = -2.0
    repeated_hedge_negative_flow_z: float = -2.0
    ordinary_short_hedge_minimum_hours: float = 8.0
    ordinary_short_hedge_positive_flow_z: float = 2.0
    fast_short_hedge_minimum_hours: float = 2.0
    fast_short_hedge_return_strength: float = 4.0
    fast_short_hedge_group_z: float = 3.0
    expansion_maximum_activity_rank: int = 10
    expansion_maximum_return_strength_rank: int = 10
    expansion_minimum_buy_share: float = 0.80
    expansion_minimum_net_flow_z: float = 10.0


def build_later_tradfi_features(
    raw_frames: dict[str, pd.DataFrame],
    contract_metadata: dict[str, dict[str, Any]],
    *,
    policy: LaterTradFiStatePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen one-minute TradFi panel from live OHLCV frames."""

    frames: list[pd.DataFrame] = []
    for raw_symbol, source in sorted(raw_frames.items()):
        symbol = _clean_symbol(raw_symbol)
        metadata = contract_metadata.get(symbol, {})
        if str(metadata.get("underlyingType", "")).upper() not in {
            "EQUITY",
            "KR_EQUITY",
            "HK_EQUITY",
        }:
            continue
        if source is None or source.empty:
            continue
        count_column = "trades" if "trades" in source else "count"
        taker_column = (
            "taker_buy_quote"
            if "taker_buy_quote" in source
            else "taker_buy_quote_volume"
        )
        required = {"close", "quote_volume", count_column, taker_column}
        if not required.issubset(source.columns):
            continue
        frame = source.loc[
            :,
            ["close", "quote_volume", count_column, taker_column],
        ].copy()
        frame = frame.rename(
            columns={
                count_column: "count",
                taker_column: "taker_buy_quote_volume",
            },
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
        frame["direct_equity"] = (
            str(metadata.get("baseAsset", "")).upper()
            not in ETF_OR_ETP_BASE_ASSETS
        )
        frames.append(frame.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    market = pd.concat(frames, ignore_index=True)
    for column in [
        "close",
        "quote_volume",
        "count",
        "taker_buy_quote_volume",
    ]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = (
        market.dropna(
            subset=[
                "time",
                "close",
                "quote_volume",
                "count",
                "taker_buy_quote_volume",
            ],
        )
        .loc[lambda value: value["close"].gt(0.0)]
        .sort_values(["symbol", "time"], kind="stable")
        .drop_duplicates(["symbol", "time"], keep="last")
        .reset_index(drop=True)
    )
    return _build_features(
        market,
        policy or LaterTradFiStatePolicy(),
    )


def generate_later_tradfi_signals(
    features: pd.DataFrame,
    group: pd.DataFrame,
    *,
    activation_time: object,
    decision_end: object,
    policy: LaterTradFiStatePolicy | None = None,
) -> pd.DataFrame:
    """Advance the frozen sequential chain as far as closed bars allow."""

    cfg = policy or LaterTradFiStatePolicy()
    if features.empty or group.empty:
        return _empty_signals()
    activation = _utc_timestamp(activation_time)
    endpoint = _utc_timestamp(decision_end)
    market = features.loc[
        features["time"].le(endpoint),
    ].sort_values(["time", "symbol"], kind="stable")
    records: list[dict[str, object]] = []
    used: set[str] = set()

    group_state = (
        group["median_ret60_rz"].le(cfg.initial_group_z)
        & group["negative_breadth"].ge(cfg.initial_negative_breadth)
    )
    onset = group_state & ~group_state.shift(1, fill_value=False)
    valid_onsets = group.index[
        onset
        & group.index.to_series().add(pd.Timedelta(minutes=1)).ge(
            activation,
        )
        & group.index.to_series().add(pd.Timedelta(minutes=1)).le(
            endpoint,
        )
    ]
    if len(valid_onsets) == 0:
        return _empty_signals()
    initial_bar = valid_onsets[0]
    primary_pool = market.loc[
        market["time"].eq(initial_bar)
        & market["eligible"]
        & market["direct_equity"]
        & market["ret60"].lt(0.0)
        & market["ret60_strength"].ge(cfg.initial_return_strength)
    ].sort_values(
        ["cluster_qrank", "cluster_trank", "symbol"],
        kind="stable",
    )
    if primary_pool.empty:
        return _empty_signals()
    primary = _record(
        primary_pool.iloc[0],
        role="ROOT",
        direction="SHORT",
        reason="initial_direct_equity_group_break",
    )
    records.append(primary)
    used.add(str(primary["symbol"]))

    follower_end = initial_bar + pd.Timedelta(
        minutes=cfg.initial_follower_minutes,
    )
    follower_pool = market.loc[
        market["time"].between(initial_bar, follower_end)
        & market["eligible"]
        & market["direct_equity"]
        & ~market["symbol"].isin(used)
        & market["ret60"].lt(0.0)
        & market["ret60_strength"].ge(cfg.initial_return_strength)
        & market["cluster_qrank"].le(2.0)
        & market["median_ret60_rz"].le(-1.0)
        & market["negative_breadth"].ge(0.90)
    ].sort_values(["time", "cluster_qrank", "symbol"], kind="stable")
    if follower_pool.empty:
        return _to_live_signals(records)
    follower = _record(
        follower_pool.iloc[0],
        role="ROOT",
        direction="SHORT",
        reason="initial_correlated_follower_break",
    )
    records.append(follower)
    used.add(str(follower["symbol"]))

    primary_close = _optional_call(
        _tactical_short_close,
        market,
        primary,
        policy=cfg,
    )
    follower_close = _optional_call(
        _legacy_short_close,
        market,
        follower,
        policy=cfg,
    )
    reentries: dict[str, dict[str, object]] = {}
    for root, close in (
        (primary, primary_close),
        (follower, follower_close),
    ):
        if close is None:
            continue
        reentry = _optional_call(
            _long_reentry,
            market,
            root,
            close,
            policy=cfg,
        )
        if reentry is None:
            continue
        records.append(reentry)
        reentries[str(root["symbol"])] = reentry

    hedges: dict[str, dict[str, object]] = {}
    for symbol, reentry in reentries.items():
        hedge = _first_long_hedge(market, reentry, policy=cfg)
        if hedge is None:
            continue
        records.append(hedge)
        hedges[symbol] = hedge

    primary_symbol = str(primary["symbol"])
    repeated: dict[str, object] | None = None
    if primary_symbol in hedges and primary_symbol in reentries:
        hedge_close = _optional_call(
            _tactical_short_close,
            market,
            hedges[primary_symbol],
            policy=cfg,
        )
        if hedge_close is not None:
            repeated = _optional_call(
                _repeated_long_hedge,
                market,
                reentries[primary_symbol],
                hedge_close,
                policy=cfg,
            )
    if repeated is not None:
        records.append(repeated)
        slot = {"opened_at": repeated["signal_time"], "expires_at": endpoint}
        first_expansion = _optional_call(
            _next_expansion,
            market,
            slot,
            used,
            policy=cfg,
        )
        if first_expansion is not None:
            records.append(first_expansion)
            used.add(str(first_expansion["symbol"]))
            first_long_hedge = _optional_call(
                _short_to_long_hedge,
                market,
                first_expansion,
                policy=cfg,
            )
            if first_long_hedge is not None:
                records.append(first_long_hedge)
                second_slot = {
                    "opened_at": first_long_hedge["signal_time"],
                    "expires_at": endpoint,
                }
                second_expansion = _optional_call(
                    _next_expansion,
                    market,
                    second_slot,
                    used,
                    policy=cfg,
                )
                if second_expansion is not None:
                    records.append(second_expansion)
                    second_long_hedge = _optional_call(
                        _short_to_long_hedge,
                        market,
                        second_expansion,
                        policy=cfg,
                    )
                    if second_long_hedge is not None:
                        records.append(second_long_hedge)
    return _to_live_signals(records)


def _build_features(
    market: pd.DataFrame,
    policy: LaterTradFiStatePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = market.copy()
    grouped = frame.groupby("symbol", sort=False, group_keys=False)
    frame["ret1"] = grouped["close"].pct_change(1, fill_method=None)
    frame["ret15"] = grouped["close"].pct_change(15, fill_method=None)
    frame["ret60"] = grouped["close"].pct_change(60, fill_method=None)
    frame["net_flow"] = (
        frame["taker_buy_quote_volume"] * 2.0 - frame["quote_volume"]
    )
    frame["flow5"] = grouped["net_flow"].transform(
        lambda value: value.rolling(5, min_periods=5).sum(),
    )
    frame["buy_share"] = frame["taker_buy_quote_volume"].div(
        frame["quote_volume"].replace(0.0, np.nan),
    )
    frame["net_flow_rz"] = grouped["net_flow"].transform(
        lambda value: _robust_z(
            value,
            policy.robust_window_minutes,
            policy.robust_minimum_minutes,
        ),
    )
    frame["flow5_rz"] = grouped["flow5"].transform(
        lambda value: _robust_z(
            value,
            policy.robust_window_minutes,
            policy.robust_minimum_minutes,
        ),
    )
    frame["ret60_strength"] = grouped["ret60"].transform(
        lambda value: value.abs().div(
            value.abs()
            .rolling(
                policy.robust_window_minutes,
                min_periods=policy.robust_minimum_minutes,
            )
            .median()
            .replace(0.0, np.nan),
        ),
    )

    returns = frame.pivot_table(
        index="time",
        columns="symbol",
        values="ret1",
        aggfunc="last",
    )
    if "SKHYNIXUSDT" not in returns:
        raise RuntimeError("SKHYNIXUSDT anchor is missing from archive")
    anchor = returns["SKHYNIXUSDT"]
    correlations = pd.DataFrame(index=returns.index)
    for symbol in returns:
        correlations[symbol] = returns[symbol].rolling(
            policy.correlation_window_minutes,
            min_periods=policy.correlation_minimum_minutes,
        ).corr(anchor)
    correlation_long = (
        correlations.rename_axis(index="time", columns="symbol")
        .reset_index()
        .melt(
            id_vars="time",
            var_name="symbol",
            value_name="corr",
        )
    )
    frame = frame.merge(
        correlation_long,
        on=["time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    frame["eligible"] = frame["corr"].ge(
        policy.minimum_anchor_correlation,
    ) | frame["symbol"].eq("SKHYNIXUSDT")

    eligible = frame.loc[
        frame["eligible"],
        [
            "time",
            "symbol",
            "quote_volume",
            "count",
            "ret60_strength",
        ],
    ].copy()
    eligible["cluster_qrank"] = eligible.groupby("time")[
        "quote_volume"
    ].rank(method="min", ascending=False)
    eligible["cluster_trank"] = eligible.groupby("time")["count"].rank(
        method="min",
        ascending=False,
    )
    eligible["ret60_strength_rank"] = eligible.groupby("time")[
        "ret60_strength"
    ].rank(method="min", ascending=False)
    frame = frame.merge(
        eligible[
            [
                "time",
                "symbol",
                "cluster_qrank",
                "cluster_trank",
                "ret60_strength_rank",
            ]
        ],
        on=["time", "symbol"],
        how="left",
        validate="one_to_one",
    )

    cluster = frame.loc[frame["eligible"]].groupby("time").agg(
        n=("symbol", "size"),
        median_ret60=("ret60", "median"),
        negative_breadth=("ret60", lambda value: value.lt(0.0).mean()),
        quote_volume=("quote_volume", "sum"),
    )
    cluster["median_ret60_rz"] = _robust_z(
        cluster["median_ret60"],
        policy.robust_window_minutes,
        policy.robust_minimum_minutes,
    )
    cluster["quote_volume_rz"] = _robust_z(
        cluster["quote_volume"],
        policy.robust_window_minutes,
        policy.robust_minimum_minutes,
    )
    frame = frame.merge(
        cluster[
            [
                "median_ret60",
                "negative_breadth",
                "median_ret60_rz",
                "quote_volume_rz",
            ]
        ],
        left_on="time",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    return frame, cluster


def _robust_z(
    value: pd.Series,
    window: int,
    minimum: int,
) -> pd.Series:
    median = value.rolling(window, min_periods=minimum).median()
    deviation = (value - median).abs()
    scale = (
        deviation.rolling(window, min_periods=minimum).median() * 1.4826
    )
    return (value - median).div(scale.replace(0.0, np.nan))


def _generate_candidates(
    features: pd.DataFrame,
    group: pd.DataFrame,
    policy: LaterTradFiStatePolicy,
) -> pd.DataFrame:
    market = features.loc[
        features["time"].le(MARKET_END),
    ].sort_values(["time", "symbol"], kind="stable")
    records: list[dict[str, object]] = []
    used: set[str] = set()

    group_state = (
        group["median_ret60_rz"].le(policy.initial_group_z)
        & group["negative_breadth"].ge(policy.initial_negative_breadth)
    )
    onset = group_state & ~group_state.shift(1, fill_value=False)
    valid_onsets = group.index[
        onset
        & group.index.to_series().add(pd.Timedelta(minutes=1)).ge(
            REGIME_START,
        )
    ]
    if len(valid_onsets) == 0:
        raise RuntimeError("later initial group pulse was not found")
    initial_bar = valid_onsets[0]
    primary_pool = market.loc[
        market["time"].eq(initial_bar)
        & market["eligible"]
        & market["direct_equity"]
        & market["ret60"].lt(0.0)
        & market["ret60_strength"].ge(policy.initial_return_strength)
    ].sort_values(
        ["cluster_qrank", "cluster_trank", "symbol"],
        kind="stable",
    )
    if primary_pool.empty:
        raise RuntimeError("later primary TradFi seed was not found")
    primary = primary_pool.iloc[0]
    primary_record = _record(
        primary,
        role="ROOT",
        direction="SHORT",
        reason="initial_direct_equity_group_break",
    )
    records.append(primary_record)
    used.add(str(primary.symbol))

    follower_end = initial_bar + pd.Timedelta(
        minutes=policy.initial_follower_minutes,
    )
    follower_pool = market.loc[
        market["time"].between(initial_bar, follower_end)
        & market["eligible"]
        & market["direct_equity"]
        & ~market["symbol"].isin(used)
        & market["ret60"].lt(0.0)
        & market["ret60_strength"].ge(policy.initial_return_strength)
        & market["cluster_qrank"].le(2.0)
        & market["median_ret60_rz"].le(-1.0)
        & market["negative_breadth"].ge(0.90)
    ].sort_values(["time", "cluster_qrank", "symbol"], kind="stable")
    if follower_pool.empty:
        raise RuntimeError("later follower TradFi seed was not found")
    follower = follower_pool.iloc[0]
    follower_record = _record(
        follower,
        role="ROOT",
        direction="SHORT",
        reason="initial_correlated_follower_break",
    )
    records.append(follower_record)
    used.add(str(follower.symbol))

    primary_close = _tactical_short_close(
        market,
        primary_record,
        policy=policy,
    )
    follower_close = _legacy_short_close(
        market,
        follower_record,
        policy=policy,
    )
    reentries: list[dict[str, object]] = []
    for root, close in [
        (primary_record, primary_close),
        (follower_record, follower_close),
    ]:
        reentry = _long_reentry(
            market,
            root,
            close,
            policy=policy,
        )
        records.append(reentry)
        reentries.append(reentry)

    hedge_records: list[dict[str, object]] = []
    for reentry in reentries:
        hedge = _first_long_hedge(
            market,
            reentry,
            policy=policy,
        )
        if hedge is None:
            message = f"no first long hedge for {reentry['symbol']}"
            raise RuntimeError(message)
        records.append(hedge)
        hedge_records.append(hedge)

    primary_hedge = next(
        row
        for row in hedge_records
        if row["symbol"] == primary_record["symbol"]
    )
    primary_hedge_close = _tactical_short_close(
        market,
        primary_hedge,
        policy=policy,
    )
    repeated = _repeated_long_hedge(
        market,
        next(
            row
            for row in reentries
            if row["symbol"] == primary_record["symbol"]
        ),
        primary_hedge_close,
        policy=policy,
    )
    records.append(repeated)
    hedge_records.append(repeated)

    slots = [
        {
            "opened_at": pd.Timestamp(repeated["signal_time"]),
            "expires_at": MARKET_END,
        },
    ]
    first_expansion = _next_expansion(
        market,
        slots[0],
        used,
        policy=policy,
    )
    records.append(first_expansion)
    used.add(str(first_expansion["symbol"]))
    first_short_hedge = _short_to_long_hedge(
        market,
        first_expansion,
        policy=policy,
    )
    records.append(first_short_hedge)
    hedge_records.append(first_short_hedge)

    second_slot = {
        "opened_at": pd.Timestamp(first_short_hedge["signal_time"]),
        "expires_at": MARKET_END,
    }
    second_expansion = _next_expansion(
        market,
        second_slot,
        used,
        policy=policy,
    )
    records.append(second_expansion)
    used.add(str(second_expansion["symbol"]))
    second_short_hedge = _short_to_long_hedge(
        market,
        second_expansion,
        policy=policy,
    )
    records.append(second_short_hedge)

    return (
        pd.DataFrame.from_records(records)
        .sort_values(["signal_time", "symbol"], kind="stable")
        .reset_index(drop=True)
    )


def _record(
    row: pd.Series,
    *,
    role: str,
    direction: str,
    reason: str,
) -> dict[str, object]:
    return {
        "candidate_id": (
            f"{pd.Timestamp(row.time).value}|{row.symbol}|{direction}|{role}"
        ),
        "signal_time": pd.Timestamp(row.time) + pd.Timedelta(minutes=1),
        "bar_time": pd.Timestamp(row.time),
        "symbol": str(row.symbol),
        "direction": direction,
        "operation_type": "OPEN",
        "lifecycle_role": role,
        "signal_price": float(row.close),
        "reason": reason,
        "ret15": float(row.ret15),
        "ret60": float(row.ret60),
        "ret60_strength": float(row.ret60_strength),
        "corr": float(row["corr"]),
        "cluster_qrank": float(row.cluster_qrank),
        "cluster_trank": float(row.cluster_trank),
        "net_flow_rz": float(row.net_flow_rz),
        "flow5_rz": float(row.flow5_rz),
        "median_ret60_rz": float(row.median_ret60_rz),
        "negative_breadth": float(row.negative_breadth),
        "quote_volume_rz": float(row.quote_volume_rz),
    }


def _symbol_after(
    market: pd.DataFrame,
    symbol: str,
    timestamp: object,
) -> pd.DataFrame:
    return market.loc[
        market["symbol"].eq(symbol)
        & market["time"].add(pd.Timedelta(minutes=1)).ge(
            pd.Timestamp(timestamp),
        )
    ].sort_values("time", kind="stable")


def _tactical_short_close(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    future = _symbol_after(market, str(root["symbol"]), root["signal_time"])
    target = float(root["signal_price"]) * (
        1.0 - policy.tactical_short_take_profit_pct / 100.0
    )
    hit = future.loc[future["close"].le(target)]
    if hit.empty:
        message = f"no tactical close for {root['candidate_id']}"
        raise RuntimeError(message)
    row = hit.iloc[0]
    return {
        "signal_time": pd.Timestamp(row.time) + pd.Timedelta(minutes=1),
        "signal_price": float(row.close),
    }


def _legacy_short_close(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    future = _symbol_after(market, str(root["symbol"]), root["signal_time"])
    eligible = future.loc[
        future["time"]
        .add(pd.Timedelta(minutes=1))
        .ge(
            pd.Timestamp(root["signal_time"])
            + pd.Timedelta(
                hours=policy.legacy_short_minimum_holding_hours,
            ),
        )
        & future["time"].dt.hour.eq(
            policy.legacy_short_session_hour_utc,
        )
        & future["time"].dt.minute.eq(
            policy.legacy_short_session_minute_utc - 1,
        )
        & future["close"].le(
            float(root["signal_price"])
            * (1.0 - policy.legacy_short_minimum_profit_pct / 100.0),
        )
    ]
    if eligible.empty:
        message = f"no legacy close for {root['candidate_id']}"
        raise RuntimeError(message)
    row = eligible.iloc[0]
    return {
        "signal_time": pd.Timestamp(row.time) + pd.Timedelta(minutes=1),
        "signal_price": float(row.close),
    }


def _long_reentry(
    market: pd.DataFrame,
    root: dict[str, object],
    close: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    duration_hours = (
        pd.Timestamp(close["signal_time"])
        - pd.Timestamp(root["signal_time"])
    ).total_seconds() / 3600.0
    cooldown = (
        policy.quick_campaign_reentry_cooldown_hours
        if duration_hours < policy.quick_campaign_hours
        else policy.swing_campaign_reentry_cooldown_hours
    )
    maximum_absolute_flow_z = (
        policy.quick_reentry_maximum_absolute_flow_z
        if duration_hours < policy.quick_campaign_hours
        else policy.swing_reentry_maximum_absolute_flow_z
    )
    eligible_time = pd.Timestamp(close["signal_time"]) + pd.Timedelta(
        hours=cooldown,
    )
    future = _symbol_after(market, str(root["symbol"]), eligible_time)
    state = (
        future["close"].le(
            float(close["signal_price"])
            * (1.0 - policy.reentry_discount_pct / 100.0),
        )
        & future["ret60"].gt(0.0)
        & future["ret60_strength"].ge(policy.reentry_return_strength)
        & future["median_ret60_rz"].ge(policy.reentry_group_z)
        & future["negative_breadth"].le(
            policy.reentry_maximum_negative_breadth,
        )
        & future["quote_volume_rz"].ge(
            policy.reentry_group_activity_z,
        )
        & future["flow5_rz"].abs().le(maximum_absolute_flow_z)
        & future["net_flow_rz"].abs().le(maximum_absolute_flow_z)
    )
    transition = state & ~state.shift(1, fill_value=False)
    hits = future.loc[transition]
    if hits.empty:
        message = f"no long reentry for {root['candidate_id']}"
        raise RuntimeError(message)
    return _record(
        hits.iloc[0],
        role="ROOT",
        direction="LONG",
        reason="discounted_post_campaign_group_recovery",
    )


def _entry_scale_pct(
    symbol_market: pd.DataFrame,
    entry_time: object,
    policy: LaterTradFiStatePolicy,
) -> float:
    history = symbol_market.loc[
        symbol_market["time"]
        .add(pd.Timedelta(minutes=1))
        .lt(pd.Timestamp(entry_time)),
        "close",
    ].tail(policy.volatility_window_minutes + 1)
    log_return = np.log(history / history.shift(1)).dropna()
    raw = (
        float(log_return.std(ddof=0))
        * math.sqrt(policy.volatility_horizon_minutes)
        * 100.0
    )
    return float(
        np.clip(
            raw,
            policy.minimum_hedge_scale_pct,
            policy.maximum_hedge_scale_pct,
        ),
    )


def _first_long_hedge(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object] | None:
    symbol_market = market.loc[market["symbol"].eq(root["symbol"])].copy()
    future = _symbol_after(
        market,
        str(root["symbol"]),
        pd.Timestamp(root["signal_time"])
        + pd.Timedelta(minutes=policy.initial_hedge_minimum_minutes),
    ).copy()
    scale = _entry_scale_pct(
        symbol_market,
        root["signal_time"],
        policy,
    )
    future["adverse_pct"] = (
        float(root["signal_price"]) / future["close"] - 1.0
    ) * 100.0
    future["recent_abs_group_z"] = (
        future["median_ret60_rz"]
        .abs()
        .rolling(
            policy.defensive_recent_group_minutes + 1,
            min_periods=1,
        )
        .max()
    )
    defensive = (
        future["adverse_pct"].ge(scale)
        & future["ret60"].lt(0.0)
        & future["ret60_strength"].ge(
            policy.defensive_return_strength,
        )
        & future["flow5_rz"].le(-policy.defensive_flow_z)
        & future["recent_abs_group_z"].ge(
            policy.defensive_recent_group_z,
        )
    )
    future["recent_negative_group_z"] = (
        -future["median_ret60_rz"]
    ).rolling(
        policy.defensive_recent_group_minutes + 1,
        min_periods=1,
    ).max()
    age_hours = (
        future["time"].add(pd.Timedelta(minutes=1))
        - pd.Timestamp(root["signal_time"])
    ).dt.total_seconds() / 3600.0
    profit_pct = (
        future["close"] / float(root["signal_price"]) - 1.0
    ) * 100.0
    aged_profit_lock = (
        age_hours.ge(policy.aged_profit_lock_hours)
        & profit_pct.gt(0.0)
        & future["ret60"].lt(0.0)
        & future["ret60_strength"].ge(
            policy.aged_profit_lock_return_strength,
        )
        & future["recent_negative_group_z"].ge(
            policy.aged_profit_lock_recent_negative_group_z,
        )
        & future["flow5_rz"].ge(
            policy.aged_profit_lock_positive_flow_z,
        )
        & future["median_ret60_rz"].ge(0.0)
    )
    state = defensive | aged_profit_lock
    hits = future.loc[state & ~state.shift(1, fill_value=False)]
    if hits.empty:
        return None
    row = hits.iloc[0]
    reason = (
        "aged_profitable_inventory_group_shock_recovery"
        if bool(aged_profit_lock.loc[row.name])
        else "adverse_inventory_market_break"
    )
    return _record(
        row,
        role="HEDGE",
        direction="SHORT",
        reason=reason,
    )


def _repeated_long_hedge(
    market: pd.DataFrame,
    root: dict[str, object],
    prior_hedge_close: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    symbol_market = market.loc[market["symbol"].eq(root["symbol"])].copy()
    scale = _entry_scale_pct(
        symbol_market,
        root["signal_time"],
        policy,
    )
    start = pd.Timestamp(prior_hedge_close["signal_time"]) + pd.Timedelta(
        hours=policy.repeated_hedge_cooldown_hours,
    )
    future = _symbol_after(
        market,
        str(root["symbol"]),
        start,
    ).copy()
    future["adverse_pct"] = (
        float(root["signal_price"]) / future["close"] - 1.0
    ) * 100.0
    state = (
        future["adverse_pct"].ge(scale)
        & future["ret60"].lt(0.0)
        & future["ret60_strength"].ge(
            policy.defensive_return_strength,
        )
        & future["median_ret60_rz"].le(
            policy.repeated_hedge_current_group_z,
        )
        & future["flow5_rz"].le(
            policy.repeated_hedge_negative_flow_z,
        )
    )
    hits = future.loc[state & ~state.shift(1, fill_value=False)]
    if hits.empty:
        raise RuntimeError("no repeated TradFi long hedge")
    return _record(
        hits.iloc[0],
        role="HEDGE",
        direction="SHORT",
        reason="rearmed_inventory_group_break",
    )


def _next_expansion(
    market: pd.DataFrame,
    slot: dict[str, object],
    used: set[str],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    eligible = market.loc[
        market["time"]
        .add(pd.Timedelta(minutes=1))
        .between(
            pd.Timestamp(slot["opened_at"]),
            pd.Timestamp(slot["expires_at"]),
        )
        & market["eligible"]
        & market["direct_equity"]
        & ~market["symbol"].isin(used)
        & market["corr"].lt(policy.maximum_expansion_correlation)
        & market["ret60"].lt(0.0)
        & market["cluster_qrank"].le(
            policy.expansion_maximum_activity_rank,
        )
        & market["ret60_strength_rank"].le(
            policy.expansion_maximum_return_strength_rank,
        )
        & market["buy_share"].ge(policy.expansion_minimum_buy_share)
        & market["net_flow_rz"].ge(
            policy.expansion_minimum_net_flow_z,
        )
    ].sort_values(["time", "cluster_qrank", "symbol"], kind="stable")
    if eligible.empty:
        raise RuntimeError("hedged inventory slot found no expansion")
    return _record(
        eligible.iloc[0],
        role="ROOT",
        direction="SHORT",
        reason="hedged_slot_weak_price_taker_buy_absorption",
    )


def _short_to_long_hedge(
    market: pd.DataFrame,
    root: dict[str, object],
    *,
    policy: LaterTradFiStatePolicy,
) -> dict[str, object]:
    symbol_market = market.loc[market["symbol"].eq(root["symbol"])].copy()
    scale = _entry_scale_pct(
        symbol_market,
        root["signal_time"],
        policy,
    )
    future = _symbol_after(
        market,
        str(root["symbol"]),
        root["signal_time"],
    ).copy()
    age_hours = (
        future["time"].add(pd.Timedelta(minutes=1))
        - pd.Timestamp(root["signal_time"])
    ).dt.total_seconds() / 3600.0
    adverse_pct = (
        future["close"] / float(root["signal_price"]) - 1.0
    ) * 100.0
    ordinary = (
        age_hours.ge(policy.ordinary_short_hedge_minimum_hours)
        & adverse_pct.ge(scale)
        & future["ret60"].gt(0.0)
        & future["ret60_strength"].ge(
            policy.defensive_return_strength,
        )
        & future["flow5_rz"].ge(
            policy.ordinary_short_hedge_positive_flow_z,
        )
    )
    fast = (
        age_hours.ge(policy.fast_short_hedge_minimum_hours)
        & adverse_pct.ge(scale)
        & future["ret60"].gt(0.0)
        & future["ret60_strength"].ge(
            policy.fast_short_hedge_return_strength,
        )
        & future["median_ret60_rz"].ge(
            policy.fast_short_hedge_group_z,
        )
        & future["flow5_rz"].ge(policy.defensive_flow_z)
    )
    state = ordinary | fast
    hits = future.loc[state & ~state.shift(1, fill_value=False)]
    if hits.empty:
        message = f"no short-to-long hedge for {root['candidate_id']}"
        raise RuntimeError(message)
    row = hits.iloc[0]
    reason = (
        "fast_extreme_group_recovery"
        if bool(fast.loc[row.name])
        else "mature_short_inventory_recovery"
    )
    return _record(
        row,
        role="HEDGE",
        direction="LONG",
        reason=reason,
    )


def _to_live_signals(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return _empty_signals()
    output = pd.DataFrame.from_records(records).sort_values(
        ["signal_time", "symbol"],
        kind="stable",
    )
    output["action"] = "OPEN_" + output["direction"].astype(str)
    output["factor_family"] = output["reason"].astype(str)
    output["lifecycle_profile"] = np.where(
        output["lifecycle_role"].eq("HEDGE"),
        "TRADFI_HEDGE",
        "TRADFI_ROOT",
    )
    output["policy_id"] = POLICY_ID
    output["market_only"] = True
    output["leader_direction_used"] = False
    output["source_factor"] = POLICY_ID
    output["signal_id"] = output["candidate_id"].astype(str)
    return output.reset_index(drop=True)


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
            "lifecycle_profile",
            "policy_id",
            "market_only",
            "leader_direction_used",
            "source_factor",
        ],
    )


__all__ = [
    "POLICY_ID",
    "LaterTradFiStatePolicy",
    "build_later_tradfi_features",
    "generate_later_tradfi_signals",
]
