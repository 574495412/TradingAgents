"""非 ETH 跨品种事件开仓因子。

本模块只接收决策时已经闭合的市场特征和交易所元数据，不接收带单员事件、
持仓、收益率或标的白名单。它把单币连续触发收敛为可解释的板块事件，并在
每个事件中只选择少量流动性合格的候选。

当前版本仅用于同窗描述性研究，不能据此声称已经完成前向验证或可以上线。
"""

# ruff: noqa: C901, EM101, PERF401, RUF001, RUF002, TRY003

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.common import clean_symbol

POLICY_ID = "weijie_non_eth_group_event_open_research_v1"
ADD_POLICY_ID = "weijie_non_eth_group_event_add_research_v1"

REQUIRED_COLUMNS = frozenset(
    {
        "bar_time",
        "symbol",
        "close",
        "rolling_quote_volume_24h",
        "contract_type",
        "subtype",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_144",
        "normalized_return_288",
        "quote_volume_ratio",
        "signed_seller_pressure",
        "range_location_288",
        "eligible",
    },
)


@dataclass(frozen=True)
class NonEthGroupEventFactorConfig:
    """可复用的市场结构阈值，不包含历史事件身份。"""

    minimum_tradfi_group_size: int = 8
    tradfi_negative_3_breadth: float = 0.80
    tradfi_negative_12_breadth: float = 0.90
    tradfi_minimum_mean_return_144: float = 2.0
    tradfi_minimum_mean_return_288: float = 6.0
    tradfi_candidate_minimum_return_144: float = 4.0
    tradfi_event_capacity: int = 2

    minimum_layer1_group_size: int = 5
    layer1_short_minimum_mean_return_12: float = 4.0
    layer1_short_minimum_mean_return_144: float = 3.0
    layer1_short_maximum_mean_return_36: float = 4.0
    layer1_short_minimum_high_volume_breadth: float = 0.40
    layer1_short_minimum_positive_3_breadth: float = 0.70
    layer1_short_candidate_minimum_return_12: float = 2.0
    layer1_short_candidate_minimum_return_144: float = 3.0

    minimum_payment_group_size: int = 3
    payment_minimum_mean_return_36: float = 2.0
    payment_minimum_mean_return_144: float = 6.0
    payment_minimum_mean_return_12: float = 0.0
    payment_minimum_negative_3_breadth: float = 0.66
    payment_candidate_minimum_return_36: float = 3.0
    payment_candidate_minimum_return_144: float = 6.0
    payment_candidate_maximum_return_3: float = -1.5
    payment_candidate_minimum_quote_ratio: float = 0.8

    idiosyncratic_minimum_return_288: float = 12.0
    idiosyncratic_maximum_return_288: float = 30.0
    idiosyncratic_minimum_return_144: float = 4.0
    idiosyncratic_maximum_return_144: float = 12.0
    idiosyncratic_minimum_return_36: float = 4.0
    idiosyncratic_maximum_return_12: float = -1.0
    idiosyncratic_maximum_return_3: float = -1.0
    idiosyncratic_minimum_quote_ratio: float = 0.8
    idiosyncratic_minimum_seller_pressure: float = 0.1
    idiosyncratic_minimum_range_location: float = 0.75
    idiosyncratic_market_minimum_mean_return_12: float = -1.0
    idiosyncratic_market_maximum_mean_return_12: float = 1.0
    idiosyncratic_market_maximum_mean_return_36: float = 0.0
    idiosyncratic_rearm_hours: float = 24.0

    layer1_long_maximum_mean_return_288: float = -12.0
    layer1_long_minimum_mean_return_36: float = 2.0
    layer1_long_minimum_negative_288_breadth: float = 0.50
    layer1_long_minimum_positive_36_breadth: float = 0.40
    layer1_long_minimum_high_volume_breadth: float = 0.20
    layer1_long_minimum_seller_breadth: float = 0.20
    layer1_long_candidate_maximum_return_288: float = -8.0
    layer1_long_candidate_minimum_return_36: float = 2.0
    layer1_long_candidate_maximum_abs_return_3: float = 0.75
    layer1_long_candidate_minimum_quote_ratio: float = 1.5
    layer1_long_candidate_minimum_seller_pressure: float = 0.15
    layer1_long_candidate_maximum_range_location: float = 0.35


@dataclass(frozen=True)
class NonEthGroupEventAddConfig:
    """只在策略已有同向库存时允许执行的补仓候选阈值。"""

    minimum_payment_group_size: int = 3
    payment_maximum_mean_return_36: float = -1.5
    payment_minimum_candidate_return_3: float = 0.5
    payment_maximum_candidate_return_3: float = 1.25
    payment_maximum_candidate_return_12: float = -1.0
    payment_maximum_candidate_return_36: float = -2.0
    payment_maximum_candidate_return_144: float = -3.0
    payment_maximum_candidate_return_288: float = -4.0
    payment_minimum_range_location: float = 0.18
    payment_maximum_range_location: float = 0.25
    payment_rearm_hours: float = 1.0

    continuation_minimum_return_288: float = 2.0
    continuation_maximum_return_288: float = 10.0
    continuation_minimum_return_144: float = 0.0
    continuation_maximum_return_144: float = 5.0
    continuation_minimum_return_36: float = 2.0
    continuation_minimum_return_12: float = 2.0
    continuation_minimum_return_3: float = 1.0
    continuation_maximum_return_3: float = 2.0
    continuation_minimum_quote_ratio: float = 3.0
    continuation_maximum_seller_pressure: float = -0.2
    continuation_minimum_range_location: float = 0.4
    continuation_maximum_range_location: float = 0.75
    continuation_market_minimum_mean_return_36: float = -1.0
    continuation_market_maximum_mean_return_36: float = 2.0

    minimum_layer1_group_size: int = 5
    layer1_maximum_mean_return_288: float = -12.0
    layer1_minimum_mean_return_144: float = -5.0
    layer1_maximum_mean_return_144: float = 0.0
    layer1_minimum_mean_return_36: float = -1.5
    layer1_maximum_mean_return_12: float = -2.0
    layer1_minimum_negative_3_breadth: float = 0.8
    layer1_minimum_seller_breadth: float = 0.3
    layer1_candidate_maximum_return_288: float = -12.0
    layer1_candidate_maximum_return_12: float = -1.5
    layer1_candidate_maximum_return_3: float = -0.5
    layer1_candidate_minimum_seller_pressure: float = 0.15
    layer1_candidate_maximum_range_location: float = 0.25


def generate_non_eth_group_event_open_signals(
    feature_rows: pd.DataFrame,
    *,
    config: NonEthGroupEventFactorConfig | None = None,
) -> pd.DataFrame:
    """从闭合市场特征生成稀疏的非 ETH 初始开仓候选。"""

    cfg = config or NonEthGroupEventFactorConfig()
    market = _normalize_feature_rows(feature_rows)
    if market.empty:
        return _empty_signals()

    coin = _aggregate_group(
        market[market["contract_type"].eq("PERPETUAL")],
    )
    payment = _aggregate_group(
        market[market["subtype"].eq("Payment")],
    )
    tradfi = _aggregate_group(
        market[market["contract_type"].eq("TRADIFI_PERPETUAL")],
    )
    layer1 = _aggregate_group(
        market[market["subtype"].eq("Layer-1")],
    )
    market = market.merge(
        coin.add_prefix("coin_").rename(columns={"coin_bar_time": "bar_time"}),
        on="bar_time",
        how="left",
    )
    market = market.merge(
        payment.add_prefix("payment_").rename(
            columns={"payment_bar_time": "bar_time"},
        ),
        on="bar_time",
        how="left",
    )

    records: list[dict[str, object]] = []
    tradfi_events = _group_onsets(
        tradfi,
        tradfi["group_size"].ge(cfg.minimum_tradfi_group_size)
        & tradfi["negative_3_breadth"].ge(
            cfg.tradfi_negative_3_breadth,
        )
        & tradfi["negative_12_breadth"].ge(
            cfg.tradfi_negative_12_breadth,
        )
        & tradfi["mean_return_144"].ge(
            cfg.tradfi_minimum_mean_return_144,
        )
        & tradfi["mean_return_288"].ge(
            cfg.tradfi_minimum_mean_return_288,
        ),
    )
    for event_time in tradfi_events["bar_time"]:
        candidates = market[
            market["bar_time"].eq(event_time)
            & market["contract_type"].eq("TRADIFI_PERPETUAL")
            & market["normalized_return_144"].ge(
                cfg.tradfi_candidate_minimum_return_144,
            )
            & market["normalized_return_3"].lt(0.0)
        ]
        for row in _top_liquidity(
            candidates,
            capacity=cfg.tradfi_event_capacity,
        ):
            records.append(
                _signal_record(
                    row,
                    action="OPEN_SHORT",
                    family="tradfi_extended_breadth_break",
                    score=(
                        float(row["normalized_return_144"])
                        - float(row["normalized_return_3"])
                    ),
                    reason=(
                        "TradFi 多日上涨扩张后出现板块广度下破，"
                        "选择高流动性且仍有趋势延伸的标的"
                    ),
                ),
            )

    layer1_short_events = _group_onsets(
        layer1,
        layer1["group_size"].ge(cfg.minimum_layer1_group_size)
        & layer1["mean_return_12"].ge(
            cfg.layer1_short_minimum_mean_return_12,
        )
        & layer1["mean_return_144"].ge(
            cfg.layer1_short_minimum_mean_return_144,
        )
        & layer1["mean_return_36"].le(
            cfg.layer1_short_maximum_mean_return_36,
        )
        & layer1["high_volume_breadth"].ge(
            cfg.layer1_short_minimum_high_volume_breadth,
        )
        & layer1["positive_3_breadth"].ge(
            cfg.layer1_short_minimum_positive_3_breadth,
        ),
    )
    for event_time in layer1_short_events["bar_time"]:
        candidates = market[
            market["bar_time"].eq(event_time)
            & market["subtype"].eq("Layer-1")
            & market["normalized_return_12"].ge(
                cfg.layer1_short_candidate_minimum_return_12,
            )
            & market["normalized_return_144"].ge(
                cfg.layer1_short_candidate_minimum_return_144,
            )
        ]
        for row in _top_liquidity(candidates, capacity=1):
            records.append(
                _signal_record(
                    row,
                    action="OPEN_SHORT",
                    family="layer1_squeeze_exhaustion",
                    score=float(row["normalized_return_12"]),
                    reason="Layer-1 板块放量冲高但中周期动量开始衰减",
                ),
            )

    payment_raw = (
        market["subtype"].eq("Payment")
        & market["payment_group_size"].ge(
            cfg.minimum_payment_group_size,
        )
        & market["payment_mean_return_36"].ge(
            cfg.payment_minimum_mean_return_36,
        )
        & market["payment_mean_return_144"].ge(
            cfg.payment_minimum_mean_return_144,
        )
        & market["payment_mean_return_12"].ge(
            cfg.payment_minimum_mean_return_12,
        )
        & market["payment_negative_3_breadth"].ge(
            cfg.payment_minimum_negative_3_breadth,
        )
        & market["normalized_return_36"].ge(
            cfg.payment_candidate_minimum_return_36,
        )
        & market["normalized_return_144"].ge(
            cfg.payment_candidate_minimum_return_144,
        )
        & market["normalized_return_3"].le(
            cfg.payment_candidate_maximum_return_3,
        )
        & market["quote_volume_ratio"].ge(
            cfg.payment_candidate_minimum_quote_ratio,
        )
    )
    for row in _symbol_onsets(market, payment_raw).to_dict(
        orient="records",
    ):
        records.append(
            _signal_record(
                row,
                action="OPEN_SHORT",
                family="payment_trend_rollover",
                score=(
                    float(row["normalized_return_144"])
                    - float(row["normalized_return_3"])
                ),
                reason="支付币板块长趋势延伸后的短周期同步回落",
            ),
        )

    idiosyncratic_raw = (
        market["contract_type"].eq("PERPETUAL")
        & market["normalized_return_288"].between(
            cfg.idiosyncratic_minimum_return_288,
            cfg.idiosyncratic_maximum_return_288,
        )
        & market["normalized_return_144"].between(
            cfg.idiosyncratic_minimum_return_144,
            cfg.idiosyncratic_maximum_return_144,
        )
        & market["normalized_return_36"].ge(
            cfg.idiosyncratic_minimum_return_36,
        )
        & market["normalized_return_12"].le(
            cfg.idiosyncratic_maximum_return_12,
        )
        & market["normalized_return_3"].le(
            cfg.idiosyncratic_maximum_return_3,
        )
        & market["quote_volume_ratio"].ge(
            cfg.idiosyncratic_minimum_quote_ratio,
        )
        & market["signed_seller_pressure"].ge(
            cfg.idiosyncratic_minimum_seller_pressure,
        )
        & market["range_location_288"].ge(
            cfg.idiosyncratic_minimum_range_location,
        )
        & market["coin_mean_return_12"].between(
            cfg.idiosyncratic_market_minimum_mean_return_12,
            cfg.idiosyncratic_market_maximum_mean_return_12,
        )
        & market["coin_mean_return_36"].le(
            cfg.idiosyncratic_market_maximum_mean_return_36,
        )
    )
    idiosyncratic_onsets = _apply_symbol_rearm(
        _symbol_onsets(market, idiosyncratic_raw),
        hours=cfg.idiosyncratic_rearm_hours,
    )
    for row in idiosyncratic_onsets.to_dict(orient="records"):
        records.append(
            _signal_record(
                row,
                action="OPEN_SHORT",
                family="extended_idiosyncratic_rollover",
                score=(
                    float(row["normalized_return_288"])
                    - float(row["normalized_return_12"])
                ),
                reason="相对市场独立上涨后的量价回落与主动卖压确认",
            ),
        )

    layer1_long_events = _group_onsets(
        layer1,
        layer1["group_size"].ge(cfg.minimum_layer1_group_size)
        & layer1["mean_return_288"].le(
            cfg.layer1_long_maximum_mean_return_288,
        )
        & layer1["mean_return_36"].ge(
            cfg.layer1_long_minimum_mean_return_36,
        )
        & layer1["negative_288_breadth"].ge(
            cfg.layer1_long_minimum_negative_288_breadth,
        )
        & layer1["positive_36_breadth"].ge(
            cfg.layer1_long_minimum_positive_36_breadth,
        )
        & layer1["active_volume_breadth"].ge(
            cfg.layer1_long_minimum_high_volume_breadth,
        )
        & layer1["seller_breadth"].ge(
            cfg.layer1_long_minimum_seller_breadth,
        ),
    )
    for event_time in layer1_long_events["bar_time"]:
        candidates = market[
            market["bar_time"].eq(event_time)
            & market["subtype"].eq("Layer-1")
            & market["normalized_return_288"].le(
                cfg.layer1_long_candidate_maximum_return_288,
            )
            & market["normalized_return_36"].ge(
                cfg.layer1_long_candidate_minimum_return_36,
            )
            & market["normalized_return_3"].abs().le(
                cfg.layer1_long_candidate_maximum_abs_return_3,
            )
            & market["quote_volume_ratio"].ge(
                cfg.layer1_long_candidate_minimum_quote_ratio,
            )
            & market["signed_seller_pressure"].ge(
                cfg.layer1_long_candidate_minimum_seller_pressure,
            )
            & market["range_location_288"].le(
                cfg.layer1_long_candidate_maximum_range_location,
            )
        ]
        for row in _top_liquidity(candidates, capacity=1):
            records.append(
                _signal_record(
                    row,
                    action="OPEN_LONG",
                    family="layer1_capitulation_absorption",
                    score=(
                        -float(row["normalized_return_288"])
                        + float(row["normalized_return_36"])
                    ),
                    reason="Layer-1 多日下跌后的板块反弹与放量卖压吸收",
                ),
            )

    if not records:
        return _empty_signals()
    output = pd.DataFrame.from_records(records)
    output = output.sort_values(
        ["signal_time", "symbol", "action", "factor_family"],
        kind="stable",
    ).drop_duplicates(
        ["signal_time", "symbol", "action"],
        keep="first",
    )
    output["signal_id"] = [
        (
            f"{POLICY_ID}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}"
        )
        for row in output.itertuples(index=False)
    ]
    return output[
        [
            "signal_id",
            "signal_time",
            "symbol",
            "action",
            "operation_type",
            "signal_price",
            "factor_family",
            "score",
            "reason",
            "policy_id",
            "market_only",
            "research_only",
        ]
    ].reset_index(drop=True)


def generate_non_eth_group_event_add_signals(
    feature_rows: pd.DataFrame,
    *,
    config: NonEthGroupEventAddConfig | None = None,
) -> pd.DataFrame:
    """生成补仓候选；执行层必须再核验策略自身已有同向库存。"""

    cfg = config or NonEthGroupEventAddConfig()
    market = _normalize_feature_rows(feature_rows)
    if market.empty:
        return _empty_add_signals()
    payment = _aggregate_group(
        market[market["subtype"].eq("Payment")],
    )
    coin = _aggregate_group(
        market[market["contract_type"].eq("PERPETUAL")],
    )
    layer1 = _aggregate_group(
        market[market["subtype"].eq("Layer-1")],
    )
    market = market.merge(
        payment.add_prefix("payment_").rename(
            columns={"payment_bar_time": "bar_time"},
        ),
        on="bar_time",
        how="left",
    )
    market = market.merge(
        coin.add_prefix("coin_").rename(columns={"coin_bar_time": "bar_time"}),
        on="bar_time",
        how="left",
    )
    records: list[dict[str, object]] = []

    payment_raw = (
        market["subtype"].eq("Payment")
        & market["payment_group_size"].ge(
            cfg.minimum_payment_group_size,
        )
        & market["payment_mean_return_36"].le(
            cfg.payment_maximum_mean_return_36,
        )
        & market["normalized_return_3"].between(
            cfg.payment_minimum_candidate_return_3,
            cfg.payment_maximum_candidate_return_3,
        )
        & market["normalized_return_12"].le(
            cfg.payment_maximum_candidate_return_12,
        )
        & market["normalized_return_36"].le(
            cfg.payment_maximum_candidate_return_36,
        )
        & market["normalized_return_144"].le(
            cfg.payment_maximum_candidate_return_144,
        )
        & market["normalized_return_288"].le(
            cfg.payment_maximum_candidate_return_288,
        )
        & market["range_location_288"].between(
            cfg.payment_minimum_range_location,
            cfg.payment_maximum_range_location,
        )
    )
    payment_onsets = _apply_symbol_rearm(
        _symbol_onsets(market, payment_raw),
        hours=cfg.payment_rearm_hours,
    )
    for row in payment_onsets.to_dict(orient="records"):
        records.append(
            _add_signal_record(
                row,
                action="OPEN_LONG",
                inventory_side="LONG",
                family="payment_deep_pullback_recovery_add",
                score=(
                    -float(row["normalized_return_36"])
                    + float(row["normalized_return_3"])
                ),
                reason="支付币深回撤后出现温和而非过热的闭合反弹",
            ),
        )

    continuation_raw = (
        market["contract_type"].eq("PERPETUAL")
        & market["normalized_return_288"].between(
            cfg.continuation_minimum_return_288,
            cfg.continuation_maximum_return_288,
        )
        & market["normalized_return_144"].between(
            cfg.continuation_minimum_return_144,
            cfg.continuation_maximum_return_144,
        )
        & market["normalized_return_36"].ge(
            cfg.continuation_minimum_return_36,
        )
        & market["normalized_return_12"].ge(
            cfg.continuation_minimum_return_12,
        )
        & market["normalized_return_3"].between(
            cfg.continuation_minimum_return_3,
            cfg.continuation_maximum_return_3,
        )
        & market["quote_volume_ratio"].ge(
            cfg.continuation_minimum_quote_ratio,
        )
        & market["signed_seller_pressure"].le(
            cfg.continuation_maximum_seller_pressure,
        )
        & market["range_location_288"].between(
            cfg.continuation_minimum_range_location,
            cfg.continuation_maximum_range_location,
        )
        & market["coin_mean_return_36"].between(
            cfg.continuation_market_minimum_mean_return_36,
            cfg.continuation_market_maximum_mean_return_36,
        )
    )
    for row in _symbol_onsets(market, continuation_raw).to_dict(
        orient="records",
    ):
        records.append(
            _add_signal_record(
                row,
                action="OPEN_LONG",
                inventory_side="LONG",
                family="taker_flow_continuation_add",
                score=(
                    float(row["normalized_return_12"])
                    + float(row["quote_volume_ratio"])
                    - float(row["signed_seller_pressure"])
                ),
                reason="中周期上行结构中的主动买盘放量续接",
            ),
        )

    layer1_events = _group_onsets(
        layer1,
        layer1["group_size"].ge(cfg.minimum_layer1_group_size)
        & layer1["mean_return_288"].le(
            cfg.layer1_maximum_mean_return_288,
        )
        & layer1["mean_return_144"].between(
            cfg.layer1_minimum_mean_return_144,
            cfg.layer1_maximum_mean_return_144,
        )
        & layer1["mean_return_36"].ge(
            cfg.layer1_minimum_mean_return_36,
        )
        & layer1["mean_return_12"].le(
            cfg.layer1_maximum_mean_return_12,
        )
        & layer1["negative_3_breadth"].ge(
            cfg.layer1_minimum_negative_3_breadth,
        )
        & layer1["seller_breadth"].ge(
            cfg.layer1_minimum_seller_breadth,
        ),
    )
    for event_time in layer1_events["bar_time"]:
        candidates = market[
            market["bar_time"].eq(event_time)
            & market["subtype"].eq("Layer-1")
            & market["normalized_return_288"].le(
                cfg.layer1_candidate_maximum_return_288,
            )
            & market["normalized_return_12"].le(
                cfg.layer1_candidate_maximum_return_12,
            )
            & market["normalized_return_3"].le(
                cfg.layer1_candidate_maximum_return_3,
            )
            & market["signed_seller_pressure"].ge(
                cfg.layer1_candidate_minimum_seller_pressure,
            )
            & market["range_location_288"].le(
                cfg.layer1_candidate_maximum_range_location,
            )
        ]
        for row in _top_liquidity(candidates, capacity=1):
            records.append(
                _add_signal_record(
                    row,
                    action="OPEN_SHORT",
                    inventory_side="SHORT",
                    family="layer1_mature_downtrend_rebreak_add",
                    score=(
                        -float(row["normalized_return_288"])
                        - float(row["normalized_return_12"])
                    ),
                    reason="Layer-1 成熟下跌结构盘整后的卖压再次扩散",
                ),
            )
    if not records:
        return _empty_add_signals()
    output = pd.DataFrame.from_records(records)
    output = output.sort_values(
        ["signal_time", "symbol", "action", "factor_family"],
        kind="stable",
    ).drop_duplicates(
        ["signal_time", "symbol", "action"],
        keep="first",
    )
    output["signal_id"] = [
        (
            f"{ADD_POLICY_ID}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}"
        )
        for row in output.itertuples(index=False)
    ]
    return output[
        [
            "signal_id",
            "signal_time",
            "symbol",
            "action",
            "operation_type",
            "signal_price",
            "factor_family",
            "score",
            "reason",
            "policy_id",
            "market_only",
            "research_only",
            "requires_existing_inventory",
            "required_inventory_side",
        ]
    ].reset_index(drop=True)


def _normalize_feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "non-ETH group-event features missing columns: "
            + ",".join(missing),
        )
    output = frame.copy()
    output["bar_time"] = pd.to_datetime(
        output["bar_time"],
        utc=True,
        errors="coerce",
    )
    if output["bar_time"].isna().any():
        raise ValueError("bar_time must contain valid UTC decision timestamps")
    output["symbol"] = output["symbol"].map(clean_symbol)
    if output["symbol"].eq("ETHUSDT").any():
        raise ValueError("non-ETH group-event factor must exclude ETHUSDT")
    if output.duplicated(["bar_time", "symbol"]).any():
        raise ValueError("group-event feature identities must be unique")
    numeric_columns = (
        "close",
        "rolling_quote_volume_24h",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_144",
        "normalized_return_288",
        "quote_volume_ratio",
        "signed_seller_pressure",
        "range_location_288",
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )
    output["eligible"] = output["eligible"].astype(bool)
    output = output[
        output["eligible"]
        & output["close"].gt(0.0)
        & output["rolling_quote_volume_24h"].gt(0.0)
    ].copy()
    output = output.replace([np.inf, -np.inf], np.nan).dropna(
        subset=numeric_columns,
    )
    return output.sort_values(
        ["bar_time", "symbol"],
        kind="stable",
    ).reset_index(drop=True)


def _aggregate_group(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bar_time",
        "group_size",
        "mean_return_3",
        "mean_return_12",
        "mean_return_36",
        "mean_return_144",
        "mean_return_288",
        "negative_3_breadth",
        "negative_12_breadth",
        "negative_288_breadth",
        "positive_3_breadth",
        "positive_36_breadth",
        "high_volume_breadth",
        "active_volume_breadth",
        "seller_breadth",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output = frame.groupby("bar_time", sort=True).agg(
        group_size=("symbol", "nunique"),
        mean_return_3=("normalized_return_3", "mean"),
        mean_return_12=("normalized_return_12", "mean"),
        mean_return_36=("normalized_return_36", "mean"),
        mean_return_144=("normalized_return_144", "mean"),
        mean_return_288=("normalized_return_288", "mean"),
        negative_3_breadth=("normalized_return_3", lambda x: x.lt(0).mean()),
        negative_12_breadth=(
            "normalized_return_12",
            lambda x: x.lt(0).mean(),
        ),
        negative_288_breadth=(
            "normalized_return_288",
            lambda x: x.le(-8.0).mean(),
        ),
        positive_3_breadth=("normalized_return_3", lambda x: x.gt(0).mean()),
        positive_36_breadth=(
            "normalized_return_36",
            lambda x: x.ge(2.0).mean(),
        ),
        high_volume_breadth=("quote_volume_ratio", lambda x: x.ge(2.0).mean()),
        active_volume_breadth=(
            "quote_volume_ratio",
            lambda x: x.ge(1.5).mean(),
        ),
        seller_breadth=(
            "signed_seller_pressure",
            lambda x: x.ge(0.15).mean(),
        ),
    )
    return output.reset_index()[columns]


def _group_onsets(
    group: pd.DataFrame,
    raw: pd.Series,
) -> pd.DataFrame:
    output = group.sort_values("bar_time", kind="stable").copy()
    selected = raw.reindex(output.index, fill_value=False).astype(bool)
    output["event_onset"] = selected & ~selected.shift(fill_value=False)
    return output[output["event_onset"]].copy()


def _symbol_onsets(
    market: pd.DataFrame,
    raw: pd.Series,
) -> pd.DataFrame:
    output = market.copy()
    output["_raw_event"] = raw.reindex(output.index, fill_value=False).astype(
        bool,
    )
    output = output.sort_values(["symbol", "bar_time"], kind="stable")
    previous = output.groupby("symbol", sort=False)["_raw_event"].shift(
        fill_value=False,
    )
    return output[output["_raw_event"] & ~previous].copy()


def _apply_symbol_rearm(
    frame: pd.DataFrame,
    *,
    hours: float,
) -> pd.DataFrame:
    rearm = pd.Timedelta(hours=max(0.0, float(hours)))
    accepted: list[int] = []
    last_by_symbol: dict[str, pd.Timestamp] = {}
    for index, row in frame.sort_values(
        ["bar_time", "symbol"],
        kind="stable",
    ).iterrows():
        symbol = str(row["symbol"])
        bar_time = pd.Timestamp(row["bar_time"])
        previous = last_by_symbol.get(symbol)
        if previous is None or bar_time - previous >= rearm:
            accepted.append(index)
            last_by_symbol[symbol] = bar_time
    return frame.loc[accepted].sort_values(
        ["bar_time", "symbol"],
        kind="stable",
    )


def _top_liquidity(
    candidates: pd.DataFrame,
    *,
    capacity: int,
) -> list[dict[str, object]]:
    if candidates.empty or capacity <= 0:
        return []
    selected = candidates.sort_values(
        ["rolling_quote_volume_24h", "symbol"],
        ascending=[False, True],
        kind="stable",
    ).head(int(capacity))
    return selected.to_dict(orient="records")


def _signal_record(
    row: dict[str, object],
    *,
    action: str,
    family: str,
    score: float,
    reason: str,
) -> dict[str, object]:
    return {
        "signal_time": pd.Timestamp(row["bar_time"]),
        "symbol": str(row["symbol"]),
        "action": action,
        "operation_type": "OPEN",
        "signal_price": float(row["close"]),
        "factor_family": family,
        "score": float(score),
        "reason": reason,
        "policy_id": POLICY_ID,
        "market_only": True,
        "research_only": True,
    }


def _add_signal_record(
    row: dict[str, object],
    *,
    action: str,
    inventory_side: str,
    family: str,
    score: float,
    reason: str,
) -> dict[str, object]:
    return {
        "signal_time": pd.Timestamp(row["bar_time"]),
        "symbol": str(row["symbol"]),
        "action": action,
        "operation_type": "ADD",
        "signal_price": float(row["close"]),
        "factor_family": family,
        "score": float(score),
        "reason": reason,
        "policy_id": ADD_POLICY_ID,
        "market_only": True,
        "research_only": True,
        "requires_existing_inventory": True,
        "required_inventory_side": inventory_side,
    }


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
            "score",
            "reason",
            "policy_id",
            "market_only",
            "research_only",
        ],
    )


def _empty_add_signals() -> pd.DataFrame:
    output = _empty_signals()
    output["requires_existing_inventory"] = pd.Series(dtype=bool)
    output["required_inventory_side"] = pd.Series(dtype=str)
    return output


__all__ = [
    "ADD_POLICY_ID",
    "POLICY_ID",
    "NonEthGroupEventAddConfig",
    "NonEthGroupEventFactorConfig",
    "generate_non_eth_group_event_add_signals",
    "generate_non_eth_group_event_open_signals",
]
