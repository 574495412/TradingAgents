"""非 ETH 板块事件因子 v2 的历史研究版本。

本版本只组合闭合五分钟行情的板块广度、相对趋势、成交量和主动成交方向。
它不接收带单员事件、交易对名单、收益或逐笔身份。参数来自已经查看过的历史
窗口, 因此只能用于同窗描述性研究; 任何推广都必须重新冻结并等待更新的未见
窗口。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from factors.crypto.code.weijie.research.non_eth_group_event_factor import (
    NonEthGroupEventAddConfig,
    NonEthGroupEventFactorConfig,
    generate_non_eth_group_event_add_signals,
    generate_non_eth_group_event_open_signals,
)

OPEN_POLICY_ID = "weijie_non_eth_group_event_open_research_v2"
ADD_POLICY_ID = "weijie_non_eth_group_event_add_research_v2"


@dataclass(frozen=True)
class NonEthGroupEventV2ResearchConfig:
    """圆整、可复用的二阶段市场质量门槛。"""

    tradfi_negative_3_breadth: float = 0.75
    payment_open_maximum_return_3: float = -0.50
    payment_open_minimum_quote_ratio: float = 3.00
    payment_open_minimum_seller_pressure: float = 0.00

    layer1_long_minimum_active_volume_breadth: float = 0.15
    layer1_long_candidate_minimum_return_144: float = -10.00
    layer1_long_candidate_minimum_quote_volume_24h: float = 250_000_000.00

    payment_add_maximum_return_144: float = -2.00
    payment_add_minimum_quote_ratio: float = 0.80
    continuation_add_maximum_seller_pressure: float = -0.40


def generate_non_eth_group_event_open_signals_v2(
    feature_rows: pd.DataFrame,
    *,
    config: NonEthGroupEventV2ResearchConfig | None = None,
) -> pd.DataFrame:
    """生成 v2 初始开仓候选, 仅使用同一决策时点的市场特征。"""

    cfg = config or NonEthGroupEventV2ResearchConfig()
    base_config = replace(
        NonEthGroupEventFactorConfig(),
        tradfi_negative_3_breadth=cfg.tradfi_negative_3_breadth,
        payment_candidate_maximum_return_3=(
            cfg.payment_open_maximum_return_3
        ),
        payment_candidate_minimum_quote_ratio=(
            cfg.payment_open_minimum_quote_ratio
        ),
        layer1_long_minimum_high_volume_breadth=(
            cfg.layer1_long_minimum_active_volume_breadth
        ),
    )
    signals = generate_non_eth_group_event_open_signals(
        feature_rows,
        config=base_config,
    )
    if signals.empty:
        return _rewrite_policy(signals, policy_id=OPEN_POLICY_ID)
    enriched = _attach_signal_features(signals, feature_rows)
    payment = enriched["factor_family"].eq("payment_trend_rollover")
    layer1_long = enriched["factor_family"].eq(
        "layer1_capitulation_absorption",
    )
    accepted = (
        (
            ~payment
            | enriched["signed_seller_pressure"].ge(
                cfg.payment_open_minimum_seller_pressure,
            )
        )
        & (
            ~layer1_long
            | (
                enriched["normalized_return_144"].ge(
                    cfg.layer1_long_candidate_minimum_return_144,
                )
                & enriched["rolling_quote_volume_24h"].ge(
                    cfg.layer1_long_candidate_minimum_quote_volume_24h,
                )
            )
        )
    )
    output = enriched.loc[accepted, signals.columns].copy()
    return _rewrite_policy(output, policy_id=OPEN_POLICY_ID)


def generate_non_eth_group_event_add_signals_v2(
    feature_rows: pd.DataFrame,
    *,
    config: NonEthGroupEventV2ResearchConfig | None = None,
) -> pd.DataFrame:
    """生成 v2 补仓候选; 执行层仍须验证同向自主库存。"""

    cfg = config or NonEthGroupEventV2ResearchConfig()
    base_config = replace(
        NonEthGroupEventAddConfig(),
        payment_maximum_candidate_return_144=(
            cfg.payment_add_maximum_return_144
        ),
        continuation_maximum_seller_pressure=(
            cfg.continuation_add_maximum_seller_pressure
        ),
    )
    signals = generate_non_eth_group_event_add_signals(
        feature_rows,
        config=base_config,
    )
    if signals.empty:
        return _rewrite_policy(signals, policy_id=ADD_POLICY_ID)
    enriched = _attach_signal_features(signals, feature_rows)
    payment = enriched["factor_family"].eq(
        "payment_deep_pullback_recovery_add",
    )
    accepted = ~payment | enriched["quote_volume_ratio"].ge(
        cfg.payment_add_minimum_quote_ratio,
    )
    output = enriched.loc[accepted, signals.columns].copy()
    return _rewrite_policy(output, policy_id=ADD_POLICY_ID)


def _attach_signal_features(
    signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "bar_time",
        "symbol",
        "normalized_return_144",
        "rolling_quote_volume_24h",
        "quote_volume_ratio",
        "signed_seller_pressure",
    }
    missing = sorted(required.difference(feature_rows.columns))
    if missing:
        raise ValueError("v2 市场质量过滤缺少特征: " + ",".join(missing))
    features = feature_rows[
        [
            "bar_time",
            "symbol",
            "normalized_return_144",
            "rolling_quote_volume_24h",
            "quote_volume_ratio",
            "signed_seller_pressure",
        ]
    ].copy()
    features["bar_time"] = pd.to_datetime(
        features["bar_time"],
        utc=True,
        errors="raise",
    )
    features = features.drop_duplicates(
        ["bar_time", "symbol"],
        keep="last",
    )
    output = signals.copy()
    output["signal_time"] = pd.to_datetime(
        output["signal_time"],
        utc=True,
        errors="raise",
    )
    output = output.merge(
        features,
        left_on=["signal_time", "symbol"],
        right_on=["bar_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    quality_columns = [
        "normalized_return_144",
        "rolling_quote_volume_24h",
        "quote_volume_ratio",
        "signed_seller_pressure",
    ]
    if output[quality_columns].isna().any(axis=None):
        message = "v2 信号无法一对一关联同一时点市场特征"
        raise ValueError(message)
    return output.drop(columns="bar_time")


def _rewrite_policy(
    signals: pd.DataFrame,
    *,
    policy_id: str,
) -> pd.DataFrame:
    output = signals.copy()
    if output.empty:
        if "policy_id" in output:
            output["policy_id"] = policy_id
        return output
    output["policy_id"] = policy_id
    output["signal_id"] = [
        (
            f"{policy_id}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}"
        )
        for row in output.itertuples(index=False)
    ]
    if output["signal_id"].duplicated().any():
        message = "v2 信号 ID 必须唯一"
        raise ValueError(message)
    return output.sort_values(
        ["signal_time", "symbol", "action"],
        kind="stable",
    ).reset_index(drop=True)


__all__ = [
    "ADD_POLICY_ID",
    "OPEN_POLICY_ID",
    "NonEthGroupEventV2ResearchConfig",
    "generate_non_eth_group_event_add_signals_v2",
    "generate_non_eth_group_event_open_signals_v2",
]
