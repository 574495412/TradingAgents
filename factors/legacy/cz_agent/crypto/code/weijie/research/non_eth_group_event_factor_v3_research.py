"""非 ETH 板块事件因子 v3 的同窗研究版本。

v3 解决两个执行语义问题：

1. 板块回落信号必须在流量质量同时满足后才判定 onset，避免先触发、后过滤
   导致质量下一根确认时永久漏单。
2. 可解释的深跌吸收和中期突破可以建立长期 campaign 库存，后续 ADD 仍须
   由真实自主账本验证同向库存，不能把 ADD 候选直接当作已执行。

本模块只接收闭合五分钟市场特征。参数已经参考过历史标签，只能用于同窗
描述性研究；推广必须重新冻结并等待新的未见窗口。
"""

from __future__ import annotations

from collections.abc import Callable

# ruff: noqa: EM101, PERF401, RUF002, SLF001, TRY003
from dataclasses import dataclass, replace

import pandas as pd

from factors.crypto.code.weijie.research import (
    non_eth_group_event_factor as base_factor,
)
from factors.crypto.code.weijie.research.non_eth_group_event_factor_v2_research import (
    NonEthGroupEventV2ResearchConfig,
    generate_non_eth_group_event_add_signals_v2,
    generate_non_eth_group_event_open_signals_v2,
)

OPEN_POLICY_ID = "weijie_non_eth_group_event_open_research_v3"
ADD_POLICY_ID = "weijie_non_eth_group_event_add_research_v3"

TACTICAL_PROFILE = "TACTICAL"
PAYMENT_CAMPAIGN_PROFILE = "PAYMENT_CAMPAIGN"
RWA_CAMPAIGN_PROFILE = "RWA_CAMPAIGN"
LAYER1_CAMPAIGN_PROFILE = "LAYER1_CAMPAIGN"
INHERIT_PROFILE = "INHERIT"


@dataclass(frozen=True)
class NonEthGroupEventV3ResearchConfig:
    """圆整且具有市场含义的 v3 开仓门槛。"""

    payment_quality_minimum_seller_pressure: float = 0.00

    payment_campaign_minimum_group_size: int = 3
    payment_campaign_maximum_group_return_12: float = -2.00
    payment_campaign_maximum_group_return_36: float = -2.00
    payment_campaign_minimum_negative_12_breadth: float = 0.66
    payment_campaign_maximum_candidate_return_12: float = -4.00
    payment_campaign_maximum_candidate_return_36: float = -5.00
    payment_campaign_maximum_candidate_return_288: float = -6.00
    payment_campaign_maximum_range_location: float = 0.10
    payment_campaign_minimum_quote_ratio: float = 1.50
    payment_campaign_minimum_seller_pressure: float = 0.15
    payment_campaign_rearm_hours: float = 24.00

    rwa_campaign_minimum_return_288: float = 6.00
    rwa_campaign_maximum_return_288: float = 15.00
    rwa_campaign_minimum_return_144: float = 6.00
    rwa_campaign_maximum_return_144: float = 12.00
    rwa_campaign_minimum_return_36: float = 2.00
    rwa_campaign_minimum_return_3: float = 2.00
    rwa_campaign_minimum_range_location: float = 0.90
    rwa_campaign_minimum_quote_ratio: float = 1.50
    rwa_campaign_minimum_seller_pressure: float = 0.15
    rwa_campaign_minimum_quote_volume_24h: float = 50_000_000.00
    rwa_campaign_minimum_coin_return_12: float = -1.00
    rwa_campaign_maximum_coin_return_12: float = 2.00
    rwa_campaign_maximum_coin_return_36: float = 0.50
    rwa_campaign_rearm_hours: float = 168.00


def generate_non_eth_group_event_open_signals_v3(
    feature_rows: pd.DataFrame,
    *,
    v2_config: NonEthGroupEventV2ResearchConfig | None = None,
    config: NonEthGroupEventV3ResearchConfig | None = None,
) -> pd.DataFrame:
    """生成 v3 OPEN，并声明每笔信号的自主生命周期类别。"""

    v2_cfg = v2_config or NonEthGroupEventV2ResearchConfig()
    cfg = config or NonEthGroupEventV3ResearchConfig()
    market = base_factor._normalize_feature_rows(feature_rows)
    if market.empty:
        return _empty_open_signals()

    payment = base_factor._aggregate_group(
        market[market["subtype"].eq("Payment")],
    )
    coin = base_factor._aggregate_group(
        market[market["contract_type"].eq("PERPETUAL")],
    )
    enriched = market.merge(
        payment.add_prefix("payment_").rename(
            columns={"payment_bar_time": "bar_time"},
        ),
        on="bar_time",
        how="left",
    )
    enriched = enriched.merge(
        coin.add_prefix("coin_").rename(
            columns={"coin_bar_time": "bar_time"},
        ),
        on="bar_time",
        how="left",
    )

    base = generate_non_eth_group_event_open_signals_v2(
        feature_rows,
        config=v2_cfg,
    )
    base = base[
        ~base["factor_family"].eq("payment_trend_rollover")
    ].copy()
    base["lifecycle_profile"] = TACTICAL_PROFILE
    base.loc[
        base["factor_family"].eq("layer1_squeeze_exhaustion"),
        "lifecycle_profile",
    ] = LAYER1_CAMPAIGN_PROFILE

    quality_confirmed_payment = _quality_confirmed_payment_rollover(
        enriched,
        v2_config=v2_cfg,
        config=cfg,
    )
    payment_campaign = _payment_capitulation_campaign(
        enriched,
        config=cfg,
    )
    rwa_campaign = _rwa_moderate_breakout_campaign(
        enriched,
        config=cfg,
    )
    parts = [
        base,
        quality_confirmed_payment,
        payment_campaign,
        rwa_campaign,
    ]
    nonempty_parts = [part for part in parts if not part.empty]
    if not nonempty_parts:
        return _empty_open_signals()
    output = pd.concat(
        nonempty_parts,
        ignore_index=True,
        sort=False,
    )
    profile_priority = {
        PAYMENT_CAMPAIGN_PROFILE: 0,
        RWA_CAMPAIGN_PROFILE: 0,
        LAYER1_CAMPAIGN_PROFILE: 1,
        TACTICAL_PROFILE: 2,
    }
    output["_profile_priority"] = (
        output["lifecycle_profile"].map(profile_priority).fillna(9)
    )
    output = (
        output.sort_values(
            [
                "signal_time",
                "symbol",
                "action",
                "_profile_priority",
                "factor_family",
            ],
            kind="stable",
        )
        .drop_duplicates(
            ["signal_time", "symbol", "action"],
            keep="first",
        )
        .drop(columns="_profile_priority")
        .reset_index(drop=True)
    )
    return _rewrite_policy(output, policy_id=OPEN_POLICY_ID)


def generate_non_eth_group_event_add_signals_v3(
    feature_rows: pd.DataFrame,
    *,
    v2_config: NonEthGroupEventV2ResearchConfig | None = None,
) -> pd.DataFrame:
    """沿用 v2 ADD 市场条件，执行层仍必须验证真实同向库存。"""

    output = generate_non_eth_group_event_add_signals_v2(
        feature_rows,
        config=v2_config or NonEthGroupEventV2ResearchConfig(),
    ).copy()
    output["lifecycle_profile"] = INHERIT_PROFILE
    return _rewrite_policy(output, policy_id=ADD_POLICY_ID)


def _quality_confirmed_payment_rollover(
    market: pd.DataFrame,
    *,
    v2_config: NonEthGroupEventV2ResearchConfig,
    config: NonEthGroupEventV3ResearchConfig,
) -> pd.DataFrame:
    base = replace(
        base_factor.NonEthGroupEventFactorConfig(),
        payment_candidate_maximum_return_3=(
            v2_config.payment_open_maximum_return_3
        ),
        payment_candidate_minimum_quote_ratio=(
            v2_config.payment_open_minimum_quote_ratio
        ),
    )
    condition = (
        market["subtype"].eq("Payment")
        & market["payment_group_size"].ge(
            base.minimum_payment_group_size,
        )
        & market["payment_mean_return_36"].ge(
            base.payment_minimum_mean_return_36,
        )
        & market["payment_mean_return_144"].ge(
            base.payment_minimum_mean_return_144,
        )
        & market["payment_mean_return_12"].ge(
            base.payment_minimum_mean_return_12,
        )
        & market["payment_negative_3_breadth"].ge(
            base.payment_minimum_negative_3_breadth,
        )
        & market["normalized_return_36"].ge(
            base.payment_candidate_minimum_return_36,
        )
        & market["normalized_return_144"].ge(
            base.payment_candidate_minimum_return_144,
        )
        & market["normalized_return_3"].le(
            base.payment_candidate_maximum_return_3,
        )
        & market["quote_volume_ratio"].ge(
            base.payment_candidate_minimum_quote_ratio,
        )
        & market["signed_seller_pressure"].ge(
            config.payment_quality_minimum_seller_pressure,
        )
    )
    onsets = base_factor._symbol_onsets(market, condition)
    return _signals_from_rows(
        onsets,
        action="OPEN_SHORT",
        family="payment_trend_rollover_quality_confirmed",
        profile=TACTICAL_PROFILE,
        reason="支付币长趋势回落且主动卖压质量完成确认",
        score=lambda row: (
            float(row["normalized_return_144"])
            - float(row["normalized_return_3"])
        ),
    )


def _payment_capitulation_campaign(
    market: pd.DataFrame,
    *,
    config: NonEthGroupEventV3ResearchConfig,
) -> pd.DataFrame:
    condition = (
        market["subtype"].eq("Payment")
        & market["payment_group_size"].ge(
            config.payment_campaign_minimum_group_size,
        )
        & market["payment_mean_return_12"].le(
            config.payment_campaign_maximum_group_return_12,
        )
        & market["payment_mean_return_36"].le(
            config.payment_campaign_maximum_group_return_36,
        )
        & market["payment_negative_12_breadth"].ge(
            config.payment_campaign_minimum_negative_12_breadth,
        )
        & market["normalized_return_12"].le(
            config.payment_campaign_maximum_candidate_return_12,
        )
        & market["normalized_return_36"].le(
            config.payment_campaign_maximum_candidate_return_36,
        )
        & market["normalized_return_288"].le(
            config.payment_campaign_maximum_candidate_return_288,
        )
        & market["range_location_288"].le(
            config.payment_campaign_maximum_range_location,
        )
        & market["quote_volume_ratio"].ge(
            config.payment_campaign_minimum_quote_ratio,
        )
        & market["signed_seller_pressure"].ge(
            config.payment_campaign_minimum_seller_pressure,
        )
    )
    onsets = base_factor._apply_symbol_rearm(
        base_factor._symbol_onsets(market, condition),
        hours=config.payment_campaign_rearm_hours,
    )
    return _signals_from_rows(
        onsets,
        action="OPEN_LONG",
        family="payment_capitulation_campaign",
        profile=PAYMENT_CAMPAIGN_PROFILE,
        reason="支付币板块同步深跌后的低位卖压吸收",
        score=lambda row: (
            -float(row["normalized_return_36"])
            + float(row["signed_seller_pressure"])
        ),
    )


def _rwa_moderate_breakout_campaign(
    market: pd.DataFrame,
    *,
    config: NonEthGroupEventV3ResearchConfig,
) -> pd.DataFrame:
    condition = (
        market["subtype"].eq("RWA")
        & market["normalized_return_288"].between(
            config.rwa_campaign_minimum_return_288,
            config.rwa_campaign_maximum_return_288,
        )
        & market["normalized_return_144"].between(
            config.rwa_campaign_minimum_return_144,
            config.rwa_campaign_maximum_return_144,
        )
        & market["normalized_return_36"].ge(
            config.rwa_campaign_minimum_return_36,
        )
        & market["normalized_return_3"].ge(
            config.rwa_campaign_minimum_return_3,
        )
        & market["range_location_288"].ge(
            config.rwa_campaign_minimum_range_location,
        )
        & market["quote_volume_ratio"].ge(
            config.rwa_campaign_minimum_quote_ratio,
        )
        & market["signed_seller_pressure"].ge(
            config.rwa_campaign_minimum_seller_pressure,
        )
        & market["rolling_quote_volume_24h"].ge(
            config.rwa_campaign_minimum_quote_volume_24h,
        )
        & market["coin_mean_return_12"].between(
            config.rwa_campaign_minimum_coin_return_12,
            config.rwa_campaign_maximum_coin_return_12,
        )
        & market["coin_mean_return_36"].le(
            config.rwa_campaign_maximum_coin_return_36,
        )
    )
    onsets = base_factor._apply_symbol_rearm(
        base_factor._symbol_onsets(market, condition),
        hours=config.rwa_campaign_rearm_hours,
    )
    return _signals_from_rows(
        onsets,
        action="OPEN_LONG",
        family="rwa_moderate_breakout_campaign",
        profile=RWA_CAMPAIGN_PROFILE,
        reason="RWA 中期相对强势但未极端过热的放量突破",
        score=lambda row: (
            float(row["normalized_return_144"])
            + float(row["normalized_return_3"])
        ),
    )


def _signals_from_rows(
    rows: pd.DataFrame,
    *,
    action: str,
    family: str,
    profile: str,
    reason: str,
    score: Callable[[dict[str, object]], float],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows.to_dict(orient="records"):
        records.append(
            {
                "signal_id": "",
                "signal_time": pd.Timestamp(row["bar_time"]),
                "symbol": str(row["symbol"]),
                "action": action,
                "operation_type": "OPEN",
                "signal_price": float(row["close"]),
                "factor_family": family,
                "score": float(score(row)),
                "reason": reason,
                "policy_id": OPEN_POLICY_ID,
                "market_only": True,
                "research_only": True,
                "lifecycle_profile": profile,
            },
        )
    if not records:
        return _empty_open_signals()
    return pd.DataFrame.from_records(records)


def _rewrite_policy(
    signals: pd.DataFrame,
    *,
    policy_id: str,
) -> pd.DataFrame:
    output = signals.copy()
    output["policy_id"] = policy_id
    if output.empty:
        return output
    output["signal_time"] = pd.to_datetime(
        output["signal_time"],
        utc=True,
        errors="raise",
    )
    output["signal_id"] = [
        (
            f"{policy_id}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}"
        )
        for row in output.itertuples(index=False)
    ]
    if output["signal_id"].duplicated().any():
        raise ValueError("v3 信号 ID 必须唯一")
    return output.sort_values(
        ["signal_time", "symbol", "action"],
        kind="stable",
    ).reset_index(drop=True)


def _empty_open_signals() -> pd.DataFrame:
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
            "lifecycle_profile",
        ],
    )


__all__ = [
    "ADD_POLICY_ID",
    "INHERIT_PROFILE",
    "LAYER1_CAMPAIGN_PROFILE",
    "OPEN_POLICY_ID",
    "PAYMENT_CAMPAIGN_PROFILE",
    "RWA_CAMPAIGN_PROFILE",
    "TACTICAL_PROFILE",
    "NonEthGroupEventV3ResearchConfig",
    "generate_non_eth_group_event_add_signals_v3",
    "generate_non_eth_group_event_open_signals_v3",
]
