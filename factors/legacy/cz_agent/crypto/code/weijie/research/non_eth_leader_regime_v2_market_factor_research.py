"""非 ETH 最新制度的市场因子候选。

该模块只使用闭合五分钟价格、成交量、主动成交方向和交易所元数据生成
价格提案。衍生品持仓量与 taker 买卖量确认由独立的因果特征层完成。

带单员标签不进入本模块。当前参数已经看过同窗标签，因此只能用于描述性
研究；冻结后必须在更新的未见窗口验证。
"""

from __future__ import annotations

# ruff: noqa: EM101, RUF002, SLF001, TRY003
from dataclasses import dataclass

import pandas as pd

from factors.crypto.code.weijie.research import (
    non_eth_group_event_factor as base_factor,
)
from factors.crypto.code.weijie.research.non_eth_group_event_factor_v3_research import (
    INHERIT_PROFILE,
    TACTICAL_PROFILE,
    generate_non_eth_group_event_add_signals_v3,
    generate_non_eth_group_event_open_signals_v3,
)

POLICY_ID = "weijie_non_eth_leader_regime_v2_market_factor_research_v1"
MEME_CAMPAIGN_PROFILE = "MEME_CAMPAIGN"


@dataclass(frozen=True)
class NonEthLeaderRegimeV2PriceProposalConfig:
    """具有市场含义的圆整价格提案阈值。"""

    tradfi_maximum_group_return_36: float = -1.0
    tradfi_minimum_group_return_288: float = 10.0
    tradfi_minimum_group_seller_breadth: float = 0.50

    layer1_short_minimum_group_return_3: float = 1.50
    layer1_short_minimum_candidate_return_3: float = 1.00
    idiosyncratic_minimum_quote_ratio: float = 1.50
    payment_campaign_group_cooldown_hours: float = 48.0

    meme_minimum_group_size: int = 3
    meme_long_maximum_group_return_36: float = 0.0
    meme_long_maximum_group_return_144: float = -2.0
    meme_long_maximum_group_return_288: float = -8.0
    meme_long_minimum_negative_12_breadth: float = 0.66
    meme_long_maximum_candidate_return_12: float = -2.0
    meme_long_maximum_candidate_return_36: float = -2.0
    meme_long_maximum_candidate_return_144: float = -5.0
    meme_long_maximum_range_location: float = 0.10
    meme_long_minimum_quote_ratio: float = 0.80
    meme_long_minimum_seller_pressure: float = 0.10
    meme_long_minimum_quote_volume_24h: float = 100_000_000.0

    meme_short_minimum_group_return_36: float = 3.0
    meme_short_minimum_negative_3_breadth: float = 0.66
    meme_short_minimum_candidate_return_36: float = 5.0
    meme_short_minimum_candidate_return_144: float = 8.0
    meme_short_maximum_candidate_return_3: float = 0.0
    meme_short_minimum_range_location: float = 0.90
    meme_short_minimum_quote_ratio: float = 1.50

    payment_short_minimum_group_size: int = 2
    payment_short_minimum_group_return_36: float = 5.0
    payment_short_minimum_group_return_144: float = 5.0
    payment_short_maximum_market_return_144: float = 5.0
    payment_short_minimum_candidate_return_36: float = 5.0
    payment_short_minimum_candidate_return_144: float = 5.0
    payment_short_minimum_abs_return_3: float = 1.0
    payment_short_minimum_quote_ratio: float = 0.80

    rwa_shock_maximum_return_288: float = -6.0
    rwa_shock_maximum_return_3: float = -1.0
    rwa_shock_maximum_return_12: float = -3.0
    rwa_shock_minimum_quote_ratio: float = 5.0
    rwa_shock_minimum_seller_pressure: float = 0.10
    rwa_shock_minimum_range_location: float = 0.10
    rwa_shock_maximum_range_location: float = 0.25

    rwa_recovery_maximum_return_288: float = -12.0
    rwa_recovery_minimum_return_144: float = 0.0
    rwa_recovery_minimum_return_36: float = 2.0
    rwa_recovery_minimum_return_3: float = 0.50
    rwa_recovery_maximum_return_3: float = 1.50
    rwa_recovery_minimum_quote_ratio: float = 2.0
    rwa_recovery_minimum_range_location: float = 0.15
    rwa_recovery_maximum_range_location: float = 0.30

    payment_add_minimum_group_size: int = 3
    payment_add_maximum_group_return_36: float = -1.50
    payment_add_minimum_return_3: float = 0.50
    payment_add_maximum_return_3: float = 1.25
    payment_add_maximum_return_12: float = -1.0
    payment_add_maximum_return_36: float = -2.0
    payment_add_maximum_return_144: float = -2.0
    payment_add_maximum_return_288: float = -4.0
    payment_add_minimum_range_location: float = 0.08
    payment_add_maximum_range_location: float = 0.25
    payment_add_minimum_quote_ratio: float = 0.80


@dataclass(frozen=True)
class NonEthLeaderRegimeV2DerivativeConfig:
    """闭合衍生品市场确认门槛。"""

    meme_long_maximum_oi_change_pct: float = -0.50
    meme_long_maximum_taker_ratio_3h: float = 0.85
    meme_long_sector_rearm_hours: float = 72.0
    meme_long_cluster_minutes: float = 30.0

    meme_short_minimum_oi_change_3h_pct: float = 3.0
    meme_short_minimum_taker_ratio_1h: float = 1.15
    meme_short_rearm_hours: float = 24.0

    payment_short_minimum_oi_change_3h_pct: float = 1.0
    payment_short_minimum_taker_ratio_3h: float = 1.10
    payment_short_rearm_hours: float = 24.0

    rwa_add_maximum_oi_change_3h_pct: float = -0.30
    rwa_add_rearm_hours: float = 1.0

    payment_add_maximum_oi_change_3h_pct: float = -0.30
    payment_add_maximum_taker_ratio_3h: float = 0.90
    payment_add_rearm_hours: float = 24.0

    confirmation_delay_minutes: float = 5.0


def generate_non_eth_leader_regime_v2_price_proposals(
    feature_rows: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2PriceProposalConfig | None = None,
) -> pd.DataFrame:
    """生成基础因子和新制度扩展因子的市场价格提案。"""

    cfg = config or NonEthLeaderRegimeV2PriceProposalConfig()
    market = base_factor._normalize_feature_rows(feature_rows)
    if market.empty:
        return _empty_proposals()

    meme = base_factor._aggregate_group(
        market[market["subtype"].eq("Meme")],
    )
    payment = base_factor._aggregate_group(
        market[market["subtype"].eq("Payment")],
    )
    tradfi = base_factor._aggregate_group(
        market[market["contract_type"].eq("TRADIFI_PERPETUAL")],
    )
    layer1 = base_factor._aggregate_group(
        market[market["subtype"].eq("Layer-1")],
    )
    coin = base_factor._aggregate_group(
        market[market["contract_type"].eq("PERPETUAL")],
    )
    enriched = _merge_group(market, meme, prefix="meme")
    enriched = _merge_group(enriched, payment, prefix="payment")
    enriched = _merge_group(enriched, tradfi, prefix="tradfi")
    enriched = _merge_group(enriched, layer1, prefix="layer1")
    enriched = _merge_group(enriched, coin, prefix="coin")

    base_open = _refined_base_open_proposals(
        feature_rows,
        enriched=enriched,
        config=cfg,
    )
    base_add = generate_non_eth_group_event_add_signals_v3(
        feature_rows,
    )
    extensions = _extension_price_states(enriched, config=cfg)
    parts = [
        part
        for part in (base_open, base_add, extensions)
        if not part.empty
    ]
    if not parts:
        return _empty_proposals()
    output = pd.concat(parts, ignore_index=True, sort=False)
    output["signal_time"] = pd.to_datetime(
        output["signal_time"],
        utc=True,
        errors="raise",
    )
    output["policy_id"] = POLICY_ID
    output["signal_id"] = [
        (
            f"{POLICY_ID}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}|{row.factor_family}"
        )
        for row in output.itertuples(index=False)
    ]
    output = output.sort_values(
        ["signal_time", "symbol", "action", "factor_family"],
        kind="stable",
    ).drop_duplicates(
        ["signal_time", "symbol", "action", "factor_family"],
        keep="first",
    )
    if output["signal_id"].duplicated().any():
        raise ValueError("最新制度价格提案 ID 必须唯一")
    return output.reset_index(drop=True)


def _refined_base_open_proposals(
    feature_rows: pd.DataFrame,
    *,
    enriched: pd.DataFrame,
    config: NonEthLeaderRegimeV2PriceProposalConfig,
) -> pd.DataFrame:
    signals = generate_non_eth_group_event_open_signals_v3(
        feature_rows,
    )
    if signals.empty:
        return signals
    features = enriched[
        [
            "bar_time",
            "symbol",
            "normalized_return_3",
            "quote_volume_ratio",
            "tradfi_mean_return_36",
            "tradfi_mean_return_288",
            "tradfi_seller_breadth",
            "layer1_mean_return_3",
        ]
    ].drop_duplicates(["bar_time", "symbol"], keep="last")
    output = signals.merge(
        features,
        left_on=["signal_time", "symbol"],
        right_on=["bar_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if output["bar_time"].isna().any():
        raise ValueError("基础 OPEN 无法关联同一时点市场特征")

    tradfi = output["factor_family"].eq(
        "tradfi_extended_breadth_break",
    )
    layer1_short = output["factor_family"].eq(
        "layer1_squeeze_exhaustion",
    )
    idiosyncratic = output["factor_family"].eq(
        "extended_idiosyncratic_rollover",
    )
    accepted = (
        (
            ~tradfi
            | (
                output["tradfi_mean_return_36"].le(
                    config.tradfi_maximum_group_return_36,
                )
                & output["tradfi_mean_return_288"].ge(
                    config.tradfi_minimum_group_return_288,
                )
                & output["tradfi_seller_breadth"].ge(
                    config.tradfi_minimum_group_seller_breadth,
                )
            )
        )
        & (
            ~layer1_short
            | (
                output["layer1_mean_return_3"].ge(
                    config.layer1_short_minimum_group_return_3,
                )
                & output["normalized_return_3"].ge(
                    config.layer1_short_minimum_candidate_return_3,
                )
            )
        )
        & (
            ~idiosyncratic
            | output["quote_volume_ratio"].ge(
                config.idiosyncratic_minimum_quote_ratio,
            )
        )
    )
    output = output.loc[accepted, signals.columns].copy()
    return _apply_payment_campaign_cooldown(
        output,
        hours=config.payment_campaign_group_cooldown_hours,
    )


def _extension_price_states(
    market: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2PriceProposalConfig,
) -> pd.DataFrame:
    states = (
        (
            "meme_capitulation_long",
            "OPEN_LONG",
            "OPEN",
            TACTICAL_PROFILE,
            "Meme 板块同步下跌后的低位卖压吸收",
            (
                market["subtype"].eq("Meme")
                & market["meme_group_size"].ge(
                    config.meme_minimum_group_size,
                )
                & market["meme_mean_return_36"].le(
                    config.meme_long_maximum_group_return_36,
                )
                & market["meme_mean_return_144"].le(
                    config.meme_long_maximum_group_return_144,
                )
                & market["meme_mean_return_288"].le(
                    config.meme_long_maximum_group_return_288,
                )
                & market["meme_negative_12_breadth"].ge(
                    config.meme_long_minimum_negative_12_breadth,
                )
                & market["normalized_return_12"].le(
                    config.meme_long_maximum_candidate_return_12,
                )
                & market["normalized_return_36"].le(
                    config.meme_long_maximum_candidate_return_36,
                )
                & market["normalized_return_144"].le(
                    config.meme_long_maximum_candidate_return_144,
                )
                & market["range_location_288"].le(
                    config.meme_long_maximum_range_location,
                )
                & market["quote_volume_ratio"].ge(
                    config.meme_long_minimum_quote_ratio,
                )
                & market["signed_seller_pressure"].ge(
                    config.meme_long_minimum_seller_pressure,
                )
                & market["rolling_quote_volume_24h"].ge(
                    config.meme_long_minimum_quote_volume_24h,
                )
            ),
        ),
        (
            "meme_crowded_rollover_short",
            "OPEN_SHORT",
            "OPEN",
            TACTICAL_PROFILE,
            "Meme 中期拥挤上涨后的板块同步回落",
            (
                market["subtype"].eq("Meme")
                & market["meme_group_size"].ge(
                    config.meme_minimum_group_size,
                )
                & market["meme_mean_return_36"].ge(
                    config.meme_short_minimum_group_return_36,
                )
                & market["meme_negative_3_breadth"].ge(
                    config.meme_short_minimum_negative_3_breadth,
                )
                & market["normalized_return_36"].ge(
                    config.meme_short_minimum_candidate_return_36,
                )
                & market["normalized_return_144"].ge(
                    config.meme_short_minimum_candidate_return_144,
                )
                & market["normalized_return_3"].le(
                    config.meme_short_maximum_candidate_return_3,
                )
                & market["range_location_288"].ge(
                    config.meme_short_minimum_range_location,
                )
                & market["quote_volume_ratio"].ge(
                    config.meme_short_minimum_quote_ratio,
                )
            ),
        ),
        (
            "payment_crowded_momentum_short",
            "OPEN_SHORT",
            "OPEN",
            TACTICAL_PROFILE,
            "支付币中期上涨且短周期拥挤动量确认",
            (
                market["subtype"].eq("Payment")
                & market["payment_group_size"].ge(
                    config.payment_short_minimum_group_size,
                )
                & market["payment_mean_return_36"].ge(
                    config.payment_short_minimum_group_return_36,
                )
                & market["payment_mean_return_144"].ge(
                    config.payment_short_minimum_group_return_144,
                )
                & market["coin_mean_return_144"].le(
                    config.payment_short_maximum_market_return_144,
                )
                & market["normalized_return_36"].ge(
                    config.payment_short_minimum_candidate_return_36,
                )
                & market["normalized_return_144"].ge(
                    config.payment_short_minimum_candidate_return_144,
                )
                & market["normalized_return_3"].abs().ge(
                    config.payment_short_minimum_abs_return_3,
                )
                & market["quote_volume_ratio"].ge(
                    config.payment_short_minimum_quote_ratio,
                )
            ),
        ),
        (
            "rwa_shock_deleveraging_add",
            "OPEN_LONG",
            "ADD",
            INHERIT_PROFILE,
            "RWA 长周期回撤中的放量急跌去杠杆候选",
            (
                market["subtype"].eq("RWA")
                & market["normalized_return_288"].le(
                    config.rwa_shock_maximum_return_288,
                )
                & market["normalized_return_3"].le(
                    config.rwa_shock_maximum_return_3,
                )
                & market["normalized_return_12"].le(
                    config.rwa_shock_maximum_return_12,
                )
                & market["quote_volume_ratio"].ge(
                    config.rwa_shock_minimum_quote_ratio,
                )
                & market["signed_seller_pressure"].ge(
                    config.rwa_shock_minimum_seller_pressure,
                )
                & market["range_location_288"].between(
                    config.rwa_shock_minimum_range_location,
                    config.rwa_shock_maximum_range_location,
                )
            ),
        ),
        (
            "rwa_recovery_deleveraging_add",
            "OPEN_LONG",
            "ADD",
            INHERIT_PROFILE,
            "RWA 深跌后的中周期恢复和去杠杆候选",
            (
                market["subtype"].eq("RWA")
                & market["normalized_return_288"].le(
                    config.rwa_recovery_maximum_return_288,
                )
                & market["normalized_return_144"].ge(
                    config.rwa_recovery_minimum_return_144,
                )
                & market["normalized_return_36"].ge(
                    config.rwa_recovery_minimum_return_36,
                )
                & market["normalized_return_3"].between(
                    config.rwa_recovery_minimum_return_3,
                    config.rwa_recovery_maximum_return_3,
                )
                & market["quote_volume_ratio"].ge(
                    config.rwa_recovery_minimum_quote_ratio,
                )
                & market["range_location_288"].between(
                    config.rwa_recovery_minimum_range_location,
                    config.rwa_recovery_maximum_range_location,
                )
            ),
        ),
        (
            "payment_deleveraging_recovery_add",
            "OPEN_LONG",
            "ADD",
            INHERIT_PROFILE,
            "支付币深跌去杠杆后的低位恢复候选",
            (
                market["subtype"].eq("Payment")
                & market["payment_group_size"].ge(
                    config.payment_add_minimum_group_size,
                )
                & market["payment_mean_return_36"].le(
                    config.payment_add_maximum_group_return_36,
                )
                & market["normalized_return_3"].between(
                    config.payment_add_minimum_return_3,
                    config.payment_add_maximum_return_3,
                )
                & market["normalized_return_12"].le(
                    config.payment_add_maximum_return_12,
                )
                & market["normalized_return_36"].le(
                    config.payment_add_maximum_return_36,
                )
                & market["normalized_return_144"].le(
                    config.payment_add_maximum_return_144,
                )
                & market["normalized_return_288"].le(
                    config.payment_add_maximum_return_288,
                )
                & market["range_location_288"].between(
                    config.payment_add_minimum_range_location,
                    config.payment_add_maximum_range_location,
                )
                & market["quote_volume_ratio"].ge(
                    config.payment_add_minimum_quote_ratio,
                )
            ),
        ),
    )
    parts: list[pd.DataFrame] = []
    for family, action, operation_type, profile, reason, condition in states:
        rows = market.loc[condition].copy()
        if rows.empty:
            continue
        rows["signal_id"] = ""
        rows["signal_time"] = rows["bar_time"]
        rows["action"] = action
        rows["operation_type"] = operation_type
        rows["signal_price"] = rows["close"]
        rows["factor_family"] = family
        rows["score"] = _extension_score(rows, family=family)
        rows["reason"] = reason
        rows["policy_id"] = POLICY_ID
        rows["market_only"] = True
        rows["research_only"] = True
        rows["lifecycle_profile"] = profile
        rows["proposal_stage"] = "PRICE_STATE"
        parts.append(rows[_proposal_columns()])
    if not parts:
        return _empty_proposals()
    return pd.concat(parts, ignore_index=True, sort=False)


def confirm_non_eth_leader_regime_v2_price_proposals(
    price_proposals: pd.DataFrame,
    derivative_rows: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2DerivativeConfig | None = None,
) -> pd.DataFrame:
    """用闭合 OI/taker 特征确认价格状态并形成可执行候选。"""

    cfg = config or NonEthLeaderRegimeV2DerivativeConfig()
    proposals = price_proposals.copy()
    if proposals.empty:
        return proposals
    proposals["signal_time"] = pd.to_datetime(
        proposals["signal_time"],
        utc=True,
        errors="raise",
    )
    base = proposals[
        ~proposals["proposal_stage"].eq("PRICE_STATE")
    ].copy()
    states = proposals[
        proposals["proposal_stage"].eq("PRICE_STATE")
    ].copy()
    if states.empty:
        return base.reset_index(drop=True)

    derivatives = _normalize_derivative_rows(derivative_rows)
    states = states.merge(
        derivatives,
        on=["signal_time", "symbol", "factor_family"],
        how="left",
        validate="one_to_one",
    )
    required = [
        "oi_change_3h_pct",
        "oi_change_12h_pct",
        "taker_buy_sell_ratio_1h",
        "taker_buy_sell_ratio_3h",
    ]
    if states[required].isna().any(axis=None):
        raise ValueError("价格状态缺少同一时点的闭合衍生品特征")
    states["_confirmed"] = _derivative_gate(states, config=cfg)
    confirmed = _confirmed_state_onsets(states, config=cfg)
    confirmed = _select_meme_sector_events(confirmed, config=cfg)
    confirmed = _delay_confirmed_execution(
        confirmed,
        feature_rows=feature_rows,
        minutes=cfg.confirmation_delay_minutes,
    )
    confirmed["proposal_stage"] = "DERIVATIVES_CONFIRMED"
    output = pd.concat([base, confirmed], ignore_index=True, sort=False)
    output["policy_id"] = POLICY_ID
    output["signal_id"] = [
        (
            f"{POLICY_ID}|{pd.Timestamp(row.signal_time).value}|"
            f"{row.symbol}|{row.action}|{row.factor_family}"
        )
        for row in output.itertuples(index=False)
    ]
    output = output.sort_values(
        ["signal_time", "symbol", "action", "factor_family"],
        kind="stable",
    ).drop_duplicates(
        ["signal_time", "symbol", "action", "factor_family"],
        keep="first",
    )
    if output["signal_id"].duplicated().any():
        raise ValueError("最新制度确认信号 ID 必须唯一")
    return output.reset_index(drop=True)


def _derivative_gate(
    states: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2DerivativeConfig,
) -> pd.Series:
    family = states["factor_family"]
    minimum_oi = states[
        ["oi_change_3h_pct", "oi_change_12h_pct"]
    ].min(axis=1)
    meme_long = (
        family.eq("meme_capitulation_long")
        & minimum_oi.le(config.meme_long_maximum_oi_change_pct)
        & states["taker_buy_sell_ratio_3h"].le(
            config.meme_long_maximum_taker_ratio_3h,
        )
    )
    meme_short = (
        family.eq("meme_crowded_rollover_short")
        & states["oi_change_3h_pct"].ge(
            config.meme_short_minimum_oi_change_3h_pct,
        )
        & states["taker_buy_sell_ratio_1h"].ge(
            config.meme_short_minimum_taker_ratio_1h,
        )
    )
    payment_short = (
        family.eq("payment_crowded_momentum_short")
        & states["oi_change_3h_pct"].ge(
            config.payment_short_minimum_oi_change_3h_pct,
        )
        & states["taker_buy_sell_ratio_3h"].ge(
            config.payment_short_minimum_taker_ratio_3h,
        )
    )
    rwa_add = (
        family.isin(
            [
                "rwa_shock_deleveraging_add",
                "rwa_recovery_deleveraging_add",
            ],
        )
        & states["oi_change_3h_pct"].le(
            config.rwa_add_maximum_oi_change_3h_pct,
        )
    )
    payment_add = (
        family.eq("payment_deleveraging_recovery_add")
        & states["oi_change_3h_pct"].le(
            config.payment_add_maximum_oi_change_3h_pct,
        )
        & states["taker_buy_sell_ratio_3h"].le(
            config.payment_add_maximum_taker_ratio_3h,
        )
    )
    return meme_long | meme_short | payment_short | rwa_add | payment_add


def _confirmed_state_onsets(
    states: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2DerivativeConfig,
) -> pd.DataFrame:
    rearm_hours = {
        "meme_capitulation_long": (
            config.meme_long_sector_rearm_hours
        ),
        "meme_crowded_rollover_short": (
            config.meme_short_rearm_hours
        ),
        "payment_crowded_momentum_short": (
            config.payment_short_rearm_hours
        ),
        "rwa_shock_deleveraging_add": config.rwa_add_rearm_hours,
        "rwa_recovery_deleveraging_add": config.rwa_add_rearm_hours,
        "payment_deleveraging_recovery_add": (
            config.payment_add_rearm_hours
        ),
    }
    ordered = states.sort_values(
        ["factor_family", "symbol", "signal_time"],
        kind="stable",
    )
    records: list[dict[str, object]] = []
    for (family, _symbol), group in ordered.groupby(
        ["factor_family", "symbol"],
        sort=False,
    ):
        previous_time: pd.Timestamp | None = None
        previous_confirmed = False
        last_signal: pd.Timestamp | None = None
        rearm = pd.Timedelta(hours=rearm_hours[str(family)])
        for row in group.to_dict(orient="records"):
            timestamp = pd.Timestamp(row["signal_time"])
            confirmed = bool(row["_confirmed"])
            continuous = (
                previous_time is not None
                and timestamp - previous_time == pd.Timedelta(minutes=5)
            )
            onset = confirmed and (
                not continuous or not previous_confirmed
            )
            if onset and (
                last_signal is None or timestamp - last_signal >= rearm
            ):
                records.append(row)
                last_signal = timestamp
            previous_time = timestamp
            previous_confirmed = confirmed
    if not records:
        return states.iloc[0:0].copy()
    return pd.DataFrame.from_records(records, columns=states.columns)


def _select_meme_sector_events(
    signals: pd.DataFrame,
    *,
    config: NonEthLeaderRegimeV2DerivativeConfig,
) -> pd.DataFrame:
    meme = signals[
        signals["factor_family"].eq("meme_capitulation_long")
    ].sort_values("signal_time", kind="stable")
    other = signals[
        ~signals["factor_family"].eq("meme_capitulation_long")
    ].copy()
    if meme.empty:
        return other

    cluster_gap = pd.Timedelta(minutes=config.meme_long_cluster_minutes)
    clusters: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    previous_time: pd.Timestamp | None = None
    for row in meme.to_dict(orient="records"):
        timestamp = pd.Timestamp(row["signal_time"])
        if (
            current
            and previous_time is not None
            and timestamp - previous_time > cluster_gap
        ):
            clusters.append(current)
            current = []
        current.append(row)
        previous_time = timestamp
    if current:
        clusters.append(current)

    winners: list[dict[str, object]] = []
    for cluster in clusters:
        winner = sorted(
            cluster,
            key=lambda row: (
                -float(row["rolling_quote_volume_24h"]),
                pd.Timestamp(row["signal_time"]),
                str(row["symbol"]),
            ),
        )[0]
        winners.append(winner)

    rearm = pd.Timedelta(hours=config.meme_long_sector_rearm_hours)
    accepted: list[dict[str, object]] = []
    last_signal: pd.Timestamp | None = None
    for row in winners:
        timestamp = pd.Timestamp(row["signal_time"])
        if last_signal is None or timestamp - last_signal >= rearm:
            accepted.append(row)
            last_signal = timestamp
    selected = pd.DataFrame.from_records(
        accepted,
        columns=signals.columns,
    )
    return pd.concat([other, selected], ignore_index=True, sort=False)


def _delay_confirmed_execution(
    signals: pd.DataFrame,
    *,
    feature_rows: pd.DataFrame,
    minutes: float,
) -> pd.DataFrame:
    if signals.empty:
        return signals
    delay = pd.Timedelta(minutes=minutes)
    output = signals.copy()
    output["_state_time"] = output["signal_time"]
    open_mask = output["operation_type"].eq("OPEN")
    output.loc[open_mask, "signal_time"] = (
        output.loc[open_mask, "signal_time"] + delay
    )
    market = base_factor._normalize_feature_rows(feature_rows)[
        ["bar_time", "symbol", "close"]
    ].drop_duplicates(["bar_time", "symbol"], keep="last")
    output = output.drop(columns="signal_price").merge(
        market,
        left_on=["signal_time", "symbol"],
        right_on=["bar_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if output["close"].isna().any():
        raise ValueError("衍生品确认延迟后缺少可执行市场价格")
    output["signal_price"] = output["close"]
    return output.drop(columns=["bar_time", "close"])


def _normalize_derivative_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_time",
        "symbol",
        "factor_family",
        "oi_last_time",
        "taker_last_time",
        "oi_change_3h_pct",
        "oi_change_12h_pct",
        "oi_change_24h_pct",
        "taker_buy_sell_ratio_1h",
        "taker_buy_sell_ratio_3h",
        "taker_buy_sell_ratio_12h",
        "market_only",
        "leader_data_used",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError("衍生品确认特征缺列: " + ",".join(missing))
    output = rows.copy()
    for column in ["signal_time", "oi_last_time", "taker_last_time"]:
        output[column] = pd.to_datetime(
            output[column],
            utc=True,
            errors="raise",
        )
    if not output["market_only"].astype(bool).all():
        raise ValueError("衍生品确认必须全部为市场数据")
    if output["leader_data_used"].astype(bool).any():
        raise ValueError("衍生品确认禁止使用带单员数据")
    cutoff = output["signal_time"] - pd.Timedelta(minutes=5)
    if (output["oi_last_time"] > cutoff).any():
        raise ValueError("OI 特征包含决策时点后数据")
    if (output["taker_last_time"] > cutoff).any():
        raise ValueError("taker 特征包含决策时点后数据")
    key = ["signal_time", "symbol", "factor_family"]
    if output.duplicated(key).any():
        raise ValueError("衍生品确认特征键必须唯一")
    return output


def _extension_score(rows: pd.DataFrame, *, family: str) -> pd.Series:
    if family == "meme_capitulation_long":
        return -rows["normalized_return_144"]
    if family == "meme_crowded_rollover_short":
        return rows["normalized_return_144"]
    if family == "payment_crowded_momentum_short":
        return rows["normalized_return_144"] + rows[
            "normalized_return_36"
        ]
    if family in {
        "rwa_shock_deleveraging_add",
        "rwa_recovery_deleveraging_add",
    }:
        return -rows["normalized_return_288"]
    return -rows["normalized_return_36"]


def _apply_payment_campaign_cooldown(
    signals: pd.DataFrame,
    *,
    hours: float,
) -> pd.DataFrame:
    if signals.empty:
        return signals
    output: list[dict[str, object]] = []
    last_campaign: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=hours)
    ordered = signals.sort_values(
        ["signal_time", "symbol", "action"],
        kind="stable",
    )
    for row in ordered.to_dict(orient="records"):
        if str(row["factor_family"]) != "payment_capitulation_campaign":
            output.append(row)
            continue
        timestamp = pd.Timestamp(row["signal_time"])
        if last_campaign is not None and timestamp - last_campaign < cooldown:
            continue
        output.append(row)
        last_campaign = timestamp
    return pd.DataFrame.from_records(output, columns=signals.columns)


def _merge_group(
    market: pd.DataFrame,
    group: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    renamed = group.add_prefix(f"{prefix}_").rename(
        columns={f"{prefix}_bar_time": "bar_time"},
    )
    return market.merge(renamed, on="bar_time", how="left")


def _proposal_columns() -> list[str]:
    return [
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
        "proposal_stage",
        "rolling_quote_volume_24h",
        "normalized_return_3",
        "normalized_return_12",
        "normalized_return_36",
        "normalized_return_144",
        "normalized_return_288",
        "quote_volume_ratio",
        "signed_seller_pressure",
        "range_location_288",
    ]


def _empty_proposals() -> pd.DataFrame:
    return pd.DataFrame(columns=_proposal_columns())


__all__ = [
    "MEME_CAMPAIGN_PROFILE",
    "POLICY_ID",
    "NonEthLeaderRegimeV2DerivativeConfig",
    "NonEthLeaderRegimeV2PriceProposalConfig",
    "confirm_non_eth_leader_regime_v2_price_proposals",
    "generate_non_eth_leader_regime_v2_price_proposals",
]
