"""根据反趋势候选信号推进开仓、持有、退出和冷却活动。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from .policy import CountertrendFactorPolicy
from .scoring import build_countertrend_scores


@dataclass(frozen=True, slots=True)
class CountertrendCampaignPolicy:
    maximum_adds: int = 3
    minimum_bars_between_fills: int = 6
    add_distance_atr_multiple: float = 1.0
    recovery_target_pct: float = 0.45
    hard_stop_pct: float = 6.0
    minimum_holding_bars: int = 6
    maximum_holding_bars: int = 288
    post_exit_cooldown_bars: int = 12
    round_trip_cost_bps: float = 14.0

    def with_add_distance_multiplier(self, multiplier: float) -> CountertrendCampaignPolicy:
        if not 0.5 <= multiplier <= 1.5:
            raise ValueError("campaign add-distance multiplier must be in [0.5, 1.5]")
        return replace(
            self,
            add_distance_atr_multiple=self.add_distance_atr_multiple * multiplier,
        )


def evaluate_countertrend_campaigns(
    market_panel: pd.DataFrame,
    *,
    mechanism: str,
    factor_policy: CountertrendFactorPolicy | None = None,
    campaign_policy: CountertrendCampaignPolicy | None = None,
) -> pd.DataFrame:
    """evaluate_countertrend_campaigns：计算可解释的市场结构因子。"""

    factor = factor_policy or CountertrendFactorPolicy(enabled_mechanisms=(mechanism,))
    campaign = campaign_policy or CountertrendCampaignPolicy()
    if mechanism not in factor.enabled_mechanisms:
        raise ValueError("campaign mechanism must be enabled in factor policy")
    if campaign.maximum_adds < 0:
        raise ValueError("campaign maximum adds cannot be negative")
    market = market_panel.sort_values("signal_time", kind="stable").reset_index(drop=True).copy()
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    scores = build_countertrend_scores(market, policy=factor)
    return evaluate_scored_countertrend_campaigns(
        market,
        scores=scores,
        mechanism=mechanism,
        score_threshold=factor.threshold(mechanism),
        release_hysteresis=factor.release_hysteresis,
        minimum_directional_edge=factor.minimum_directional_edge,
        campaign_policy=campaign,
    )


def evaluate_scored_countertrend_campaigns(
    market_panel: pd.DataFrame,
    *,
    scores: pd.DataFrame,
    mechanism: str,
    score_threshold: float,
    release_hysteresis: float,
    minimum_directional_edge: float,
    campaign_policy: CountertrendCampaignPolicy | None = None,
) -> pd.DataFrame:
    """evaluate_scored_countertrend_campaigns：计算可解释的市场结构因子。"""

    campaign = campaign_policy or CountertrendCampaignPolicy()
    if not 0.0 < score_threshold <= 1.0:
        raise ValueError("campaign score threshold must be in (0, 1]")
    if not 0.0 <= release_hysteresis < score_threshold:
        raise ValueError("campaign release hysteresis must be below threshold")
    if minimum_directional_edge < 0.0:
        raise ValueError("campaign minimum directional edge cannot be negative")
    if campaign.maximum_adds < 0:
        raise ValueError("campaign maximum adds cannot be negative")
    required_scores = {"signal_time", "action", f"{mechanism}_score"}
    missing_scores = sorted(required_scores.difference(scores.columns))
    if missing_scores:
        raise ValueError("campaign score frame missing columns: " + ", ".join(missing_scores))
    market = market_panel.sort_values("signal_time", kind="stable").reset_index(drop=True).copy()
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    scored = scores.loc[:, list(required_scores)].copy()
    scored["signal_time"] = pd.to_datetime(scored["signal_time"], utc=True)
    scored[f"{mechanism}_score"] = pd.to_numeric(
        scored[f"{mechanism}_score"],
        errors="coerce",
    ).fillna(0.0)
    if scored.duplicated(["signal_time", "action"]).any():
        raise ValueError("campaign score frame contains duplicate time/action rows")
    unknown_actions = sorted(set(scored["action"].astype(str)).difference({"OPEN_LONG", "OPEN_SHORT"}))
    if unknown_actions:
        raise ValueError("campaign score frame contains unsupported actions: " + ", ".join(unknown_actions))
    score_by_time_action = {
        (pd.Timestamp(row["signal_time"]), str(row["action"])): float(
            row[f"{mechanism}_score"],
        )
        for _, row in scored.iterrows()
    }
    active = {"OPEN_LONG": False, "OPEN_SHORT": False}
    position: dict[str, Any] | None = None
    last_exit_index = -campaign.post_exit_cooldown_bars - 1
    records: list[dict[str, object]] = []
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        rising: dict[str, bool] = {}
        for action in active:
            score = score_by_time_action.get((timestamp, action), 0.0)
            threshold = score_threshold
            rising[action] = False
            if score < threshold - release_hysteresis:
                active[action] = False
            elif score >= threshold:
                rising[action] = not active[action]
                active[action] = True

        if position is not None:
            exit_price, exit_reason = _risk_or_recovery_exit(
                bar,
                position=position,
                bar_index=bar_index,
                policy=campaign,
            )
            opposite = "OPEN_SHORT" if position["action"] == "OPEN_LONG" else "OPEN_LONG"
            bars_held = bar_index - int(position["entry_index"])
            if (
                exit_price is None
                and bars_held >= campaign.minimum_holding_bars
                and rising[opposite]
            ):
                exit_price, exit_reason = float(bar["close"]), "opposite_market_transition"
            if exit_price is None and bars_held >= campaign.maximum_holding_bars:
                exit_price, exit_reason = float(bar["close"]), "maximum_holding_time"
            if exit_price is not None:
                records.append(
                    _closed_record(
                        position,
                        exit_time=timestamp,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        holding_bars=bars_held,
                        round_trip_cost_bps=campaign.round_trip_cost_bps,
                    ),
                )
                position = None
                last_exit_index = bar_index
                continue
            if _should_add(
                bar,
                position=position,
                bar_index=bar_index,
                score=score_by_time_action.get((timestamp, str(position["action"])), 0.0),
                threshold=score_threshold,
                policy=campaign,
            ):
                fill_price = float(bar["close"])
                position["fill_prices"].append(fill_price)
                position["fill_times"].append(timestamp)
                position["last_fill_index"] = bar_index
                position["last_fill_price"] = fill_price
                position["average_entry_price"] = sum(position["fill_prices"]) / len(
                    position["fill_prices"],
                )
            continue

        if bar_index - last_exit_index < campaign.post_exit_cooldown_bars:
            continue
        eligible = [
            (score_by_time_action.get((timestamp, action), 0.0), action)
            for action in active
            if rising[action]
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda item: (-item[0], item[1]))
        score, action = eligible[0]
        if len(eligible) > 1 and score - eligible[1][0] < minimum_directional_edge:
            continue
        fill_price = float(bar["close"])
        position = {
            "signal_time": timestamp,
            "entry_index": bar_index,
            "last_fill_index": bar_index,
            "last_fill_price": fill_price,
            "average_entry_price": fill_price,
            "fill_prices": [fill_price],
            "fill_times": [timestamp],
            "action": action,
            "direction": 1.0 if action == "OPEN_LONG" else -1.0,
            "mechanism": mechanism,
            "entry_score": score,
        }
    if position is not None:
        bar = market.iloc[-1]
        records.append(
            _closed_record(
                position,
                exit_time=pd.Timestamp(bar["signal_time"]),
                exit_price=float(bar["close"]),
                exit_reason="development_window_end",
                holding_bars=len(market) - 1 - int(position["entry_index"]),
                round_trip_cost_bps=campaign.round_trip_cost_bps,
            ),
        )
    return pd.DataFrame.from_records(records)


def _risk_or_recovery_exit(
    bar: pd.Series,
    *,
    position: dict[str, Any],
    bar_index: int,
    policy: CountertrendCampaignPolicy,
) -> tuple[float | None, str]:
    direction = float(position["direction"])
    average = float(position["average_entry_price"])
    stop = average * (1.0 - direction * policy.hard_stop_pct / 100.0)
    if direction > 0.0:
        if float(bar["open"]) <= stop:
            return float(bar["open"]), "gap_hard_stop"
        if float(bar["low"]) <= stop:
            return stop, "intrabar_hard_stop"
    else:
        if float(bar["open"]) >= stop:
            return float(bar["open"]), "gap_hard_stop"
        if float(bar["high"]) >= stop:
            return stop, "intrabar_hard_stop"
    if bar_index - int(position["entry_index"]) < policy.minimum_holding_bars:
        return None, ""
    target = average * (1.0 + direction * policy.recovery_target_pct / 100.0)
    if direction > 0.0 and float(bar["high"]) >= target:
        return target, "recovery_target"
    if direction < 0.0 and float(bar["low"]) <= target:
        return target, "recovery_target"
    return None, ""


def _should_add(
    bar: pd.Series,
    *,
    position: dict[str, Any],
    bar_index: int,
    score: float,
    threshold: float,
    policy: CountertrendCampaignPolicy,
) -> bool:
    if len(position["fill_prices"]) - 1 >= policy.maximum_adds:
        return False
    if bar_index - int(position["last_fill_index"]) < policy.minimum_bars_between_fills:
        return False
    if score < threshold:
        return False
    atr_pct = float(bar["atr14_pct"])
    if not atr_pct > 0.0:
        return False
    direction = float(position["direction"])
    adverse_move = -direction * (
        float(bar["close"]) / float(position["last_fill_price"]) - 1.0
    ) * 100.0
    return adverse_move >= atr_pct * policy.add_distance_atr_multiple


def _closed_record(
    position: dict[str, Any],
    *,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    holding_bars: int,
    round_trip_cost_bps: float,
) -> dict[str, object]:
    average = float(position["average_entry_price"])
    gross = float(position["direction"]) * (exit_price / average - 1.0) * 100.0
    return {
        "entry_time": position["signal_time"],
        "exit_time": exit_time,
        "action": position["action"],
        "mechanism": position["mechanism"],
        "entry_score": position["entry_score"],
        "fill_count": len(position["fill_prices"]),
        "add_count": len(position["fill_prices"]) - 1,
        "fill_times": "|".join(pd.Timestamp(value).isoformat() for value in position["fill_times"]),
        "average_entry_price": average,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_bars": holding_bars,
        "gross_return_pct": gross,
        "net_return_pct": gross - round_trip_cost_bps / 100.0,
    }


__all__ = [
    "CountertrendCampaignPolicy",
    "evaluate_countertrend_campaigns",
    "evaluate_scored_countertrend_campaigns",
]
