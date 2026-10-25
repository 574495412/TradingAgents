"""把反趋势评分转换为带机制说明的标准交易信号。"""

from __future__ import annotations

import pandas as pd

from .policy import CountertrendFactorPolicy
from .scoring import COUNTERTREND_MECHANISMS, build_countertrend_scores


def build_countertrend_candidates(
    market_panel: pd.DataFrame,
    *,
    policy: CountertrendFactorPolicy | None = None,
) -> pd.DataFrame:
    cfg = policy or CountertrendFactorPolicy()
    scores = build_countertrend_scores(market_panel, policy=cfg)
    active = {
        (action, mechanism): False
        for action in ("OPEN_LONG", "OPEN_SHORT")
        for mechanism in COUNTERTREND_MECHANISMS
    }
    last_open_index = -cfg.open_cooldown_bars - 1
    records: list[dict[str, object]] = []
    for bar_index, (signal_time, group) in enumerate(scores.groupby("signal_time", sort=True)):
        rising: list[tuple[float, str, str, pd.Series]] = []
        for _, row in group.iterrows():
            action = str(row["action"])
            for mechanism in COUNTERTREND_MECHANISMS:
                key = (action, mechanism)
                score = float(row[f"{mechanism}_score"])
                threshold = cfg.threshold(mechanism)
                if score < threshold - cfg.release_hysteresis:
                    active[key] = False
                elif score >= threshold:
                    was_active = active[key]
                    active[key] = True
                    if not was_active and mechanism in cfg.enabled_mechanisms:
                        rising.append((score, action, mechanism, row))
        if not rising or bar_index - last_open_index < cfg.open_cooldown_bars:
            continue
        rising.sort(key=lambda item: (-item[0], item[1], item[2]))
        winner = rising[0]
        opposite = [score for score, action, _, _ in rising if action != winner[1]]
        if opposite and winner[0] - max(opposite) < cfg.minimum_directional_edge:
            continue
        score, action, mechanism, row = winner
        records.append(
            {
                "signal_id": (
                    f"countertrend:{pd.Timestamp(signal_time).isoformat()}:"
                    f"{action}:{mechanism}"
                ),
                "source_bar_time": row["source_bar_time"],
                "signal_time": signal_time,
                "symbol": row["symbol"],
                "action": action,
                "mechanism": mechanism,
                "score": score,
                "threshold": cfg.threshold(mechanism),
                "market_data_valid": bool(row["market_data_valid"]),
            },
        )
        last_open_index = bar_index
    return pd.DataFrame.from_records(records)


__all__ = ["build_countertrend_candidates"]
