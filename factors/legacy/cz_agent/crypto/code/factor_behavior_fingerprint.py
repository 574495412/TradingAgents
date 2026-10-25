"""Leader-labeled OHLCV behavior fingerprint for the Liulian strategy.

Leader rows are labels for calibration only.  The functions here score
candidate trades from as-of OHLCV features and never emit a trade because a
leader event happened at that timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cz.trading.crypto.leader.replica_backtest import build_replica_feature_frame


FINGERPRINT_FEATURES: tuple[str, ...] = (
    "ret1",
    "ret3",
    "ret5",
    "ret15",
    "ret20",
    "ret60",
    "ret240",
    "from_high20",
    "from_high60",
    "from_high240",
    "from_low20",
    "from_low60",
    "from_low240",
    "vol_ratio_med30",
    "vol_ratio_mean30",
    "bar_range_pct",
    "close_pos",
    "atr_pct",
    "ema20_deviation",
    "round_anchor_distance_pct",
    "human_anchor_distance_pct",
    "price_level",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
)


@dataclass(frozen=True)
class LiulianBehaviorFingerprintConfig:
    enabled: bool = True
    min_group_samples: int = 2
    distance_quantile: float = 0.35
    threshold_multiplier: float = 1.25
    max_distance: float = 1.20
    daily_cap: int = 2
    symbol_direction_daily_cap: int = 1
    template_return_weight: float = 0.20
    candidate_distance_weight: float = 0.50
    use_template_hold: bool = True
    max_template_hold_minutes: float = 4320.0


def fit_liulian_behavior_fingerprints(
    labels: pd.DataFrame,
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    config: LiulianBehaviorFingerprintConfig,
) -> dict[tuple[str, str], dict[str, Any]]:
    if labels is None or labels.empty or not bool(config.enabled):
        return {}
    prepared = _prepare_label_features(labels, bars_by_symbol=bars_by_symbol)
    models: dict[tuple[str, str], dict[str, Any]] = {}
    min_samples = max(1, int(config.min_group_samples))
    for (symbol, direction), group in prepared.groupby(["标的", "方向"], dropna=True):
        if len(group) < min_samples:
            continue
        usable = [
            feature
            for feature in FINGERPRINT_FEATURES
            if feature in group and pd.to_numeric(group[feature], errors="coerce").notna().sum() >= min_samples
        ]
        if not usable:
            continue
        values = group[usable].apply(pd.to_numeric, errors="coerce")
        center = values.median()
        scale = (values.quantile(0.75) - values.quantile(0.25)).replace(0.0, np.nan)
        scale = scale.fillna(values.std(ddof=0).replace(0.0, np.nan)).fillna(1.0).replace(0.0, 1.0)
        label_matrix = ((values - center) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        leave_one_out: list[float] = []
        if len(label_matrix) >= 2:
            for idx in range(len(label_matrix)):
                distances = np.sqrt(((label_matrix - label_matrix[idx]) ** 2).mean(axis=1))
                distances = np.delete(distances, idx)
                if len(distances):
                    leave_one_out.append(float(np.min(distances)))
        base_threshold = (
            float(np.nanquantile(leave_one_out, _clip(config.distance_quantile, 0.0, 1.0)))
            if leave_one_out
            else float(config.max_distance)
        )
        threshold = min(float(config.max_distance), max(0.10, base_threshold * float(config.threshold_multiplier)))
        returns = pd.to_numeric(group.get("收益率估算"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        holds = pd.to_numeric(group.get("持仓分钟"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        models[(str(symbol), str(direction))] = {
            "symbol": str(symbol),
            "direction": str(direction),
            "features": usable,
            "center": center,
            "scale": scale,
            "label_matrix": label_matrix,
            "labels": group.reset_index(drop=True),
            "threshold": float(threshold),
            "sample_count": int(len(group)),
            "median_hold_minutes": float(holds.median()) if holds.notna().any() else np.nan,
            "median_return_pct": float(returns.median()) if returns.notna().any() else np.nan,
            "win_rate_pct": float(returns.gt(0.0).mean() * 100.0) if returns.notna().any() else np.nan,
        }
    return models


def apply_liulian_behavior_fingerprint_acceptance(
    candidates: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    config: LiulianBehaviorFingerprintConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates is None or candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    if labels is None or labels.empty or not bool(config.enabled):
        return candidates.copy(), pd.DataFrame()
    models = fit_liulian_behavior_fingerprints(labels, bars_by_symbol=bars_by_symbol, config=config)
    if not models:
        return candidates.copy(), pd.DataFrame()
    scored = _score_candidates(candidates, bars_by_symbol=bars_by_symbol, models=models, config=config)
    accepted = scored[scored["_fingerprint_accepted"].fillna(False)].copy()
    if accepted.empty:
        return accepted, _fingerprint_summary(models, before=len(candidates), after=0)
    accepted = _apply_frequency_caps(accepted, config=config)
    accepted = accepted.drop(columns=[col for col in accepted.columns if col.startswith("_fingerprint_")], errors="ignore")
    return accepted.reset_index(drop=True), _fingerprint_summary(models, before=len(candidates), after=len(accepted))


def _prepare_label_features(labels: pd.DataFrame, *, bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = labels.copy()
    out["标的"] = out.get("标的", "").astype(str).str.upper().str.replace("/", "", regex=False)
    out["方向"] = out.get("方向", "").astype(str)
    out["开仓时间"] = pd.to_datetime(out.get("开仓时间"), errors="coerce")
    for symbol, raw_bars in bars_by_symbol.items():
        mask = out["标的"].eq(str(symbol).upper().replace("/", "")) & out["开仓时间"].notna()
        if not bool(mask.any()):
            continue
        features = _feature_frame(raw_bars)
        if features.empty:
            continue
        for idx in out.index[mask]:
            ts = pd.Timestamp(out.at[idx, "开仓时间"])
            pos = features.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                continue
            row = features.iloc[int(pos)]
            for feature in FINGERPRINT_FEATURES:
                value = _finite(row.get(feature), np.nan)
                if np.isfinite(value):
                    out.at[idx, feature] = value
    return out


def _score_candidates(
    candidates: pd.DataFrame,
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    models: dict[tuple[str, str], dict[str, Any]],
    config: LiulianBehaviorFingerprintConfig,
) -> pd.DataFrame:
    out = candidates.copy()
    feature_frames = {
        str(symbol).upper().replace("/", ""): _feature_frame(raw_bars)
        for symbol, raw_bars in bars_by_symbol.items()
    }
    distances: list[float] = []
    thresholds: list[float] = []
    nearest_returns: list[float] = []
    nearest_holds: list[float] = []
    ranks: list[float] = []
    accepted: list[bool] = []
    for _, row in out.iterrows():
        symbol = str(row.get("标的", "")).upper().replace("/", "")
        direction = str(row.get("方向", ""))
        model = models.get((symbol, direction))
        distance = np.nan
        threshold = np.nan
        nearest_return = np.nan
        nearest_hold = np.nan
        rank = -np.inf
        ok = False
        features = feature_frames.get(symbol)
        if model is not None and features is not None and not features.empty:
            feature_row = _feature_row_at_or_before(features, pd.to_datetime(row.get("开仓时间"), errors="coerce"))
            if feature_row is not None:
                distance, nearest_idx = _nearest_distance(feature_row, model)
                threshold = _finite(model.get("threshold"), np.nan)
                ok = np.isfinite(distance) and np.isfinite(threshold) and distance <= threshold
                labels = model.get("labels")
                if labels is not None and len(labels) and nearest_idx >= 0:
                    nearest = labels.iloc[int(nearest_idx)]
                    nearest_return = _finite(nearest.get("收益率估算"), np.nan)
                    nearest_hold = _finite(nearest.get("持仓分钟"), np.nan)
                candidate_distance = _finite(row.get("模板距离"), 0.0)
                rank = (
                    _finite(nearest_return, 0.0) * float(config.template_return_weight)
                    - _finite(distance, 99.0)
                    - candidate_distance * float(config.candidate_distance_weight)
                )
        distances.append(distance)
        thresholds.append(threshold)
        nearest_returns.append(nearest_return)
        nearest_holds.append(nearest_hold)
        ranks.append(rank)
        accepted.append(ok)
    out["_fingerprint_distance"] = distances
    out["_fingerprint_threshold"] = thresholds
    out["_fingerprint_template_return"] = nearest_returns
    out["_fingerprint_template_hold"] = nearest_holds
    out["_fingerprint_rank"] = ranks
    out["_fingerprint_accepted"] = accepted
    trigger = out.get("触发特征", pd.Series("", index=out.index)).astype(str)
    out["触发特征"] = [
        f"{text};behavior_fingerprint_dist={dist:.3f};behavior_fingerprint_thr={thr:.3f}"
        if np.isfinite(dist) and np.isfinite(thr)
        else f"{text};behavior_fingerprint_unavailable"
        for text, dist, thr in zip(trigger, distances, thresholds)
    ]
    out["信号来源"] = "OHLCV_market_structure_rules+behavior_fingerprint"
    out["leader用途"] = "labels_for_fingerprint_calibration"
    if bool(config.use_template_hold):
        _apply_template_hold_exits(out, max_hold_minutes=float(config.max_template_hold_minutes))
    return out


def _apply_frequency_caps(candidates: pd.DataFrame, *, config: LiulianBehaviorFingerprintConfig) -> pd.DataFrame:
    daily_cap = max(0, int(config.daily_cap))
    symbol_direction_cap = max(0, int(config.symbol_direction_daily_cap))
    if daily_cap <= 0 and symbol_direction_cap <= 0:
        return candidates.sort_values(["开仓时间", "标的", "方向", "_fingerprint_rank"], ascending=[True, True, True, False])
    work = candidates.copy()
    work["_fingerprint_day"] = pd.to_datetime(work.get("开仓时间"), errors="coerce").dt.date
    work = work.sort_values(["_fingerprint_day", "_fingerprint_rank", "开仓时间"], ascending=[True, False, True])
    rows: list[pd.Series] = []
    for day, group in work.groupby("_fingerprint_day", dropna=False):
        picked = 0
        per_key: dict[tuple[str, str], int] = {}
        for _, row in group.iterrows():
            key = (str(row.get("标的", "")), str(row.get("方向", "")))
            if symbol_direction_cap > 0 and per_key.get(key, 0) >= symbol_direction_cap:
                continue
            if daily_cap > 0 and picked >= daily_cap:
                break
            rows.append(row)
            picked += 1
            per_key[key] = per_key.get(key, 0) + 1
    if not rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(rows).sort_values(["开仓时间", "标的", "方向"]).reset_index(drop=True)


def _apply_template_hold_exits(candidates: pd.DataFrame, *, max_hold_minutes: float) -> None:
    open_times = pd.to_datetime(candidates.get("开仓时间"), errors="coerce")
    holds = pd.to_numeric(candidates.get("_fingerprint_template_hold"), errors="coerce")
    cap = max(5.0, float(max_hold_minutes))
    valid = open_times.notna() & holds.notna() & holds.gt(0.0)
    if not bool(valid.any()):
        return
    clipped = holds.clip(lower=5.0, upper=cap)
    candidates.loc[valid, "平仓时间"] = open_times[valid] + pd.to_timedelta(clipped[valid], unit="m")
    candidates.loc[valid, "持仓分钟"] = clipped[valid].astype(float)
    candidates.loc[valid, "模板持仓分钟"] = clipped[valid].astype(float)
    candidates.loc[valid, "平仓原因"] = "行为指纹持仓风控"
    template_returns = pd.to_numeric(candidates.get("_fingerprint_template_return"), errors="coerce")
    candidates.loc[valid & template_returns.notna(), "模板收益率%"] = template_returns[valid & template_returns.notna()]


def _fingerprint_summary(models: dict[tuple[str, str], dict[str, Any]], *, before: int, after: int) -> pd.DataFrame:
    rows = []
    for (symbol, direction), model in sorted(models.items()):
        rows.append(
            {
                "策略类型": "autonomous_factor_rule",
                "信号来源": "OHLCV_market_structure_rules+behavior_fingerprint",
                "带单事件作为信号": False,
                "leader用途": "labels_for_fingerprint_calibration",
                "标的": symbol,
                "方向": direction,
                "训练样本": int(model.get("sample_count", 0)),
                "模型样本": int(model.get("sample_count", 0)),
                "特征数": int(len(model.get("features", []))),
                "因子": "behavior_fingerprint",
                "距离阈值": _finite(model.get("threshold"), np.nan),
                "目标收益%": _finite(model.get("median_return_pct"), np.nan),
                "长持仓分钟": _finite(model.get("median_hold_minutes"), np.nan),
                "模型": "带单员行为指纹接受层",
                "候选数_过滤前": int(before),
                "候选数_过滤后": int(after),
            }
        )
    return pd.DataFrame(rows)


def _feature_frame(raw_bars: pd.DataFrame | None) -> pd.DataFrame:
    if raw_bars is None or raw_bars.empty or "close" not in raw_bars:
        return pd.DataFrame()
    frame = build_replica_feature_frame(raw_bars)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()].sort_index()
    return frame


def _feature_row_at_or_before(features: pd.DataFrame, ts: pd.Timestamp | pd.NaT) -> pd.Series | None:
    if pd.isna(ts):
        return None
    if features is None or features.empty:
        return None
    pos = features.index.searchsorted(pd.Timestamp(ts), side="right") - 1
    if pos < 0:
        return None
    return features.iloc[int(pos)]


def _nearest_distance(row: pd.Series, model: dict[str, Any]) -> tuple[float, int]:
    features = list(model.get("features") or [])
    if not features:
        return np.nan, -1
    values = pd.Series({feature: _finite(row.get(feature), np.nan) for feature in features}, dtype=float)
    center = pd.to_numeric(model["center"].reindex(features), errors="coerce")
    scale = pd.to_numeric(model["scale"].reindex(features), errors="coerce").replace(0.0, 1.0)
    z = ((values - center) / scale).replace([np.inf, -np.inf], np.nan).fillna(999.0).to_numpy(dtype=float)
    label_matrix = np.asarray(model.get("label_matrix"), dtype=float)
    if label_matrix.size == 0:
        return np.nan, -1
    distances = np.sqrt(((label_matrix - z) ** 2).mean(axis=1))
    nearest = int(np.nanargmin(distances))
    return float(distances[nearest]), nearest


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
