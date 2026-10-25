"""固定因子方向的分层、信息与持久性诊断，不模拟账户或授予生产资格。"""

from collections.abc import Sequence
from itertools import pairwise
from numbers import Integral

import numpy as np
import pandas as pd

from .regression import correlation, daily_stat

VERSION = "ta-factor-diagnostics.v1"
IDENTITY = ("asset_class", "clock_ms", "symbol")
HOUR = 3_600_000


def _clock(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name}必须为整数毫秒")
    return int(value)


def _numeric(frame, column):
    if column in frame and len(frame) == 0:
        return np.empty(0, dtype=float)
    if (
        column not in frame
        or not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ):
        raise ValueError(f"{column}必须为数值列，未知可以保留NaN")
    return frame[column].to_numpy(dtype=float, na_value=np.nan)


def _correlation(left, right, *, rank=False):
    """有限大数先缩放，避免信息指标因平方溢出变成未声明的NaN。"""
    if rank:
        return correlation(left, right, rank=True)
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) == 0 or len(b) != len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    a_scale, b_scale = np.max(np.abs(a)), np.max(np.abs(b))
    if a_scale == 0 or b_scale == 0:
        return None
    value = correlation(a / a_scale, b / b_scale)
    return value if value is not None and np.isfinite(value) else None


def _mean_summary(values):
    known = [value for value in values if value is not None and np.isfinite(value)]
    return {"valid_periods": len(known), "mean": float(np.mean(known)) if known else None}


def assign_memberships(
    frame,
    *,
    score_column,
    calendar_ms: Sequence[int],
    common_features=(),
    direction=1,
    asset_classes=None,
    frequency_ms=HOUR,
    minimum_members=20,
):
    """仅由已知特征定人口及分位，未来标签不属于本接口的输入参数。

    同值整体保留：q=floor(5*(平均秩−0.5)/N)+1，空桶不强补成员。
    显式共同特征人口内只要有未知score，整截面不生成分位，避免模型间缩样本。
    """
    frequency = _clock(frequency_ms, "采样间隔")
    clocks = tuple(_clock(value, "计划时钟") for value in calendar_ms)
    if frequency <= 0 or not clocks or any(b - a != frequency for a, b in pairwise(clocks)):
        raise ValueError("计划日历必须非空、唯一递增且保持固定采样间隔")
    direction = _clock(direction, "因子方向")
    if direction not in {-1, 1}:
        raise ValueError("因子方向必须事前固定为+1或−1")
    if (
        isinstance(minimum_members, bool)
        or not isinstance(minimum_members, Integral)
        or minimum_members < 3
    ):
        raise ValueError("最少共同成员数必须为至少3的整数")
    population_columns = tuple(common_features) or (score_column,)
    if len(set(population_columns)) != len(population_columns):
        raise ValueError("共同特征不能重复")
    if frame.columns.has_duplicates or not set(IDENTITY) <= set(frame.columns):
        raise ValueError("输入必须包含资产类别、整数时钟和合约身份")
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("输入存在重复资产类别、时钟、合约")
    for name in ("asset_class", "symbol"):
        if any(not isinstance(value, str) or not value.strip() for value in frame[name]):
            raise ValueError("资产类别和合约身份必须为非空文本")
    if frame.groupby("symbol").asset_class.nunique().gt(1).any():
        raise ValueError("同一合约不能在研究区间内变更资产类别")
    observed_clocks = [_clock(value, "样本时钟") for value in frame.clock_ms]
    if not set(observed_clocks) <= set(clocks):
        raise ValueError("样本时钟不属于明确计划日历")
    assets = (
        tuple(asset_classes)
        if asset_classes is not None
        else tuple(sorted(frame.asset_class.unique()))
    )
    if not assets or len(set(assets)) != len(assets) or not set(frame.asset_class) <= set(assets):
        raise ValueError("必须声明非空唯一资产类别全集，不能漏掉输入资产")
    scores = _numeric(frame, score_column)
    population = np.isfinite(
        np.column_stack([_numeric(frame, name) for name in population_columns])
    ).all(axis=1)
    members = frame[list(IDENTITY)].copy().reset_index(drop=True)
    members["score"], members["oriented_score"] = scores, scores * direction
    members["population_member"] = population
    for name in ("rank", "rank_percentile", "quantile"):
        members[name] = np.nan
    groups = members.groupby(["asset_class", "clock_ms"], sort=False).groups
    sections = []
    for asset in assets:
        for clock in clocks:
            indices = groups.get((asset, clock), [])
            part = members.loc[indices]
            known_indices = part.index[part.population_member]
            known = members.loc[known_indices, "oriented_score"]
            status = "READY"
            if not len(part):
                status = "MISSING_SECTION"
            elif len(known) < minimum_members:
                status = "INSUFFICIENT_COMMON_MEMBERS"
            elif not np.isfinite(known).all():
                status = "INCOMPLETE_SCORE_ON_COMMON_POPULATION"
            else:
                ranks = known.rank(method="average")
                percentile = (ranks - 0.5) / len(known)
                members.loc[known_indices, "rank"] = ranks
                members.loc[known_indices, "rank_percentile"] = percentile
                members.loc[known_indices, "quantile"] = np.floor(5 * percentile) + 1
                if known.nunique() == 1:
                    status = "CONSTANT_SCORE"
            quantiles = members.loc[known_indices, "quantile"]
            counts = {str(q): int(quantiles.eq(q).sum()) for q in range(1, 6)}
            sections.append(
                {
                    "asset_class": asset,
                    "clock_ms": clock,
                    "status": status,
                    "scheduled_members": len(part),
                    "common_feature_members": len(known),
                    "unknown_population_members": len(part) - len(known),
                    "finite_score_members": int(np.isfinite(known).sum()),
                    "distinct_scores": int(known[np.isfinite(known)].nunique()),
                    "maximum_tie_size": int(known.value_counts().max())
                    if len(known.dropna())
                    else 0,
                    "quantile_counts": counts,
                    "empty_quantiles": [q for q in range(1, 6) if counts[str(q)] == 0],
                }
            )
    return members.sort_values(list(IDENTITY)).reset_index(drop=True), sections


def _ratio(values):
    valid = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)], dtype=float
    )
    deviation = float(valid.std(ddof=1)) if len(valid) > 1 else None
    return {
        "valid_periods": len(valid),
        "mean": float(valid.mean()) if len(valid) else None,
        "standard_deviation_ddof1": deviation,
        "icir_unannualized": float(valid.mean() / deviation)
        if deviation is not None and deviation > 0
        else None,
        "positive_fraction": float((valid > 0).mean()) if len(valid) else None,
        "annualization_applied": False,
    }


def evaluate(
    frame,
    *,
    score_column,
    target_column,
    calendar_ms,
    scope,
    common_features=(),
    direction=1,
    asset_classes=None,
    frequency_ms=HOUR,
    lag_ms=(HOUR,),
    minimum_members=20,
):
    """消费一个明确阶段的样本；调用方须先冻结方向、标签与阶段边界。

    原收益和共同人口截面去均值收益分列，后者不是对冲账户收益。
    换手与秩自相关只查精确t−lag，不把缺时前后的记录压缩成相邻。
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("必须明确本次诊断的阶段范围")
    frequency_ms = _clock(frequency_ms, "采样间隔")
    if frequency_ms <= 0:
        raise ValueError("采样间隔必须为正")
    calendar_ms = tuple(calendar_ms)
    common_features = tuple(common_features)
    direction = _clock(direction, "因子方向")
    if target_column == score_column or target_column in common_features:
        raise ValueError("未来收益标签不能作为分位score或共同人口特征")
    targets = _numeric(frame, target_column)
    lags = tuple(_clock(value, "持久性间隔") for value in lag_ms)
    if (
        not lags
        or len(set(lags)) != len(lags)
        or any(value <= 0 or value % frequency_ms for value in lags)
    ):
        raise ValueError("持久性间隔必须唯一、为采样周期的正整数倍")
    members, sections = assign_memberships(
        frame,
        score_column=score_column,
        calendar_ms=calendar_ms,
        common_features=common_features,
        direction=direction,
        asset_classes=asset_classes,
        frequency_ms=frequency_ms,
        minimum_members=minimum_members,
    )
    labels = frame[list(IDENTITY)].copy()
    labels["target"] = targets
    evaluated = members.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    member_groups = dict(iter(evaluated.groupby(["asset_class", "clock_ms"], sort=False)))
    lookup = {(row["asset_class"], row["clock_ms"]): row for row in sections}
    ready_statuses = {"READY", "CONSTANT_SCORE"}
    for row in sections:
        asset, clock = row["asset_class"], row["clock_ms"]
        part = member_groups.get((asset, clock), evaluated.iloc[:0])
        population = part[part.population_member]
        usable = row["status"] in ready_statuses
        paired = population[np.isfinite(population.target) & np.isfinite(population.oriented_score)]
        universe_complete = usable and len(population) > 0 and np.isfinite(population.target).all()
        universe_mean = float(population.target.mean()) if universe_complete else None
        row.update(
            day=pd.Timestamp(clock, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
            label_known_members=int(np.isfinite(population.target).sum()),
            paired_ic_members=len(paired),
            universe_raw_mean=universe_mean,
            pearson_ic=_correlation(paired.oriented_score, paired.target)
            if usable and len(paired) >= minimum_members
            else None,
            rank_ic=_correlation(paired.oriented_score, paired.target, rank=True)
            if usable and len(paired) >= minimum_members
            else None,
        )
        row["ic_status"] = (
            "UNAVAILABLE_MEMBERSHIP"
            if not usable
            else "INSUFFICIENT_PAIRED_MEMBERS"
            if len(paired) < minimum_members
            else "CONSTANT_SCORE_OR_TARGET"
            if row["rank_ic"] is None
            else "READY"
        )
        quantiles = {}
        for q in range(1, 6):
            group = population[population["quantile"] == q]
            complete = usable and len(group) > 0 and np.isfinite(group.target).all()
            raw = float(group.target.mean()) if complete else None
            quantiles[str(q)] = {
                "members": len(group),
                "known_labels": int(np.isfinite(group.target).sum()),
                "raw_mean_return": raw,
                "cross_section_demeaned_mean_return": raw - universe_mean
                if raw is not None and universe_mean is not None
                else None,
                "status": "UNAVAILABLE_MEMBERSHIP"
                if not usable
                else "EMPTY_QUANTILE"
                if not len(group)
                else "SELECTED_TARGET_MISSING"
                if not complete
                else "READY",
            }
        row["quantiles"] = quantiles
        row["spreads"] = {}
        for name in ("raw_mean_return", "cross_section_demeaned_mean_return"):
            low, high = quantiles["1"][name], quantiles["5"][name]
            spread = high - low if high is not None and low is not None else None
            row["spreads"][name] = {
                "q5_minus_q1": spread,
                "half_gross_long_short_diagnostic": spread / 2 if spread is not None else None,
            }
        persistence = {}
        for lag in lags:
            previous = lookup.get((asset, clock - lag))
            prior = member_groups.get((asset, clock - lag), evaluated.iloc[:0])
            status = (
                "MISSING_LAG_CLOCK"
                if previous is None
                else "UNAVAILABLE_MEMBERSHIP"
                if not usable or previous["status"] not in ready_statuses
                else "READY"
            )
            current_ids = set(population.symbol)
            prior_population = prior[prior.population_member]
            previous_ids = set(prior_population.symbol)
            common = sorted(current_ids & previous_ids)
            value = {
                "status": status,
                "common_members": len(common),
                "rank_autocorrelation": None,
                "quantile_turnover": {},
            }
            if status == "READY" and len(common) >= minimum_members:
                left = population.set_index("symbol").loc[common, "rank"]
                right = prior_population.set_index("symbol").loc[common, "rank"]
                value["rank_autocorrelation"] = _correlation(left, right)
                if value["rank_autocorrelation"] is None:
                    value["status"] = "CONSTANT_INTERSECTION_RANKS"
            elif status == "READY":
                value["status"] = "INSUFFICIENT_COMMON_MEMBERS"
            for q in range(1, 6):
                current_q = set(population.loc[population["quantile"] == q, "symbol"])
                previous_q = set(prior_population.loc[prior_population["quantile"] == q, "symbol"])
                available = status == "READY"
                additions, removals = current_q - previous_q, previous_q - current_q
                value["quantile_turnover"][str(q)] = {
                    "current_members": len(current_q),
                    "previous_members": len(previous_q),
                    "entry_fraction": len(additions) / len(current_q)
                    if available and current_q
                    else None,
                    "exit_fraction": len(removals) / len(previous_q)
                    if available and previous_q
                    else None,
                    "entries_from_new_population": len(additions - previous_ids)
                    if available
                    else None,
                    "entries_from_reranking": len(additions & previous_ids) if available else None,
                    "status": status
                    if not available
                    else "EMPTY_CURRENT_QUANTILE"
                    if not current_q
                    else "READY",
                }
            persistence[str(lag)] = value
        row["persistence"] = persistence
    summaries = {}
    for asset in sorted({row["asset_class"] for row in sections}):
        rows = [row for row in sections if row["asset_class"] == asset]
        days = (
            pd.date_range(
                pd.Timestamp(min(calendar_ms), unit="ms", tz="UTC").normalize(),
                pd.Timestamp(max(calendar_ms), unit="ms", tz="UTC").normalize(),
                freq="D",
            )
            .strftime("%Y-%m-%d")
            .tolist()
        )
        statistics = {}
        for metric in ("pearson_ic", "rank_ic"):
            daily = daily_stat(rows, lambda row, name=metric: row[name], days)
            statistics[metric] = {
                "period": {"period_ms": int(frequency_ms), **_ratio([row[metric] for row in rows])},
                "daily": {
                    "period_ms": 86_400_000,
                    **_ratio(list(daily["daily_values"].values())),
                    "hac_mean": daily,
                },
            }
        summaries[asset] = {
            "scheduled_periods": len(rows),
            "information": statistics,
            "quantile_returns": {
                str(q): {
                    metric: _mean_summary([row["quantiles"][str(q)][metric] for row in rows])
                    for metric in ("raw_mean_return", "cross_section_demeaned_mean_return")
                }
                for q in range(1, 6)
            },
            "spreads": {
                metric: {
                    measure: _mean_summary([row["spreads"][metric][measure] for row in rows])
                    for measure in ("q5_minus_q1", "half_gross_long_short_diagnostic")
                }
                for metric in ("raw_mean_return", "cross_section_demeaned_mean_return")
            },
            "persistence": {
                str(lag): {
                    "rank_autocorrelation": _mean_summary(
                        [row["persistence"][str(lag)]["rank_autocorrelation"] for row in rows]
                    ),
                    "quantile_turnover": {
                        str(q): {
                            metric: _mean_summary(
                                [
                                    row["persistence"][str(lag)]["quantile_turnover"][str(q)][
                                        metric
                                    ]
                                    for row in rows
                                ]
                            )
                            for metric in ("entry_fraction", "exit_fraction")
                        }
                        for q in range(1, 6)
                    },
                }
                for lag in lags
            },
            "return_and_persistence_summary_weights": "equal_known_sampling_periods",
        }
    return {
        "schema": VERSION,
        "scope": scope,
        "protocol": {
            "score_column": score_column,
            "target_column": target_column,
            "population_columns": list(tuple(common_features) or (score_column,)),
            "direction": direction,
            "frequency_ms": int(frequency_ms),
            "lag_ms": list(lags),
            "minimum_members": int(minimum_members),
            "quantiles": 5,
            "icir_standard_deviation_ddof": 1,
            "daily_hac_lag_days": 3,
            "tie_policy": "average_rank_midpoint_keep_equal_values_together",
            "membership_uses_future_labels": False,
            "ic_uses_finite_label_pairs_without_changing_membership": True,
            "rank_persistence_uses_full_cross_section_ranks_then_identity_intersection": True,
            "missing_calendar_periods_are_not_compressed": True,
            "scope_is_caller_supplied_not_causal_validation_proof": True,
        },
        "sections": sections,
        "summary": summaries,
        "returns_are_diagnostic_labels_not_account_equity": True,
        "costs_or_funding_deducted": False,
        "economic_account_evaluated": False,
        "composite_score_created": False,
        "new_significance_tests_declared": False,
        "economic_gate_changed": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
