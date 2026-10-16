"""固定截面因子的预测评分与量化回归，不产生交易指令。"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

FACTORS = ("return_3m", "return_20m", "return_60m", "flow_3m", "flow_12m")
CONTROLS = ("volatility_60m", "log_quote_volume_60m")
FEATURES = FACTORS + CONTROLS
MIN_MEMBERS = 20


def correlation(x, y, *, rank=False):
    """常量或有效成员不足时保留未知，不用零伪装没有预测力。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if rank:
        x, y = (
            pd.Series(x).rank(method="average").to_numpy(),
            pd.Series(y).rank(method="average").to_numpy(),
        )
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def hac_mean(values, lag=3):
    """固定日历序列的Bartlett均值标准误；缺日不压缩成相邻观测。"""
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {
            "mean": None,
            "days": 0,
            "standard_error": None,
            "t": None,
            "p_asymptotic_two_sided": None,
        }
    valid = np.isfinite(x)
    n = int(valid.sum())
    if not n:
        return {
            "mean": None,
            "days": 0,
            "standard_error": None,
            "t": None,
            "p_asymptotic_two_sided": None,
        }
    mean = float(x[valid].mean())
    errors = np.where(valid, x - mean, 0.0)
    meat = float(errors @ errors)
    for offset in range(1, min(lag, len(x) - 1) + 1):
        meat += 2 * (1 - offset / (lag + 1)) * float(errors[offset:] @ errors[:-offset])
    standard_error = math.sqrt(max(0.0, meat)) / n if n >= lag + 2 else None
    t = mean / standard_error if standard_error is not None and standard_error > 0 else None
    return {
        "mean": mean,
        "days": n,
        "standard_error": standard_error,
        "t": t,
        "p_asymptotic_two_sided": math.erfc(abs(t) / math.sqrt(2)) if t is not None else None,
        "hac_lag_days": lag,
        "small_sample_exploratory_only": True,
    }


def bh_adjust(values):
    """所有已声明检验一起校正，未知p值保留未知。"""
    valid = sorted((float(p), index) for index, p in enumerate(values) if p is not None)
    result, previous = [None] * len(values), 1.0
    for position in range(len(valid) - 1, -1, -1):
        p, index = valid[position]
        if not 0 <= p <= 1:
            raise ValueError("多重检验p值不在单位区间")
        previous = min(previous, p * len(valid) / (position + 1))
        result[index] = previous
    return result


def ranked(frame):
    """只由当前完整特征成员决定截面变换，与标签是否缺失无关。"""
    if len(frame) < MIN_MEMBERS:
        return None
    return (frame[list(FEATURES)].rank(method="average") - 1) / (len(frame) - 1) - 0.5


def cross_section(frame, target):
    """先定已知特征全集，再检验标签覆盖；多元矩阵奇异时不临时删因子。"""
    if frame["symbol"].duplicated().any():
        raise ValueError("同一截面出现重复合约")
    known = frame[np.isfinite(frame[list(FEATURES)]).all(axis=1)].sort_values("symbol").copy()
    transformed = ranked(known)
    record = {
        "members_scheduled": len(frame),
        "known_feature_members": len(known),
        "known_target_members": int(np.isfinite(known[target]).sum()),
        "factor_statistics": {},
        "multivariate": None,
        "multivariate_failure": None,
    }
    if transformed is None:
        record["multivariate_failure"] = "INSUFFICIENT_FEATURE_MEMBERS"
        return record, known, transformed
    usable = np.isfinite(known[target]).to_numpy()
    if usable.sum() < MIN_MEMBERS:
        record["multivariate_failure"] = "INSUFFICIENT_TARGET_MEMBERS"
        return record, known, transformed
    y = known.loc[usable, target].to_numpy(dtype=float)
    for factor in FACTORS:
        x = transformed.loc[usable, factor].to_numpy()
        matrix = np.column_stack([np.ones(len(x)), x])
        coefficient = (
            None
            if np.linalg.matrix_rank(matrix) < 2
            else float(np.linalg.lstsq(matrix, y, rcond=None)[0][1])
        )
        record["factor_statistics"][factor] = {
            "pearson_ic": correlation(known.loc[usable, factor], y),
            "rank_ic": correlation(x, y, rank=True),
            "coefficient": coefficient,
        }
    x = np.column_stack([np.ones(int(usable.sum())), transformed.loc[usable].to_numpy()])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        record["multivariate_failure"] = "RANK_DEFICIENT_FIXED_DESIGN"
    else:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        record["multivariate"] = {
            name: float(value) for name, value in zip(("intercept",) + FEATURES, beta, strict=True)
        }
    return record, known, transformed


def diagnostic_spread(known, scores, target, direction):
    """未来标签不能改变选组；任何已选成员缺标签就让整个截面收益未知。"""
    if scores is None or len(known) < MIN_MEMBERS or direction not in {-1, 1}:
        return {
            "gross": None,
            "base_cost_sensitivity": None,
            "stress_cost_sensitivity": None,
            "reason": "UNAVAILABLE_SCORE_OR_DIRECTION",
        }
    selection = known[["symbol", target]].copy()
    selection["score"] = np.asarray(scores, dtype=float)
    selection = selection.sort_values(["score", "symbol"])
    size = max(1, len(selection) // 5)
    low, high = selection.iloc[:size], selection.iloc[-size:]
    if not np.isfinite(low[target]).all() or not np.isfinite(high[target]).all():
        return {
            "gross": None,
            "base_cost_sensitivity": None,
            "stress_cost_sensitivity": None,
            "reason": "SELECTED_FUTURE_TARGET_MISSING",
            "selected_per_side": size,
        }
    gross = direction * (float(high[target].mean()) - float(low[target].mean())) / 2
    return {
        "gross": gross,
        "base_cost_sensitivity": gross - 0.0016,
        "stress_cost_sensitivity": gross - 0.0030,
        "selected_per_side": size,
        "reason": None,
    }


def daily_stat(rows, accessor, days):
    """先按UTC日等权，再估计跨日均值，避免把合约行数当作独立时间样本。"""
    grouped = {day: [] for day in days}
    for row in rows:
        value = accessor(row)
        if value is not None and math.isfinite(value):
            grouped[row["day"]].append(value)
    values = [float(np.mean(grouped[day])) if grouped[day] else math.nan for day in days]
    result = hac_mean(values)
    result["daily_values"] = {
        day: value if math.isfinite(value) else None
        for day, value in zip(days, values, strict=True)
    }
    valid = np.asarray([v for v in values if math.isfinite(v)])
    result["daily_mean_over_standard_deviation"] = (
        float(valid.mean() / valid.std(ddof=1))
        if len(valid) > 1 and valid.std(ddof=1) > 0
        else None
    )
    return result


def evaluate(samples, protocol):
    """训练方向及均值系数冻结，验证和后续日不反转、不重估、不拼接赢家。"""
    records, contexts = [], {}
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        for horizon in (20, 60):
            phase = frame[f"phase_{horizon}"].iloc[0]
            if phase == "excluded_boundary":
                continue
            value, known, transformed = cross_section(frame, f"target_{horizon}")
            value.update(
                asset_class=asset,
                clock_ms=int(clock),
                day=pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                horizon_minutes=horizon,
                phase=phase,
            )
            key = asset, int(clock), horizon
            contexts[key] = known, transformed
            records.append(value)
    groups, tests = {}, []
    for asset in sorted(samples["asset_class"].unique()):
        for horizon in (20, 60):
            subset = [
                r for r in records if r["asset_class"] == asset and r["horizon_minutes"] == horizon
            ]
            training = [r for r in subset if r["phase"] == "training"]
            directions = {}
            for factor in FACTORS:
                values = [r["factor_statistics"].get(factor, {}).get("rank_ic") for r in training]
                observed = [v for v in values if v is not None]
                average = float(np.mean(observed)) if observed else 0.0
                directions[factor] = 1 if average > 0 else -1 if average < 0 else 0
            fitted = [r["multivariate"] for r in training if r["multivariate"] is not None]
            mean_beta = (
                {f: float(np.mean([v[f] for v in fitted])) for f in FEATURES} if fitted else None
            )
            for row in subset:
                known, transformed = contexts[asset, row["clock_ms"], horizon]
                target = f"target_{horizon}"
                row["spreads"] = {
                    factor: diagnostic_spread(
                        known,
                        transformed[factor] if transformed is not None else None,
                        target,
                        directions[factor],
                    )
                    for factor in FACTORS
                }
                (
                    row["prediction_rank_ic"],
                    row["prediction_squared_error"],
                    row["zero_baseline_squared_error"],
                    row["prediction_members"],
                ) = None, None, None, 0
                if mean_beta and transformed is not None:
                    predictions = transformed.to_numpy() @ np.asarray(
                        [mean_beta[f] for f in FEATURES]
                    )
                    valid = np.isfinite(known[target]).to_numpy()
                    if valid.sum() >= MIN_MEMBERS:
                        y = known.loc[valid, target].to_numpy()
                        y = y - y.mean()
                        row["prediction_rank_ic"] = correlation(predictions[valid], y, rank=True)
                        row["prediction_squared_error"] = float(
                            np.sum((y - predictions[valid]) ** 2)
                        )
                        row["zero_baseline_squared_error"] = float(np.sum(y**2))
                        row["prediction_members"] = int(valid.sum())
                    row["prediction_spread"] = diagnostic_spread(known, predictions, target, 1)
            phases = {}
            for phase, bounds in protocol["phases"].items():
                part = [r for r in subset if r["phase"] == phase]
                days = list(
                    pd.date_range(
                        bounds["start"], bounds["end"], inclusive="left", freq="D"
                    ).strftime("%Y-%m-%d")
                )
                factors = {}
                for factor in FACTORS:
                    stats = {
                        metric: daily_stat(
                            part,
                            lambda r, f=factor, m=metric: r["factor_statistics"].get(f, {}).get(m),
                            days,
                        )
                        for metric in ("pearson_ic", "rank_ic", "coefficient")
                    }
                    stats["multivariate_coefficient"] = daily_stat(
                        part,
                        lambda r, f=factor: r["multivariate"].get(f) if r["multivariate"] else None,
                        days,
                    )
                    stats["diagnostic_spread"] = {
                        metric: daily_stat(
                            part, lambda r, f=factor, m=metric: r["spreads"][f][m], days
                        )
                        for metric in ("gross", "base_cost_sensitivity", "stress_cost_sensitivity")
                    }
                    stats["training_direction"] = directions[factor]
                    factors[factor] = stats
                    tests.append(
                        (asset, horizon, phase, factor, stats["rank_ic"]["p_asymptotic_two_sided"])
                    )
                valid = [r for r in part if r["prediction_squared_error"] is not None]
                baseline = sum(r["zero_baseline_squared_error"] for r in valid)
                phases[phase] = {
                    "scheduled_cross_sections": len(part),
                    "multivariate_successes": sum(r["multivariate"] is not None for r in part),
                    "multivariate_failures": dict(
                        Counter(
                            r["multivariate_failure"] for r in part if r["multivariate_failure"]
                        )
                    ),
                    "factors": factors,
                    "frozen_training_model": {
                        "rank_ic": daily_stat(part, lambda r: r["prediction_rank_ic"], days),
                        "excess_return_r_squared_vs_zero": 1
                        - sum(r["prediction_squared_error"] for r in valid) / baseline
                        if baseline > 0
                        else None,
                        "prediction_members": sum(r["prediction_members"] for r in valid),
                    },
                }
            groups[f"{asset}:{horizon}m"] = {
                "training_mean_coefficients": mean_beta,
                "training_multivariate_fits": len(fitted),
                "phases": phases,
            }
    adjusted = bh_adjust([value[-1] for value in tests])
    for (asset, horizon, phase, factor, _), q in zip(tests, adjusted, strict=True):
        groups[f"{asset}:{horizon}m"]["phases"][phase]["factors"][factor]["rank_ic"][
            "bh_q_all_declared_tests"
        ] = q
    return {
        "groups": groups,
        "hourly_records": records,
        "declared_rank_ic_tests": len(tests),
        "valid_p_value_tests": sum(v[-1] is not None for v in tests),
        "factor_signs_refitted_after_training": False,
        "economic_account_evaluated": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
