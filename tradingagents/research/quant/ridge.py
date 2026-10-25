"""成熟标签上的固定岭回归路径；截距不惩罚，内层只按过去日历选参数。"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
PENALTIES = (1e-6, 1e-4, 1e-2, 1.0, 100.0)


def _integer(value, name):
    """毫秒和数量拒绝布尔、小数及超出有符号整数范围的值。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name}必须为整数")
    if not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{name}超出非负毫秒范围")
    return int(value)


def _columns(frame, features, common_features, target):
    """模型列和共同成员列分别冻结，标签及身份列不可混入特征。"""
    features = tuple(features)
    common = tuple(common_features) if common_features is not None else features
    if (
        not features
        or not common
        or len(set(features)) != len(features)
        or len(set(common)) != len(common)
        or not set(features) <= set(common)
        or any(
            not isinstance(f, str)
            or f in {target, "symbol", "asset_class", "clock_ms"}
            or f.startswith(("target_", "label_", "phase_"))
            for f in common
        )
        or frame.columns.has_duplicates
        or not {"clock_ms", target, *common} <= set(frame.columns)
    ):
        raise ValueError("模型／共同人口列不合法或缺少时钟及标签")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in frame.clock_ms):
        raise ValueError("样本时钟必须为整数毫秒")
    if (frame.clock_ms < 0).any() or (
        "symbol" in frame and frame.duplicated(["symbol", "clock_ms"]).any()
    ):
        raise ValueError("时钟为负或合约时钟重复")
    return features, common


def _penalties(values):
    """固定网格只允许不同的有限正惩罚，不以零惩罚冒充岭回归。"""
    values = tuple(values)
    if not values or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise ValueError("惩罚网格必须为有限正数")
    result = tuple(float(v) for v in values)
    if len(set(result)) != len(result):
        raise ValueError("惩罚网格重复")
    return result


def fit_path(
    frame,
    *,
    features,
    target,
    refit_ms,
    training_start_ms,
    label_delay_ms,
    penalties=PENALTIES,
    common_features=None,
    minimum_rows=20,
):
    """最小化平均平方误差加λ平方范数；一次共同设计复用全部固定惩罚。"""
    features, common = _columns(frame, features, common_features, target)
    refit = _integer(refit_ms, "重训时钟")
    start = _integer(training_start_ms, "训练起点")
    delay = _integer(label_delay_ms, "标签延迟")
    minimum_rows = _integer(minimum_rows, "最少训练行")
    if start >= refit or minimum_rows < 2:
        raise ValueError("训练起点须早于重训且至少两行")
    penalties = _penalties(penalties)
    # 相减而非加标签延迟，防止大整数时钟相加溢出。
    prior = frame[(frame.clock_ms >= start) & (frame.clock_ms < refit - delay)]
    known = prior[np.isfinite(prior[list(common)].to_numpy(dtype=float)).all(axis=1)]
    valid = np.isfinite(known[target].to_numpy(dtype=float))
    fit = known.loc[valid]
    record = {
        "schema": "ta-causal-ridge-path.v1",
        "features": list(features),
        "common_features": list(common),
        "target": target,
        "refit_ms": refit,
        "training_start_ms": start,
        "label_delay_ms": delay,
        "objective": "mean_squared_error_plus_lambda_sum_beta_squared",
        "intercept_penalized": False,
        "standard_deviation_ddof": 0,
        "common_feature_rows": len(known),
        "known_target_rows": len(fit),
        "maximum_scaler_label_end_ms": int(known.clock_ms.max()) + delay if len(known) else None,
        "maximum_fitted_label_end_ms": int(fit.clock_ms.max()) + delay if len(fit) else None,
        "means": {},
        "scales": {},
        "constant_features": [],
        "fits": [],
    }
    status, matrix, response = "READY", None, None
    if not len(known) or len(fit) < minimum_rows:
        status = "INSUFFICIENT_TRAINING_ROWS"
    else:
        raw = known[list(features)].to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            constant = np.ptp(raw, axis=0) == 0
            means = raw.mean(axis=0)
            means[constant] = raw[0, constant]
            scales = raw.std(axis=0, ddof=0)
            scales[constant] = 1.0
            matrix = (raw[valid] - means) / scales
            matrix[:, constant] = 0.0
        if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            status = "NONFINITE_TRAINING_SCALER"
        elif not np.isfinite(matrix).all():
            status = "NONFINITE_STANDARDIZED_TRAINING"
        else:
            record.update(
                means=dict(zip(features, map(float, means), strict=True)),
                scales=dict(zip(features, map(float, scales), strict=True)),
                constant_features=[f for f, c in zip(features, constant, strict=True) if c],
            )
            response = fit[target].to_numpy(dtype=float)
            # 缺标签行参与特征标准化；拟合行再中心化，使截距始终不被惩罚。
            x_mean, y_mean = matrix.mean(axis=0), float(response.mean())
            centered, centered_y = matrix - x_mean, response - y_mean
            with np.errstate(over="ignore", invalid="ignore"):
                gram = centered.T @ centered / len(fit)
                rhs = centered.T @ centered_y / len(fit)
            if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
                status = "NONFINITE_NORMAL_EQUATION"
    for penalty in penalties:
        item = {"penalty": penalty, "status": status, "intercept": None, "coefficients": None}
        if status == "READY":
            try:
                beta = np.linalg.solve(gram + penalty * np.eye(len(features)), rhs)
                beta[constant] = 0.0
                intercept = float(y_mean - x_mean @ beta)
                if not np.isfinite(beta).all() or not math.isfinite(intercept):
                    item["status"] = "NONFINITE_COEFFICIENT"
                else:
                    item.update(
                        intercept=intercept,
                        coefficients=dict(zip(features, map(float, beta), strict=True)),
                    )
            except np.linalg.LinAlgError:
                item["status"] = "LINEAR_ALGEBRA_FAILURE"
        record["fits"].append(item)
    return record


def predict(frame, record, penalty):
    """完整共同特征人口先决定可预测成员；不读取标签或再拟合标准化。"""
    features, common = tuple(record["features"]), tuple(record["common_features"])
    if (
        frame.index.has_duplicates
        or frame.columns.has_duplicates
        or not set(common) <= set(frame.columns)
        or not set(features) <= set(common)
        or len(set(common)) != len(common)
        or len(set(features)) != len(features)
    ):
        raise ValueError("预测列或行身份与冻结设计不符")
    selected = [item for item in record["fits"] if item["penalty"] == penalty]
    if len(selected) != 1:
        raise ValueError("惩罚未精确对应一个冻结模型")
    item = selected[0]
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if item["status"] != "READY":
        return result
    if (
        set(item["coefficients"]) != set(features)
        or set(record["means"]) != set(features)
        or set(record["scales"]) != set(features)
        or not set(record["constant_features"]) <= set(features)
        or not math.isfinite(item["intercept"])
        or not np.isfinite(list(item["coefficients"].values())).all()
        or not np.isfinite(list(record["means"].values())).all()
        or not np.isfinite(list(record["scales"].values())).all()
        or any(v <= 0 for v in record["scales"].values())
        or any(item["coefficients"][f] != 0 for f in record["constant_features"])
    ):
        raise ValueError("冻结系数或标准化记录不合法")
    known = np.isfinite(frame[list(common)].to_numpy(dtype=float)).all(axis=1)
    active = [f for f in features if f not in record["constant_features"]]
    values = np.full(int(known.sum()), item["intercept"], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if active:
            matrix = frame.loc[known, active].to_numpy(dtype=float)
            matrix = (matrix - [record["means"][f] for f in active]) / [
                record["scales"][f] for f in active
            ]
            values += matrix @ np.asarray([item["coefficients"][f] for f in active])
    result.loc[known] = np.where(np.isfinite(values), values, np.nan)
    return result


def select_penalty(
    frame,
    *,
    features,
    target,
    training_start_ms,
    outer_refit_ms,
    label_delay_ms,
    folds,
    penalties=PENALTIES,
    common_features=None,
    minimum_rows=20,
):
    """显式完整UTC日内层折，等权日MSE最小；同分取更大λ，外层标签不参与。"""
    outer = _integer(outer_refit_ms, "外层重训时钟")
    train_start = _integer(training_start_ms, "训练起点")
    delay = _integer(label_delay_ms, "标签延迟")
    features, common_features = _columns(frame, features, common_features, target)
    penalties = _penalties(penalties)
    folds = tuple(tuple(_integer(v, "内层折时钟") for v in fold) for fold in folds)
    if not folds or any(len(fold) != 2 for fold in folds):
        raise ValueError("内层折须明确起止日历")
    previous_end = train_start
    for start, end in folds:
        if (
            not train_start < start < end <= outer
            or start < previous_end
            or start % DAY_MS
            or end % DAY_MS
        ):
            raise ValueError("内层折须为不重叠UTC整日，且在外层训练截止之前")
        previous_end = end
    records, scores = [], {penalty: [] for penalty in penalties}
    for start, end in folds:
        path = fit_path(
            frame,
            features=features,
            target=target,
            refit_ms=start,
            training_start_ms=train_start,
            label_delay_ms=delay,
            penalties=penalties,
            common_features=common_features,
            minimum_rows=minimum_rows,
        )
        validation = frame[(frame.clock_ms >= start) & (frame.clock_ms < end - delay)]
        days = list(range(start // DAY_MS, end // DAY_MS))
        members = np.isfinite(validation[list(common_features)].to_numpy(dtype=float)).all(axis=1)
        by_penalty = []
        for penalty in penalties:
            values = predict(validation, path, penalty)
            unknown_predictions = int((members & ~np.isfinite(values)).sum())
            known = np.isfinite(values) & np.isfinite(validation[target])
            errors = (values[known] - validation.loc[known, target]) ** 2
            means = errors.groupby(validation.loc[known, "clock_ms"] // DAY_MS).mean()
            daily = [float(means.get(day, np.nan)) for day in days]
            ready = not unknown_predictions and all(math.isfinite(value) for value in daily)
            status = (
                "INCOMPLETE_INNER_PREDICTIONS" if unknown_predictions else "INCOMPLETE_INNER_DAYS"
            )
            by_penalty.append(
                {
                    "penalty": penalty,
                    "status": "READY" if ready else status,
                    "known_feature_rows": int(members.sum()),
                    "unknown_prediction_rows": unknown_predictions,
                    "daily_mse": [v if math.isfinite(v) else None for v in daily],
                }
            )
            scores[penalty].extend(daily if ready else [float("nan")] * len(days))
        records.append(
            {"start_ms": start, "end_ms": end, "training_path": path, "validation": by_penalty}
        )
    grid = [
        {
            "penalty": p,
            "mean_daily_mse": math.fsum(x / len(v) for x in v)
            if v and np.isfinite(v).all()
            else None,
        }
        for p, v in scores.items()
    ]
    ready = [row for row in grid if row["mean_daily_mse"] is not None]
    choice = min(ready, key=lambda row: (row["mean_daily_mse"], -row["penalty"])) if ready else None
    selected = choice["penalty"] if choice else None
    final = (
        fit_path(
            frame,
            features=features,
            target=target,
            refit_ms=outer,
            training_start_ms=train_start,
            label_delay_ms=delay,
            penalties=(selected,),
            common_features=common_features,
            minimum_rows=minimum_rows,
        )
        if choice
        else None
    )
    status = "NO_COMPLETE_INNER_VALIDATION"
    if final:
        status = "READY" if final["fits"][0]["status"] == "READY" else "FINAL_FIT_NOT_READY"
    return {
        "schema": "ta-causal-ridge-inner-selection.v1",
        "status": status,
        "selected_penalty": selected,
        "grid": grid,
        "folds": records,
        "final_path": final,
    }
