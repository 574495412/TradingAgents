"""在已使用历史上执行固定扩展训练，不将滚动检验称为新增独立样本外。"""

import hashlib
import json
from collections import Counter
from numbers import Integral

import numpy as np

from .factors.funding import NAMES
from .funding_study import MODELS
from .linear_family import fit_section
from .regression import FEATURES
from .regression_run import MINUTE, PHASES, ms

REFITS = tuple(ms(f"2026-10-{day:02}T00:00:00Z") for day in (6, 8, 10, 12))
END_MS = ms("2026-10-14T00:00:00Z")
TRAINING_START_MS = ms(PHASES["training"]["start"])
HORIZONS = (20, 60, 240)


def _specification(samples, protocol):
    """核对原模型族与原阶段，拒绝用重新命名的阶段掩盖跨界标签。"""
    features = FEATURES + NAMES
    models = {name: tuple(columns) for name, columns in protocol["models"].items()}
    if (
        tuple(protocol["features"]) != features
        or models != MODELS
        or tuple(protocol["horizons_minutes"]) != HORIZONS
        or protocol["phases"] != PHASES
    ):
        raise ValueError("滚动训练必须沿用原特征、五模型、期限及完整窗口阶段")
    required = {
        "symbol",
        "asset_class",
        "clock_ms",
        *features,
        *(f"{prefix}_{h}" for h in HORIZONS for prefix in ("target", "phase")),
    }
    if not required <= set(samples.columns):
        raise ValueError("滚动训练面板缺少原身份、特征或标签列")
    if samples.empty or samples[["symbol", "asset_class", "clock_ms"]].isna().any().any():
        raise ValueError("滚动训练面板或样本身份不能未知")
    if samples.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("滚动训练面板存在重复合约时钟")
    if any(
        isinstance(clock, (bool, np.bool_)) or not isinstance(clock, Integral)
        for clock in samples.clock_ms
    ):
        raise ValueError("滚动训练时钟必须为整数毫秒")
    if samples.groupby("symbol").asset_class.nunique().gt(1).any():
        raise ValueError("同一合约不能跨资产类别")
    return features, models


def _hourly_record(frame, asset, clock, horizon, features, models):
    """只拟合最后重训前已成熟的标签，原阶段缺口始终保留。"""
    label_end = clock + (horizon + 1) * MINUTE
    expected = next(
        (
            name
            for name, bounds in PHASES.items()
            if ms(bounds["start"]) <= clock and label_end < ms(bounds["end"])
        ),
        "excluded_boundary",
    )
    phases = frame[f"phase_{horizon}"]
    if phases.nunique(dropna=False) != 1 or phases.iloc[0] != expected:
        raise ValueError("原阶段标签与完整预测窗口的固定日历不同")
    target = f"target_{horizon}"
    if expected == "excluded_boundary" and np.isfinite(frame[target]).any():
        raise ValueError("原阶段边界的未知标签不能恢复或补零")
    known = frame[np.isfinite(frame[list(features)]).all(axis=1)]
    record = {
        "asset_class": asset,
        "clock_ms": clock,
        "horizon_minutes": horizon,
        "label_end_ms": label_end,
        "source_phase": expected,
        "scheduled_members": len(frame),
        "known_feature_members": len(known),
        "known_target_members": int(np.isfinite(known[target]).sum()),
        "fits": {},
        "failures": {},
    }
    if expected == "excluded_boundary":
        record["failures"] = dict.fromkeys(models, "ORIGINAL_PHASE_BOUNDARY")
    elif label_end >= REFITS[-1]:
        record["failures"] = dict.fromkeys(models, "LABEL_NOT_MATURE_AT_ANY_REFIT")
    else:
        try:
            _, _, fits, failures = fit_section(frame, target, features, models)
        except np.linalg.LinAlgError:
            fits, failures = {}, dict.fromkeys(models, "LINEAR_ALGEBRA_FAILURE")
        for name, coefficients in list(fits.items()):
            if not np.isfinite(list(coefficients.values())).all():
                fits.pop(name)
                failures[name] = "NONFINITE_COEFFICIENT"
        record["fits"], record["failures"] = fits, failures
    return record


def _snapshot(records, refit, models):
    """五模型共享同一组成熟且全部可拟合的时刻，不回退旧系数或零系数。"""
    prior = [row for row in records if row["clock_ms"] < refit]
    mature = [row for row in prior if row["label_end_ms"] < refit]
    common = [row for row in mature if len(row["fits"]) == len(models)]
    clocks = [row["clock_ms"] for row in common]
    coefficients = {
        name: {
            feature: float(np.mean([row["fits"][name][feature] for row in common]))
            for feature in columns
        }
        if common
        else None
        for name, columns in models.items()
    }
    status = "READY" if common else "NO_COMMON_FIT_HOURS"
    if common and any(not np.isfinite(list(beta.values())).all() for beta in coefficients.values()):
        coefficients = dict.fromkeys(models)
        status = "NONFINITE_AGGREGATE_COEFFICIENT"
    return {
        "status": status,
        "model_status": dict.fromkeys(models, status),
        "frozen_training_coefficients": coefficients,
        "training_common_fit_hours": len(common),
        "common_training_clocks_ms": clocks,
        "common_training_clocks_sha256": hashlib.sha256(
            json.dumps(clocks, separators=(",", ":")).encode()
        ).hexdigest(),
        "maximum_training_label_end_ms": max((row["label_end_ms"] for row in common), default=None),
        "diagnostics": {
            "observed_prior_hours": len(prior),
            "mature_label_clock_hours": len(mature),
            "purged_unmatured_hours": len(prior) - len(mature),
            "original_boundary_hours": sum(
                row["source_phase"] == "excluded_boundary" for row in prior
            ),
            "all_target_unknown_hours": sum(row["known_target_members"] == 0 for row in mature),
            "excluded_from_common_hours": len(mature) - len(common),
            "partial_model_fit_hours": sum(0 < len(row["fits"]) < len(models) for row in mature),
            "individually_fitted_hours": {
                name: sum(name in row["fits"] for row in mature) for name in models
            },
            "failure_counts_by_model": {
                name: dict(
                    Counter(row["failures"][name] for row in mature if name in row["failures"])
                )
                for name in models
            },
        },
    }


def fit_expanding(samples, protocol):
    """返回固定四次重训的系数与诊断；不写文件、不选模型、不执行账户。

    每个资产及期限沿用原九特征共同人口和逐小时截面估计，再对五模型
    共同可拟合时刻的斜率取等权平均。原标签只在退出严格早于重训时成熟。
    """
    features, models = _specification(samples, protocol)
    assets = sorted(samples.asset_class.unique())
    historical = samples[(samples.clock_ms >= TRAINING_START_MS) & (samples.clock_ms < REFITS[-1])]
    records = []
    for (asset, clock), frame in historical.groupby(["asset_class", "clock_ms"], sort=True):
        for horizon in HORIZONS:
            records.append(_hourly_record(frame, asset, int(clock), horizon, features, models))
    refits = {}
    for refit, end in zip(REFITS, (*REFITS[1:], END_MS), strict=True):
        groups = {}
        for asset in assets:
            for horizon in HORIZONS:
                subset = [
                    row
                    for row in records
                    if row["asset_class"] == asset and row["horizon_minutes"] == horizon
                ]
                groups[f"{asset}:{horizon}m"] = _snapshot(subset, refit, models)
        refits[str(refit)] = {"refit_ms": refit, "end_ms": end, "groups": groups}
    return {
        "schema": "ta-historical-expanding-training.v1",
        "training_start_ms": TRAINING_START_MS,
        "start_ms": REFITS[0],
        "end_ms": END_MS,
        "refit_times_ms": list(REFITS),
        "features": list(features),
        "models": {name: list(columns) for name, columns in models.items()},
        "horizons_minutes": list(HORIZONS),
        "refits": refits,
        "hourly_diagnostics": [
            {
                **{key: value for key, value in row.items() if key != "fits"},
                "fitted_models": list(row["fits"]),
            }
            for row in records
        ],
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
    }


def coefficients_at(result, asset_class, horizon_minutes, model, clock_ms):
    """供特征白名单预测路径查询当块系数，无共同拟合时显式返回未知。"""
    if (
        isinstance(clock_ms, (bool, np.bool_))
        or not isinstance(clock_ms, Integral)
        or not REFITS[0] <= clock_ms < END_MS
    ):
        raise ValueError("决策时钟必须位于固定历史滚动检验区间")
    refit = next(value for value in reversed(REFITS) if value <= clock_ms)
    try:
        group = result["refits"][str(refit)]["groups"][f"{asset_class}:{horizon_minutes}m"]
        coefficients = group["frozen_training_coefficients"][model]
    except KeyError as error:
        raise ValueError("资产、期限或模型不属于已声明滚动训练结果") from error
    return dict(coefficients) if coefficients is not None else None
