"""将原生图选中的结算退出假说固定为可复算回归，不产生订单。"""

import argparse
import importlib.metadata
import math
import shutil
import sys
import time
from collections import defaultdict
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read, save, sha, verify_receipts
from .backtesting.search import funding_source_path
from .factors.funding import NAMES as FUNDING_NAMES, calculate as funding_factors
from .factors.settlement import H01_NAME, calculate as settlement_factor
from .funding_study import regular_events
from .regression import FEATURES as LEGACY_FEATURES, bh_adjust, correlation, daily_stat
from .regression_run import MINUTE, PHASES, features_from_minutes, ms, validated_rows

HYPOTHESIS_ID = "phase3_settlement_unwind_01"
FEATURES = LEGACY_FEATURES + FUNDING_NAMES + (H01_NAME,)
MODELS = {
    "legacy": LEGACY_FEATURES,
    "expanded": LEGACY_FEATURES + FUNDING_NAMES,
    "add_h01": LEGACY_FEATURES + (H01_NAME,),
    "expanded_h01": FEATURES,
}
CONTRASTS = {
    "add_h01_vs_legacy": ("legacy", "add_h01"),
    "expanded_h01_vs_expanded": ("expanded", "expanded_h01"),
}
REFITS = tuple(ms(f"2026-10-{day:02}T00:00:00Z") for day in (6, 8, 10, 12))
END_MS = ms(PHASES["retrospective_validation"]["end"])
TRAINING_START_MS = ms(PHASES["training"]["start"])
LABEL_DELAY_MS = 21 * MINUTE


def phase_at(clock):
    """标签结束严格早于阶段终点，恰好到边界仍不进入该阶段。"""
    return next(
        (
            name
            for name, bounds in PHASES.items()
            if ms(bounds["start"]) <= clock and clock + LABEL_DELAY_MS < ms(bounds["end"])
        ),
        "excluded_boundary",
    )


def price_features(minutes):
    """从完整分钟网格生成旧七项特征与精确的二十分钟收盘标签。"""
    result = features_from_minutes(minutes)[list(LEGACY_FEATURES)].copy()
    result["target_20"] = minutes["close"].shift(-21) / minutes["open"].shift(-2) - 1
    # 价格路径缺口单列保存，账户须独立拒绝无法完整盯市的持仓路径。
    observed = minutes[["open", "close"]].notna().all(axis=1)
    result["holding_path_observed"] = observed.rolling(20, min_periods=20).sum().shift(-21) == 20
    return result


def fit_record(frame, refit_ms, *, features=FEATURES, models=MODELS):
    """成熟共同特征先定标准化人口，再过滤标签；零H01不删行。"""
    if isinstance(refit_ms, (bool, np.bool_)) or not isinstance(refit_ms, Integral):
        raise ValueError("重训时钟必须为整数毫秒")
    required = {"clock_ms", "target_20", *features}
    if not required <= set(frame.columns):
        raise ValueError("训练缺少声明特征、时钟或标签")
    prior = frame[
        (frame.clock_ms >= TRAINING_START_MS) & (frame.clock_ms + LABEL_DELAY_MS < refit_ms)
    ]
    known = prior[np.isfinite(prior[list(features)]).all(axis=1)]
    means = known[list(features)].mean()
    scales = known[list(features)].std(ddof=0)
    ready = np.isfinite(known.target_20)
    fit = known.loc[ready]
    coefficients, statuses, directions = {}, {}, {}
    for model, columns in models.items():
        beta, status = None, "READY"
        if len(known) == 0 or len(fit) < max(20, len(columns) + 1):
            status = "INSUFFICIENT_TRAINING_ROWS"
        elif any(not math.isfinite(scales[f]) or scales[f] <= 0 for f in columns):
            status = "ZERO_OR_UNKNOWN_TRAINING_VARIANCE"
        else:
            x = ((fit[list(columns)] - means[list(columns)]) / scales[list(columns)]).to_numpy()
            matrix = np.column_stack([np.ones(len(x)), x])
            try:
                if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
                    status = "RANK_DEFICIENT_FIXED_DESIGN"
                else:
                    values = np.linalg.lstsq(matrix, fit.target_20.to_numpy(), rcond=None)[0]
                    if not np.isfinite(values).all():
                        status = "NONFINITE_COEFFICIENT"
                    else:
                        beta = dict(zip(("intercept", *columns), map(float, values), strict=True))
            except np.linalg.LinAlgError:
                status = "LINEAR_ALGEBRA_FAILURE"
        coefficients[model], statuses[model] = beta, status
        directions[model] = beta[H01_NAME] > 0 if beta is not None and H01_NAME in beta else None
    return {
        "refit_ms": int(refit_ms),
        "means": {f: float(means[f]) if math.isfinite(means[f]) else None for f in features},
        "scales": {f: float(scales[f]) if math.isfinite(scales[f]) else None for f in features},
        "coefficients": coefficients,
        "status_by_model": statuses,
        "h01_positive_direction_by_model": directions,
        "common_feature_rows": len(known),
        "known_target_rows": len(fit),
        "h01_known_zero_rows": int((known[H01_NAME] == 0).sum()) if H01_NAME in known else 0,
        "maximum_standardization_label_end_ms": int(known.clock_ms.max() + LABEL_DELAY_MS)
        if len(known)
        else None,
        "maximum_fitted_label_end_ms": int(fit.clock_ms.max() + LABEL_DELAY_MS)
        if len(fit)
        else None,
        "standard_deviation_ddof": 0,
        "estimator": "pooled_ols_with_intercept",
    }


def fit_coefficients(samples):
    """四个固定重训时刻保存全部资产、四模型与失败状态。"""
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, Integral)
        or v % (60 * MINUTE) != 5 * MINUTE
        for v in samples.clock_ms
    ):
        raise ValueError("研究样本只允许UTC每小时第五分钟的整数时钟")
    if samples.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("研究样本存在重复合约时钟")
    if samples.groupby("symbol").asset_class.nunique().gt(1).any():
        raise ValueError("同一合约不能跨资产类别")
    refits = {}
    for refit, end in zip(REFITS, (*REFITS[1:], END_MS), strict=True):
        groups = {
            f"{asset}:20m": fit_record(frame, refit)
            for asset, frame in samples.groupby("asset_class", sort=True)
        }
        refits[str(refit)] = {"refit_ms": refit, "end_ms": end, "groups": groups}
    return {
        "schema": "ta-settlement-pooled-coefficients.v1",
        "hypothesis_id": HYPOTHESIS_ID,
        "features": list(FEATURES),
        "models": MODELS,
        "refits": refits,
    }


def record_at(coefficients, asset_class, clock_ms, mode="fixed"):
    """固定方式始终首轮；滚动方式只查当前已到达块，训练段仅作回顾诊断。"""
    if mode not in {"fixed", "rolling"}:
        raise ValueError("系数方式只能为fixed或rolling")
    refit = (
        REFITS[0]
        if mode == "fixed" or clock_ms < REFITS[0]
        else next(value for value in reversed(REFITS) if value <= clock_ms)
    )
    return coefficients["refits"][str(refit)]["groups"][f"{asset_class}:20m"]


def predict(frame, record, model, features=FEATURES):
    """只读取特征白名单与冻结记录；未来标签不能改变成员或分数。"""
    if (
        tuple(features) != FEATURES
        or model not in MODELS
        or not set(features) <= set(frame.columns)
    ):
        raise ValueError("预测模型或特征不属于固定设计")
    if frame.index.has_duplicates or ("symbol" in frame and frame.symbol.duplicated().any()):
        raise ValueError("预测截面身份重复")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    known = frame[np.isfinite(frame[list(features)]).all(axis=1)]
    beta = record["coefficients"][model]
    if len(known) < 20 or beta is None:
        return result
    if set(beta) != {"intercept", *MODELS[model]} or not np.isfinite(list(beta.values())).all():
        raise ValueError("冻结系数没有精确覆盖固定模型白名单")
    values = pd.Series(float(beta["intercept"]), index=known.index)
    for name, slope in beta.items():
        if name == "intercept":
            continue
        mean, scale = record["means"][name], record["scales"][name]
        if mean is None or scale is None or scale <= 0 or not math.isfinite(scale):
            raise ValueError("冻结标准化状态不能产生有效预测")
        values += (known[name] - mean) / scale * slope
    result.loc[known.index] = values.where(np.isfinite(values))
    return result


def _summarize(rows, days):
    """收益原标签相对零预测的R²，不在验证时去均值改变预测目标。"""
    models = {}
    for name in MODELS:
        values = [r["models"][name] for r in rows if r["models"][name] is not None]
        denominator = sum(v["null_sse"] for v in values)
        models[name] = {
            "cross_sections": len(values),
            "r_squared_vs_zero": 1 - sum(v["sse"] for v in values) / denominator
            if denominator > 0
            else None,
            "rank_ic": daily_stat(
                rows, lambda r, n=name: r["models"][n]["rank_ic"] if r["models"][n] else None, days
            ),
        }
    return {
        "scheduled_cross_sections": len(rows),
        "models": models,
        "comparisons": {
            name: daily_stat(rows, lambda r, n=name: r["contrasts"][n], days) for name in CONTRASTS
        },
        "h01_rank_ic": daily_stat(rows, lambda r: r["h01_rank_ic"], days),
    }


def evaluate(samples, coefficients):
    """返回全量小时诊断、逐行预测及固定十二项校正，绝不挑模型或日期。"""
    records, prediction_frames = [], []
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        phase = phase_at(int(clock))
        if not frame.phase_20.eq(phase).all():
            raise ValueError("小时标签阶段与固定边界不符")
        known = np.isfinite(frame[list(FEATURES)]).all(axis=1)
        valid = known & np.isfinite(frame.target_20)
        for mode in ("fixed", "rolling"):
            record = record_at(coefficients, asset, int(clock), mode)
            outputs = frame[
                ["symbol", "asset_class", "clock_ms", "phase_20", "target_20", H01_NAME]
                + [
                    name
                    for name in ("settlement_latest_event_ms", "settlement_age_minutes")
                    if name in frame
                ]
            ].copy()
            row = {
                "asset_class": asset,
                "clock_ms": int(clock),
                "mode": mode,
                "phase": phase,
                "day": pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                "scheduled_members": len(frame),
                "common_feature_members": int(known.sum()),
                "known_target_members": int(valid.sum()),
                "models": {},
                "contrasts": {},
                "h01_rank_ic": correlation(
                    frame.loc[valid, H01_NAME], frame.loc[valid, "target_20"], rank=True
                )
                if valid.sum() >= 20
                else None,
            }
            for model in MODELS:
                scores = predict(frame[list(FEATURES)], record, model)
                outputs[f"prediction_{model}"] = scores
                use = valid & np.isfinite(scores)
                metric = None
                if use.sum() >= 20:
                    y, prediction = (
                        frame.loc[use, "target_20"].to_numpy(),
                        scores.loc[use].to_numpy(),
                    )
                    errors = (y - prediction) ** 2
                    metric = {
                        "members": int(use.sum()),
                        "sse": float(errors.sum()),
                        "mse": float(errors.mean()),
                        "null_sse": float(y @ y),
                        "rank_ic": correlation(prediction, y, rank=True),
                    }
                row["models"][model] = metric
            for name, (base, full) in CONTRASTS.items():
                a, b = (row["models"][m] for m in (base, full))
                row["contrasts"][name] = a["mse"] - b["mse"] if a and b else None
            outputs["coefficient_mode"] = mode
            prediction_frames.append(outputs)
            records.append(row)
    groups, tests = {}, []
    for asset in sorted(samples.asset_class.unique()):
        modes = {}
        for mode in ("fixed", "rolling"):
            part = [r for r in records if r["asset_class"] == asset and r["mode"] == mode]
            phases = {}
            for phase, bounds in PHASES.items():
                days = list(
                    pd.date_range(
                        bounds["start"], bounds["end"], inclusive="left", freq="D"
                    ).strftime("%Y-%m-%d")
                )
                phases[phase] = _summarize([r for r in part if r["phase"] == phase], days)
            validation = phases["retrospective_validation"]
            for value in validation["comparisons"].values():
                value["test_p"] = value["p_asymptotic_two_sided"]
                value["test_tail"] = "two_sided_and_positive_mean_required"
                tests.append(value)
            value = validation["h01_rank_ic"]
            value["test_p"] = (
                math.erfc(value["t"] / math.sqrt(2)) / 2 if value["t"] is not None else None
            )
            value["test_tail"] = "one_sided_positive"
            tests.append(value)
            blocks = []
            for start, end in zip(REFITS, (*REFITS[1:], END_MS), strict=True):
                days = list(
                    pd.date_range(
                        pd.Timestamp(start, unit="ms", tz="UTC"),
                        pd.Timestamp(end, unit="ms", tz="UTC"),
                        inclusive="left",
                        freq="D",
                    ).strftime("%Y-%m-%d")
                )
                blocks.append(
                    {
                        "start_ms": start,
                        "end_ms": end,
                        **_summarize([r for r in part if start <= r["clock_ms"] < end], days),
                    }
                )
            modes[mode] = {"phases": phases, "two_day_blocks": blocks}
        groups[f"{asset}:20m"] = modes
    if len(tests) != 12:
        raise ValueError("声明的两资产十二项检验不能增减")
    q_values = bh_adjust([v["test_p"] if v["test_p"] is not None else 1.0 for v in tests])
    for value, q in zip(tests, q_values, strict=True):
        value["bh_q_12"] = q if value["test_p"] is not None else None
        value["positive_and_q_at_most_005"] = bool(
            value["mean"] is not None
            and value["mean"] > 0
            and value["bh_q_12"] is not None
            and value["bh_q_12"] <= 0.05
        )
    result = {
        "schema": "ta-settlement-unwind-statistics.v1",
        "hypothesis_id": HYPOTHESIS_ID,
        "groups": groups,
        "declared_tests": 12,
        "bh_adjustment_family_size": 12,
        "valid_p_value_tests": sum(v["test_p"] is not None for v in tests),
        "previous_declared_funding_tests": 216,
        "independent_oos": False,
        "duplicated_raw_h01_ic_across_modes_is_not_independent_replication": True,
        "economic_account_evaluated": False,
        "orders_enabled": False,
        "stable_profitability_proven": False,
        "production_eligible": False,
    }
    return result, records, pd.concat(prediction_frames, ignore_index=True)


def attribution(samples, coefficients, predictions):
    """全量分组误差和线性预测项，既不删失败行也不解释为现金收益。"""
    identity = ["symbol", "asset_class", "clock_ms"]
    diagnostics = [H01_NAME, "settlement_latest_event_ms", "settlement_age_minutes"]
    if not set(identity + diagnostics) <= set(samples.columns):
        raise ValueError("归因样本缺少事件身份或年龄诊断")
    if (
        samples.duplicated(identity).any()
        or predictions.duplicated(identity + ["coefficient_mode"]).any()
    ):
        raise ValueError("归因输入身份重复")
    data = predictions.drop(columns=diagnostics, errors="ignore").merge(
        samples[identity + diagnostics],
        on=identity,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not data["_merge"].eq("both").all() or len(data) != 2 * len(samples):
        raise ValueError("归因预测必须完整覆盖两种方式的全部样本")
    for mode in ("fixed", "rolling"):
        part = data[data.coefficient_mode == mode]
        if len(part) != len(samples):
            raise ValueError("归因不能删除某种方式的样本")
    data["date"] = pd.to_datetime(data.clock_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    data["utc_hour"] = pd.to_datetime(data.clock_ms, unit="ms", utc=True).dt.hour
    known = np.isfinite(data[H01_NAME])
    active = (
        known
        & np.isfinite(data.settlement_latest_event_ms)
        & (data.settlement_age_minutes.ge(5) & data.settlement_age_minutes.lt(65))
    )
    if (known & data[H01_NAME].ne(0) & ~active).any():
        raise ValueError("非零H01不能指向已过衰减期的事件")
    data["h01_known_zero"] = known & data[H01_NAME].eq(0)
    data["h01_unknown"] = ~known
    data["active_event"] = active
    data["expired_or_no_event_zero"] = known & ~active
    data["settlement_event"] = [
        f"{symbol}:{int(stamp)}"
        if use
        else ("UNKNOWN_H01" if missing else "INACTIVE_NO_ACTIVE_EVENT")
        for symbol, stamp, use, missing in zip(
            data.symbol, data.settlement_latest_event_ms, active, ~known, strict=True
        )
    ]
    for model in MODELS:
        data[f"h01_contribution_{model}"] = np.nan
    for (asset, clock, mode), part in data.groupby(
        ["asset_class", "clock_ms", "coefficient_mode"], sort=True
    ):
        record = record_at(coefficients, asset, int(clock), mode)
        for model in MODELS:
            ready = np.isfinite(part[f"prediction_{model}"])
            beta = record["coefficients"][model]
            if not ready.any():
                continue
            if beta is None:
                raise ValueError("未知冻结模型不能存在已知预测")
            component = 0.0
            if H01_NAME in MODELS[model]:
                scale = record["scales"][H01_NAME]
                if scale is None or scale <= 0:
                    raise ValueError("H01贡献缺少有效冻结标准差")
                component = (
                    (part.loc[ready, H01_NAME] - record["means"][H01_NAME]) / scale * beta[H01_NAME]
                )
            data.loc[part.index[ready], f"h01_contribution_{model}"] = component

    def finite_summary(values):
        values = values[np.isfinite(values)]
        return {
            "rows": len(values),
            "sum": float(values.sum()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
        }

    grouped = {}
    for axis in ("date", "utc_hour", "symbol", "settlement_event"):
        rows = []
        for (asset, mode, key), part in data.groupby(
            ["asset_class", "coefficient_mode", axis], sort=True, dropna=False
        ):
            value = {
                "asset_class": asset,
                "coefficient_mode": mode,
                axis: int(key) if axis == "utc_hour" else str(key),
                "sample_rows": len(part),
                "h01_unknown_rows": int(part.h01_unknown.sum()),
                "h01_known_zero_rows": int(part.h01_known_zero.sum()),
                "h01_nonzero_rows": int((~part.h01_unknown & ~part.h01_known_zero).sum()),
                "active_event_rows": int(part.active_event.sum()),
                "expired_or_no_event_zero_rows": int(part.expired_or_no_event_zero.sum()),
                "models": {},
                "paired_mse_improvement": {},
            }
            for model in MODELS:
                score = part[f"prediction_{model}"]
                ready = np.isfinite(score) & np.isfinite(part.target_20)
                error = (score[ready] - part.loc[ready, "target_20"]) ** 2
                value["models"][model] = {
                    "prediction": finite_summary(score),
                    "h01_contribution": finite_summary(part[f"h01_contribution_{model}"]),
                    "squared_error": finite_summary(error),
                }
            for name, (base, augmented) in CONTRASTS.items():
                a, b = (part[f"prediction_{m}"] for m in (base, augmented))
                ready = np.isfinite(a) & np.isfinite(b) & np.isfinite(part.target_20)
                y = part.loc[ready, "target_20"]
                value["paired_mse_improvement"][name] = finite_summary(
                    (a[ready] - y) ** 2 - (b[ready] - y) ** 2
                )
            rows.append(value)
        grouped[f"by_{axis}"] = rows
    return {
        "schema": "ta-settlement-full-predictive-attribution.v1",
        "hypothesis_id": HYPOTHESIS_ID,
        "sample_rows_including_both_modes": len(data),
        "h01_component_formula": "frozen_slope*(H01-frozen_training_mean)/frozen_training_std",
        "baseline_h01_component": 0,
        "known_zero_can_have_nonzero_centered_component": True,
        "inactive_old_events_are_not_active_settlements": True,
        "component_is_predictive_not_cash_or_account_pnl": True,
        "training_phase_is_in_sample_diagnostic_only": True,
        "rows_or_subsets_selected_by_outcome": False,
        **grouped,
    }


def _price_source(source, parent):
    """只沿封存的价格来源链查找原文，不扫描leader目录或生产。"""
    candidates = [
        Path(p).parent for p in parent["input_identities"] if Path(p).name == "receipts.json"
    ]
    if len(candidates) != 1:
        raise ValueError("资金费研究必须只有一条价格父来源")
    price = candidates[0].resolve()
    if sha(price / "receipts.json") != parent["input_identities"][str(price / "receipts.json")]:
        raise ValueError("价格父来源身份与封存不符")
    verify_receipts(price)
    protocol = read(price / "protocol.json")
    if protocol["phases"] != PHASES or protocol["leader_labels_read"] is not False:
        raise ValueError("价格父来源不是固定纯公开研究")
    return price, read(price / "input_identities.json")


def run(source, funding_root, native_review, output):
    """验证并冻结来源、原生选择、全包源码后，重建新时钟和全部统计。"""
    source, funding_root, native_review, output = (
        Path(p).resolve() for p in (source, funding_root, native_review, output)
    )
    if any(
        output.is_relative_to(p) or p.is_relative_to(output)
        for p in (source, funding_root, native_review)
    ):
        raise ValueError("新实验不能覆盖或包含来源")
    verify_receipts(source)
    verify_receipts(native_review)
    parent, native = read(source / "protocol.json"), read(native_review / "result.json")
    if (
        parent["phases"] != PHASES
        or len(parent["universe"]) != 748
        or parent["orders_enabled"] is not False
        or parent["leader_labels_read"] is not False
        or native["selected_hypothesis_ids"] != [HYPOTHESIS_ID]
        or native["orders_enabled"] is not False
        or native["production_eligible"] is not False
    ):
        raise ValueError("必须沿用原全集和原生选择的唯一研究假说")
    price, raw_ids = _price_source(source, parent)
    capture, manifest = (
        read(funding_root / p) for p in ("capture_plan.json", "funding_manifest.json")
    )
    if set(capture["funding_symbols"]) != set(parent["universe"]) or any(
        capture[k] is not False for k in ("orders_enabled", "leader_labels_read")
    ):
        raise ValueError("普通结算来源没有覆盖原纯公开全集")
    identities = {
        str(root / "receipts.json"): sha(root / "receipts.json")
        for root in (source, price, native_review)
    }
    for name in ("capture_plan.json", "funding_manifest.json", "funding_events.json"):
        path = funding_root / name
        if parent["input_identities"].get(str(path)) != sha(path):
            raise ValueError("资金费来源不等于已封存研究输入")
        identities[str(path)] = sha(path)
    identities.update(
        {
            str(funding_source_path(funding_root, leaf)): leaf["sha256"]
            for leaf in manifest["leaves"]
        }
    )
    identities.update(raw_ids)
    for filename, expected in identities.items():
        if sha(Path(filename)) != expected:
            raise ValueError("运行前的原文身份不匹配")
    package = Path(__file__).resolve().parents[2]
    paths = sorted(set(package.rglob("*.py")) | set((Path(__file__).parent / "plans").glob("*.md")))
    paths.append(Path(__file__).parent / "requirements-forward.txt")
    frozen = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "ta-native-settlement-unwind-study.v1",
        "account_family": "settlement_unwind",
        "hypothesis_id": HYPOTHESIS_ID,
        "created_ms": time.time_ns() // 1_000_000,
        "research_owner": "TradingAgents",
        "features": FEATURES,
        "models": MODELS,
        "contrasts": CONTRASTS,
        "phases": PHASES,
        "universe": parent["universe"],
        "horizons_minutes": [20],
        "decision_utc_minute": 5,
        "decision_latency_ms": MINUTE,
        "label_end_delay_ms": LABEL_DELAY_MS,
        "label": "close(t+21min)/open_of_minute_starting(t+1min)-1",
        "price_rows_indexed_by_completed_close_boundary": True,
        "estimator": "pooled_ols_with_intercept",
        "training_standard_deviation_ddof": 0,
        "standardization_population": "mature_common_features_before_target_missingness_filter",
        "common_feature_population": FEATURES,
        "refit_times_ms": REFITS,
        "fixed_uses_first_refit": True,
        "predicted_return_threshold": 0.0016,
        "account_slippage_bps": [3, 10, 11],
        "feature_event_types": ["Regular"],
        "account_event_types": ["Regular", "Special"],
        "input_identities": identities,
        "source_sha256": frozen,
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "real_time_forward_required": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")},
        },
    }
    save(output / "protocol.json", protocol)
    for path in paths:
        target = output / "source" / path.relative_to(package.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    ordinary, event_counts = regular_events(funding_root, capture, manifest)
    events, raw_sources = defaultdict(list), defaultdict(list)
    for event in ordinary:
        if event["symbol"] in parent["universe"]:
            events[event["symbol"]].append(event)
    for name, expected in raw_ids.items():
        path = Path(name).resolve()
        if path.parent.name not in parent["universe"] or path.suffix != ".json":
            raise ValueError("分钟原文不属于固定公开全集")
        raw_sources[path.parent.name].append((path, expected))
    if set(raw_sources) != set(parent["universe"]):
        raise ValueError("分钟原文没有覆盖原748合约")
    calendar = np.arange(ms("2026-09-28T00:01:00Z"), ms("2026-10-15T00:00:00Z") + 1, MINUTE)
    hours = np.arange(
        TRAINING_START_MS + 5 * MINUTE, ms(PHASES["reserved_later_day"]["end"]), 60 * MINUTE
    )
    prepared, coverage = [], []
    first, last = ms(capture["start"]), ms(capture["end"]) - 1
    for number, symbol in enumerate(sorted(parent["universe"]), 1):
        rows, seen = [], set()
        for path, expected in sorted(raw_sources[symbol]):
            if sha(path) != expected:
                raise ValueError("分钟原文在读取前改变")
            rows.extend(validated_rows(read(path), symbol, seen))
        minutes = (
            pd.DataFrame(
                rows, columns=["clock_ms", "open", "close", "quote_volume", "taker_buy_quote"]
            )
            .set_index("clock_ms")
            .reindex(calendar)
        )
        frame = price_features(minutes).reindex(hours)
        frame = frame.join(funding_factors(hours, events[symbol], first, last))
        frame = frame.join(settlement_factor(hours, events[symbol], first, last))
        contract = parent["universe"][symbol]
        eligible = (hours > contract["onboard_ms"]) & (
            (contract["delivery_ms"] == 0) | (hours < contract["delivery_ms"])
        )
        frame.loc[~eligible, list(FEATURES)] = np.nan
        frame["symbol"], frame["asset_class"], frame["clock_ms"] = (
            symbol,
            contract["asset_class"],
            hours,
        )
        frame["phase_20"] = [phase_at(int(clock)) for clock in hours]
        frame.loc[frame.phase_20 == "excluded_boundary", "target_20"] = np.nan
        complete = np.isfinite(frame[list(FEATURES)]).all(axis=1)
        coverage.append(
            {
                "symbol": symbol,
                "scheduled_rows": len(frame),
                "complete_features": int(complete.sum()),
                "h01_known": int(np.isfinite(frame[H01_NAME]).sum()),
                "h01_known_zero": int((frame[H01_NAME] == 0).sum()),
                "h01_nonzero": int((np.isfinite(frame[H01_NAME]) & (frame[H01_NAME] != 0)).sum()),
            }
        )
        prepared.append(frame.reset_index(drop=True))
        if number % 100 == 0:
            print({"prepared_symbols": number, "total": len(parent["universe"])}, flush=True)
    samples = pd.concat(prepared, ignore_index=True)
    samples.to_parquet(output / "hourly_samples.parquet", index=False)
    save(
        output / "coverage.json",
        {
            "symbols": coverage,
            "sample_rows": len(samples),
            "common_feature_rows": sum(r["complete_features"] for r in coverage),
            "funding_event_types": event_counts,
            "historical_first_publication_time_proven": False,
        },
    )
    coefficients = fit_coefficients(samples)
    save(output / "coefficients.json", coefficients)
    result, hourly, predictions = evaluate(samples, coefficients)
    save(output / "hourly_predictions.json", hourly)
    predictions.to_parquet(output / "hourly_predictions.parquet", index=False)
    save(output / "attribution.json", attribution(samples, coefficients, predictions))
    save(output / "result.json", result)
    verify_receipts(source)
    verify_receipts(price)
    verify_receipts(native_review)
    if any(sha(Path(p)) != expected for p, expected in (identities | frozen).items()):
        raise ValueError("研究计算期间源码或输入发生改变")
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result


def main(argv=None):
    """统一CLI与直接模块执行共用显式来源，不隐式查找生产配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--funding-root", required=True)
    parser.add_argument("--native-review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run(args.source, args.funding_root, args.native_review, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
