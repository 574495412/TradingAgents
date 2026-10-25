"""固定 Alpha158 增量岭回归：先封存来源，再连接原收益标签。

特征和共同人口先于未来标签确定；只输出研究证据，不运行交易账户或下单。
"""

import argparse
import importlib.metadata
import math
import shutil
import sys
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .artifacts import read, save, sha, verify_receipts
from .factor_diagnostics import evaluate as diagnose
from .factors.alpha158 import NAMES as ALPHA_FEATURES, PROFILE
from .factors.funding import NAMES as FUNDING_FEATURES
from .regression import FEATURES as PRICE_FEATURES, bh_adjust, correlation, daily_stat
from .regression_run import MINUTE, PHASES, ms
from .ridge import DAY_MS, PENALTIES, predict as ridge_predict, select_penalty

IDENTITY = ("symbol", "asset_class", "clock_ms")
BASELINE_FEATURES = PRICE_FEATURES + FUNDING_FEATURES
FEATURES = BASELINE_FEATURES + ALPHA_FEATURES
MODELS = {
    "baseline_ridge_9": BASELINE_FEATURES,
    "baseline_alpha158_ridge_167": FEATURES,
}
METHODS = ("fixed", "rolling")
REFITS = tuple(ms(f"2026-10-{day:02}T00:00:00Z") for day in (6, 8, 10, 12))
TRAINING_START_MS = ms(PHASES["training"]["start"])
END_MS = ms(PHASES["retrospective_validation"]["end"])
LABEL_DELAY_MS = 21 * MINUTE
HOUR_MS = 60 * MINUTE
LABEL_COLUMNS = ("target_20", "phase_20", "holding_path_observed")
MINIMUM_MEMBERS = 20


def phase_at(clock):
    """准确退出标签须严格早于阶段终点。"""
    return next(
        (
            name
            for name, bounds in PHASES.items()
            if ms(bounds["start"]) <= clock and clock + LABEL_DELAY_MS < ms(bounds["end"])
        ),
        "excluded_boundary",
    )


def _identity(frame):
    """禁止重复身份、跨资产同名合约及非 HH:05 时钟。"""
    if frame.columns.has_duplicates or not set(IDENTITY) <= set(frame):
        raise ValueError("研究表缺少唯一身份列")
    if frame[list(IDENTITY)].isna().any().any() or frame.duplicated(list(IDENTITY)).any():
        raise ValueError("研究表身份未知或重复")
    if any(not isinstance(v, str) or not v for v in frame.symbol) or any(
        not isinstance(v, str) or not v for v in frame.asset_class
    ):
        raise ValueError("合约和资产类别必须为非空字符串")
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, Integral)
        or v < 0
        or v % HOUR_MS != 5 * MINUTE
        for v in frame.clock_ms
    ):
        raise ValueError("研究只接受 UTC 每小时第五分钟的整数时钟")
    if frame.groupby("symbol").asset_class.nunique().gt(1).any():
        raise ValueError("同一合约不能跨资产类别")


def join_labels(features, labels):
    """按完整三元身份一对一连接；未来标签不能改变共同特征门。"""
    _identity(features)
    _identity(labels)
    if set(features) != {*IDENTITY, *FEATURES} or not {*BASELINE_FEATURES, *LABEL_COLUMNS} <= set(
        labels
    ):
        raise ValueError("特征必须精确覆盖固定167列，标签须来自原研究白名单")
    feature_keys = pd.MultiIndex.from_frame(features[list(IDENTITY)])
    label_keys = pd.MultiIndex.from_frame(labels[list(IDENTITY)])
    positions = label_keys.get_indexer(feature_keys)
    if len(features) != len(labels) or (positions < 0).any():
        raise ValueError("仅特征表与原标签表没有保留完全相同的身份全集")
    original = labels.iloc[positions].reset_index(drop=True)
    for name in BASELINE_FEATURES:
        if not np.array_equal(features[name].to_numpy(), original[name].to_numpy(), equal_nan=True):
            raise ValueError(f"旧九项特征与原封存样本不同：{name}")
    if np.isinf(original.target_20.to_numpy(dtype=float)).any():
        raise ValueError("原标签不能含无穷数")
    expected = [phase_at(int(clock)) for clock in features.clock_ms]
    if original.phase_20.tolist() != expected or any(
        not isinstance(value, (bool, np.bool_)) for value in original.holding_path_observed
    ):
        raise ValueError("标签阶段或完整持仓路径标志与原固定时钟不符")
    outside = np.asarray(expected) == "excluded_boundary"
    if np.isfinite(original.target_20.to_numpy(dtype=float)[outside]).any():
        raise ValueError("阶段边界排除行不能带可用于评分的标签")
    result = features.reset_index(drop=True).copy(deep=False)
    for name in LABEL_COLUMNS:
        result[name] = original[name].to_numpy()
    result["baseline_feature_complete"] = np.isfinite(
        result[list(BASELINE_FEATURES)].to_numpy(dtype=float)
    ).all(axis=1)
    result["common_feature_complete"] = np.isfinite(
        result[list(FEATURES)].to_numpy(dtype=float)
    ).all(axis=1)
    counts = result.groupby(["asset_class", "clock_ms"])["common_feature_complete"].transform("sum")
    result["common_feature_members_at_clock"] = counts.astype(int)
    result["common_member_gate"] = result.common_feature_complete & (counts >= MINIMUM_MEMBERS)
    return result


def fit_coefficients(samples):
    """全部四重训、两资产和两个模型保留内层网格、最终拟合及失败。"""
    _identity(samples)
    assets = sorted(samples.asset_class.unique())
    if len(assets) != 2 or not {"common_member_gate", "target_20", *FEATURES} <= set(samples):
        raise ValueError("必须保留两个资产类别与固定共同人口")
    refits = {}
    for refit, end in zip(REFITS, (*REFITS[1:], END_MS), strict=True):
        groups = {}
        folds = ((refit - 4 * DAY_MS, refit - 2 * DAY_MS), (refit - 2 * DAY_MS, refit))
        for asset, frame in samples.groupby("asset_class", sort=True):
            # 训练可用成员只由本钟完整167项和预定人数门决定，不由标签决定。
            usable = frame.loc[frame.common_member_gate]
            selections = {
                name: select_penalty(
                    usable,
                    features=columns,
                    common_features=FEATURES,
                    target="target_20",
                    training_start_ms=TRAINING_START_MS,
                    outer_refit_ms=refit,
                    label_delay_ms=LABEL_DELAY_MS,
                    folds=folds,
                    penalties=PENALTIES,
                    minimum_rows=MINIMUM_MEMBERS,
                )
                for name, columns in MODELS.items()
            }
            groups[f"{asset}:20m"] = {"models": selections}
        refits[str(refit)] = {"refit_ms": refit, "end_ms": end, "groups": groups}
    return {
        "schema": "ta-alpha158-ridge-coefficients.v1",
        "features": list(FEATURES),
        "models": MODELS,
        "alpha158_profile": PROFILE,
        "refits": refits,
        "inner_daily_weight": "equal_rows_within_utc_day_then_equal_days",
    }


def record_at(coefficients, asset_class, clock_ms, mode="fixed"):
    """固定方式始终首轮；训练段用首轮参数仅作样本内诊断。"""
    if mode not in METHODS:
        raise ValueError("系数方式只能为 fixed 或 rolling")
    refit = (
        REFITS[0]
        if mode == "fixed" or clock_ms < REFITS[0]
        else max(value for value in REFITS if value <= clock_ms)
    )
    return coefficients["refits"][str(refit)]["groups"][f"{asset_class}:20m"]


def predict(frame, record, model):
    """不读取标签，仅在完整共同截面上消费冻结的内层选择及最终系数。"""
    if model not in MODELS or not set(FEATURES) <= set(frame) or frame.index.has_duplicates:
        raise ValueError("预测设计或索引不属于固定模型")
    if "symbol" in frame and frame.symbol.duplicated().any():
        raise ValueError("预测截面存在重复合约")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    known = np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    if int(known.sum()) < MINIMUM_MEMBERS:
        return result
    selection = record["models"][model]
    if selection["status"] != "READY":
        return result
    path = selection["final_path"]
    if (
        path is None
        or tuple(path["features"]) != MODELS[model]
        or tuple(path["common_features"]) != FEATURES
        or selection["selected_penalty"] not in PENALTIES
    ):
        raise ValueError("冻结参数没有精确覆盖固定特征和共同人口")
    return ridge_predict(frame[list(FEATURES)], path, selection["selected_penalty"])


def _metric(frame, scores, common):
    """共同成员有任何预测缺失就失败，不能按可预测子集救评分。"""
    missing = int((common & ~np.isfinite(scores)).sum())
    valid = common & np.isfinite(frame.target_20)
    if int(common.sum()) < MINIMUM_MEMBERS:
        return None, "INSUFFICIENT_COMMON_MEMBERS"
    if missing:
        return None, "INCOMPLETE_COMMON_PREDICTIONS"
    if int(valid.sum()) < MINIMUM_MEMBERS:
        return None, "INSUFFICIENT_PAIRED_LABELS"
    y, prediction = frame.loc[valid, "target_20"].to_numpy(), scores.loc[valid].to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        errors, null = (y - prediction) ** 2, y @ y
    with np.errstate(over="ignore", invalid="ignore"):
        sse, mse = float(errors.sum()), float(errors.mean())
    if not np.isfinite(errors).all() or not all(math.isfinite(v) for v in (float(null), sse, mse)):
        return None, "NONFINITE_SCORE_METRIC"
    return {
        "members": int(valid.sum()),
        "sse": sse,
        "mse": mse,
        "null_sse": float(null),
        "pearson_ic": correlation(prediction, y),
        "rank_ic": correlation(prediction, y, rank=True),
    }, "READY"


def _summary(rows, days):
    return {
        "scheduled_cross_sections": len(rows),
        "models": {
            model: {
                metric: daily_stat(
                    rows,
                    lambda row, m=model, k=metric: (
                        row["models"][m][k] if row["models"][m] is not None else None
                    ),
                    days,
                )
                for metric in ("mse", "pearson_ic", "rank_ic")
            }
            for model in MODELS
        },
        "paired_mse_improvement": daily_stat(rows, lambda row: row["paired_mse_improvement"], days),
    }


def evaluate(samples, coefficients):
    """输出全部身份的两方式预测、小时评分与四项共同 BH，不选赢家。"""
    assets = sorted(samples.asset_class.unique())
    if len(assets) != 2:
        raise ValueError("四项主比较必须保留两个资产类别")
    records, outputs = [], []
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        phase = phase_at(int(clock))
        if not frame.phase_20.eq(phase).all():
            raise ValueError("样本阶段与预定标签边界不同")
        common = np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
        for mode in METHODS:
            record = record_at(coefficients, asset, int(clock), mode)
            prediction = frame[
                list(IDENTITY)
                + list(LABEL_COLUMNS)
                + [
                    "baseline_feature_complete",
                    "common_feature_complete",
                    "common_member_gate",
                    "common_feature_members_at_clock",
                ]
            ].copy()
            row = {
                "asset_class": asset,
                "clock_ms": int(clock),
                "mode": mode,
                "phase": phase,
                "day": pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                "scheduled_members": len(frame),
                "common_feature_members": int(common.sum()),
                "baseline_feature_members": int(frame.baseline_feature_complete.sum()),
                "known_target_members": int((common & np.isfinite(frame.target_20)).sum()),
                "models": {},
                "status_by_model": {},
                "paired_mse_improvement": None,
            }
            for model in MODELS:
                scores = predict(frame, record, model)
                prediction[f"prediction_{model}"] = scores
                metric, status = _metric(frame, scores, common)
                if phase == "excluded_boundary":
                    metric, status = None, "EXCLUDED_LABEL_BOUNDARY"
                row["models"][model], row["status_by_model"][model] = metric, status
            baseline, full = (row["models"][model] for model in MODELS)
            if baseline is not None and full is not None:
                row["paired_mse_improvement"] = baseline["mse"] - full["mse"]
            prediction["coefficient_mode"] = mode
            outputs.append(prediction)
            records.append(row)
    groups, tests = {}, []
    for asset in assets:
        modes = {}
        for mode in METHODS:
            part = [row for row in records if row["asset_class"] == asset and row["mode"] == mode]
            phases = {}
            for phase, bounds in PHASES.items():
                days = list(
                    pd.date_range(
                        bounds["start"], bounds["end"], inclusive="left", freq="D"
                    ).strftime("%Y-%m-%d")
                )
                phases[phase] = _summary([r for r in part if r["phase"] == phase], days)
            primary = phases["retrospective_validation"]["paired_mse_improvement"]
            primary["test_p"] = primary["p_asymptotic_two_sided"]
            primary["test_tail"] = "two_sided_and_positive_mean_required"
            tests.append(primary)
            modes[mode] = {"phases": phases}
        groups[f"{asset}:20m"] = modes
    if len(tests) != 4:
        raise ValueError("主比较必须恰为四项")
    adjusted = bh_adjust([row["test_p"] if row["test_p"] is not None else 1.0 for row in tests])
    for row, q in zip(tests, adjusted, strict=True):
        row["bh_q_4"] = q if row["test_p"] is not None else None
        row["positive_and_q_at_most_005"] = bool(
            row["mean"] is not None
            and row["mean"] > 0
            and row["bh_q_4"] is not None
            and row["bh_q_4"] <= 0.05
        )
    return {
        "schema": "ta-alpha158-ridge-statistics.v1",
        "groups": groups,
        "sections": records,
        "declared_tests": 4,
        "bh_adjustment_family_size": 4,
        "valid_p_value_tests": sum(row["test_p"] is not None for row in tests),
        "previous_declared_funding_tests": 216,
        "previous_declared_h01_tests": 12,
        "outer_daily_weight": "equal_hourly_paired_mse_within_day_then_equal_days",
        "inner_daily_weight": "equal_rows_within_utc_day_then_equal_days",
        "training_phase_is_in_sample_diagnostic_only": True,
        "independent_oos": False,
        "auxiliary_pvalues_are_not_declared_discoveries": True,
        "economic_account_evaluated": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }, pd.concat(outputs, ignore_index=True)


def prediction_diagnostics(samples, predictions):
    """复用标准诊断；方向固定为预测收益正向，不新增单因子显著性筛选。"""
    populations = samples[list(IDENTITY) + ["common_feature_complete"]]
    results = {}
    for mode in METHODS:
        part = predictions[predictions.coefficient_mode == mode].merge(
            populations, on=list(IDENTITY), validate="one_to_one", suffixes=("", "_source")
        )
        part["common167_available"] = np.where(part.common_feature_complete, 0.0, np.nan)
        for model in MODELS:
            for phase, bounds in PHASES.items():
                calendar = tuple(
                    range(ms(bounds["start"]) + 5 * MINUTE, ms(bounds["end"]), HOUR_MS)
                )
                data = part[part.phase_20 == phase]
                results[f"{mode}:{model}:{phase}"] = diagnose(
                    data[
                        list(IDENTITY) + ["common167_available", "target_20", f"prediction_{model}"]
                    ],
                    score_column=f"prediction_{model}",
                    target_column="target_20",
                    calendar_ms=calendar,
                    scope=f"alpha158:{mode}:{model}:{phase}",
                    common_features=("common167_available",),
                    direction=1,
                    asset_classes=sorted(samples.asset_class.unique()),
                    frequency_ms=HOUR_MS,
                    lag_ms=(HOUR_MS,),
                    minimum_members=MINIMUM_MEMBERS,
                )
    return results


def _source_contract(feature_root, label_root):
    """只在明确的两条封存来源之间连接，不隐式发现或替换标签。"""
    verify_receipts(feature_root)
    verify_receipts(label_root)
    feature, source = read(feature_root / "protocol.json"), read(label_root / "protocol.json")
    sample_path = label_root / "hourly_samples.parquet"
    if (
        feature.get("schema") != "ta-alpha158-feature-coverage.v1"
        or source.get("schema") != "ta-native-settlement-unwind-study.v1"
        or feature.get("alpha158_profile") != PROFILE
        or feature.get("alpha158_source_sha256")
        != sha(Path(__file__).parent / "factors/alpha158.py")
        or Path(feature["source_root"]).resolve() != label_root
        or Path(feature["source_samples_path"]).resolve() != sample_path
        or feature["source_receipts_sha256"] != sha(label_root / "receipts.json")
        or feature["input_identities"].get(str(sample_path)) != sha(sample_path)
        or tuple(feature["features"]) != FEATURES
        or feature["universe"] != source["universe"]
        or source["phases"] != PHASES
        or len(source["universe"]) != 748
        or source["orders_enabled"] is not False
        or source["leader_labels_read"] is not False
    ):
        raise ValueError("特征版本、标签来源、原人口或封存身份不属于固定研究")
    columns = pq.read_schema(feature_root / "features.parquet").names
    if len(columns) != len(IDENTITY) + len(FEATURES) or set(columns) != {*IDENTITY, *FEATURES}:
        raise ValueError("仅特征文件混入标签、额外输入或漏掉固定列")
    return feature, source


def _source_grid(samples, universe):
    """全量来源必须完整保留原748合约乘384小时，资格未知保留为原行。"""
    clocks = np.arange(
        TRAINING_START_MS + 5 * MINUTE, ms(PHASES["reserved_later_day"]["end"]), HOUR_MS
    )
    if set(samples.symbol.unique()) != set(universe) or len(samples) != len(universe) * len(clocks):
        raise ValueError("研究来源没有保留原748完整小时身份")
    for symbol, frame in samples.groupby("symbol", sort=False):
        if (
            not np.array_equal(np.sort(frame.clock_ms.to_numpy()), clocks)
            or not frame.asset_class.eq(universe[symbol]["asset_class"]).all()
        ):
            raise ValueError("研究来源的时钟或资产身份与固定人口不一致")


def run(feature_source, label_source, output):
    """先冻结全部消费身份和源码，再读标签；成功与工程失败都留收据。"""
    feature_root, label_root, output = (
        Path(p).resolve() for p in (feature_source, label_source, output)
    )
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (feature_root, label_root)
    ):
        raise ValueError("新研究不能覆盖或包含原来源")
    feature, source = _source_contract(feature_root, label_root)
    identities = {
        str(root / name): sha(root / name)
        for root, names in (
            (feature_root, ("protocol.json", "receipts.json", "features.parquet")),
            (label_root, ("protocol.json", "receipts.json", "hourly_samples.parquet")),
        )
        for name in names
    }
    package = Path(__file__).resolve().parents[2]
    paths = sorted(set(package.rglob("*.py")) | set((Path(__file__).parent / "plans").glob("*.md")))
    requirement = Path(__file__).parent / "requirements-forward.txt"
    if requirement.exists():
        paths.append(requirement)
    frozen = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-alpha158-ridge-study.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "research_owner": "TradingAgents",
            "account_family": "alpha158_ridge",
            "alpha158_profile": PROFILE,
            "features": FEATURES,
            "baseline_features": BASELINE_FEATURES,
            "models": MODELS,
            "penalties": PENALTIES,
            "refit_times_ms": REFITS,
            "inner_folds_by_refit": {
                str(r): ((r - 4 * DAY_MS, r - 2 * DAY_MS), (r - 2 * DAY_MS, r)) for r in REFITS
            },
            "phases": PHASES,
            "universe": source["universe"],
            "horizons_minutes": [20],
            "label_end_delay_ms": LABEL_DELAY_MS,
            "decision_utc_minute": 5,
            "label": "close(t+21min)/open_of_minute_starting(t+1min)-1",
            "price_rows_indexed_by_completed_close_boundary": True,
            "estimator": "pooled_ridge_with_unpenalized_intercept",
            "training_standard_deviation_ddof": 0,
            "minimum_common_members": MINIMUM_MEMBERS,
            "common_population_uses_labels": False,
            "predicted_return_threshold": 0.0016,
            "account_slippage_bps": [3, 10, 11],
            "inner_daily_weight": "equal_rows_within_utc_day_then_equal_days",
            "outer_daily_weight": "equal_hourly_paired_mse_within_day_then_equal_days",
            "declared_tests": 4,
            "bh_unknown_p_value_input": 1.0,
            "hac_lag_days": 3,
            "source_sha256": frozen,
            "input_identities": identities,
            "feature_source": str(feature_root),
            "label_source": str(label_root),
            "eligibility_assumption": feature.get("eligibility_assumption"),
            "all_history_previously_viewed": True,
            "independent_oos": False,
            "real_time_forward_required": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
            "runtime": {
                "python": sys.version,
                "libraries": {
                    n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")
                },
            },
        },
    )
    for path in paths:
        target = output / "source" / path.relative_to(package.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    stage = "load_join"
    try:
        features = pd.read_parquet(
            feature_root / "features.parquet", columns=[*IDENTITY, *FEATURES]
        )
        labels = pd.read_parquet(
            label_root / "hourly_samples.parquet",
            columns=[*IDENTITY, *BASELINE_FEATURES, *LABEL_COLUMNS],
        )
        samples = join_labels(features, labels)
        _source_grid(samples, source["universe"])
        samples.to_parquet(output / "hourly_samples.parquet", index=False)
        del features, labels
        stage = "fit_coefficients"
        coefficients = fit_coefficients(samples)
        save(output / "coefficients.json", coefficients)
        stage = "evaluate"
        result, predictions = evaluate(samples, coefficients)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        stage = "prediction_diagnostics"
        save(output / "prediction_diagnostics.json", prediction_diagnostics(samples, predictions))
        stage = "verify_after"
        verify_receipts(feature_root)
        verify_receipts(label_root)
        if any(sha(path) != identity for path, identity in (identities | frozen).items()):
            raise ValueError("研究计算期间消费来源或源码发生变化")
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {
                "stage": stage,
                "type": type(error).__name__,
                "reason": str(error),
                "numeric_conclusions_allowed": False,
            },
        )
        raise
    finally:
        save(
            output / "receipts.json",
            {
                str(path.relative_to(output)): sha(path)
                for path in output.rglob("*")
                if path.is_file()
            },
        )
    return result


def main(argv=None):
    """仅显式来源；不隐式加载生产配置、市场客户端或账户。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-source", "label-source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.feature_source, args.label_source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
