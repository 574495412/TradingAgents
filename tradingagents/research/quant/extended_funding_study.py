"""固定资金费家族的两月续检，复用旧截面估计器而不搜索参数。"""

import argparse
import importlib.metadata
import shutil
import sys
import time
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from . import linear_family, regression
from .artifacts import read, save, sha, verify_receipts
from .factors.funding import NAMES
from .funding_study import CONTRASTS, MODELS
from .regression_run import ms

MINUTE = 60_000
HOUR = 60 * MINUTE
START = ms("2026-08-01T00:00:00Z")
END = ms("2026-10-01T00:00:00Z")
REFITS = tuple(
    ms(value)
    for value in (
        "2026-08-16T00:00:00Z",
        "2026-08-23T00:00:00Z",
        "2026-08-30T00:00:00Z",
        "2026-09-06T00:00:00Z",
        "2026-09-13T00:00:00Z",
        "2026-09-20T00:00:00Z",
        "2026-09-27T00:00:00Z",
    )
)
ASSETS = ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL")
METHODS = ("fixed", "rolling")
FEATURES = regression.FEATURES + NAMES
IDENTITY = ("symbol", "asset_class", "clock_ms")
FEATURE_COLUMNS = (*IDENTITY, *FEATURES, "feature_eligible")
LABEL_COLUMNS = (*IDENTITY, "target_240")
EXPECTED_SYMBOLS = 909
LABEL_DELAY = 241 * MINUTE
PLAN = Path(__file__).parent / "plans/extended_funding_plan.md"


def _validate(features, labels, universe):
    """完整双表身份先核对，未来目标的缺失不改变当期特征人口。"""
    if set(features) != set(FEATURE_COLUMNS) or set(labels) != set(LABEL_COLUMNS):
        raise ValueError("续检只接受明确特征及240分钟标签白名单")
    clocks = np.arange(START, END, HOUR)
    for frame in (features, labels):
        if (
            frame.columns.has_duplicates
            or frame[list(IDENTITY)].isna().any().any()
            or frame.duplicated(list(IDENTITY)).any()
            or len(universe) != EXPECTED_SYMBOLS
            or set(frame.symbol) != set(universe)
            or len(frame) != EXPECTED_SYMBOLS * len(clocks)
        ):
            raise ValueError("续检必须保留全部原定符号和小时身份")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in frame.clock_ms
        ):
            raise ValueError("身份时钟必须是整数毫秒")
        for symbol, part in frame.groupby("symbol", sort=False):
            asset = universe[symbol]["asset_class"]
            if (
                asset not in (*ASSETS, "UNKNOWN")
                or not part.asset_class.eq(asset).all()
                or not np.array_equal(np.sort(part.clock_ms.to_numpy()), clocks)
            ):
                raise ValueError("资产类别或完整固定日历不同")
    if (
        not pd.api.types.is_bool_dtype(features.feature_eligible)
        or features.feature_eligible.isna().any()
    ):
        raise ValueError("静态资格假设必须明确布尔，不得未知补真")
    for frame, columns in ((features, FEATURES), (labels, ("target_240",))):
        for column in columns:
            if (
                not pd.api.types.is_numeric_dtype(frame[column])
                or pd.api.types.is_bool_dtype(frame[column])
                or np.isinf(frame[column].to_numpy(dtype=float)).any()
            ):
                raise ValueError("特征及标签须为实数有限值或明确NaN")
    joined = features.merge(
        labels, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True
    )
    if not joined._merge.eq("both").all():
        raise ValueError("特征与标签不是完全相同身份的双射")
    return joined.drop(columns="_merge").sort_values(list(IDENTITY)).reset_index(drop=True)


def _fit_hours(frame):
    """仅记录可供某个已登记重训使用的成熟小时，五模型共用拟合时钟。"""
    rows = []
    prior = frame[frame.clock_ms + LABEL_DELAY < REFITS[-1]]
    for (asset, clock), part in prior.groupby(["asset_class", "clock_ms"], sort=True):
        if asset not in ASSETS:
            continue
        eligible = part[part.feature_eligible]
        try:
            _, _, fits, failures = linear_family.fit_section(
                eligible, "target_240", FEATURES, MODELS
            )
        except np.linalg.LinAlgError:
            fits, failures = {}, dict.fromkeys(MODELS, "LINEAR_ALGEBRA_FAILURE")
        for name in list(fits):
            if not np.isfinite(list(fits[name].values())).all():
                fits.pop(name)
                failures[name] = "NONFINITE_COEFFICIENT"
        rows.append(
            {"asset_class": asset, "clock_ms": int(clock), "fits": fits, "failures": failures}
        )
    return rows


def _snapshots(fits):
    """严格按标签结束筛选，禁止回退到旧系数或用零系数伪装成功。"""
    snapshots = {}
    for refit in REFITS:
        groups = {}
        for asset in ASSETS:
            mature = [
                r for r in fits if r["asset_class"] == asset and r["clock_ms"] + LABEL_DELAY < refit
            ]
            common = [r for r in mature if len(r["fits"]) == len(MODELS)]
            coefficients = {
                name: {
                    column: float(np.mean([r["fits"][name][column] for r in common]))
                    for column in columns
                }
                if common
                else None
                for name, columns in MODELS.items()
            }
            status = "READY" if common else "NO_COMMON_FIT_HOURS"
            if common and any(
                not np.isfinite(list(beta.values())).all() for beta in coefficients.values()
            ):
                coefficients = dict.fromkeys(MODELS)
                status = "NONFINITE_AGGREGATE_COEFFICIENT"
            groups[asset] = {
                "status": status,
                "training_common_fit_hours": len(common),
                "common_training_clocks_ms": [r["clock_ms"] for r in common],
                "maximum_training_label_end_ms": max(
                    (r["clock_ms"] + LABEL_DELAY for r in common), default=None
                ),
                "frozen_training_coefficients": coefficients,
                "mature_hours": len(mature),
                "failure_counts": dict(
                    Counter(
                        f"{name}:{reason}" for r in mature for name, reason in r["failures"].items()
                    )
                ),
            }
        snapshots[str(refit)] = {"refit_ms": refit, "groups": groups}
    return snapshots


def _metric(known, scores):
    """任一共同成员预测未知使整钟失效；标签仅用于事后评分。"""
    if len(known) < 20 or not np.isfinite(scores).all():
        return None
    valid = np.isfinite(known.target_240.to_numpy())
    if valid.sum() < 20:
        return None
    y = known.loc[valid, "target_240"].to_numpy()
    y = y - y.mean()
    prediction = scores[valid]
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean((prediction - y) ** 2))
    if not np.isfinite(mse):
        return None
    return {
        "members": int(valid.sum()),
        "mse": mse,
        "ic": regression.correlation(prediction, y),
        "rank_ic": regression.correlation(prediction, y, rank=True),
    }


def analyze(features, labels, universe):
    """返回完整预测、全部检验与拟合诊断；固定截止而不接受结果调参。"""
    frame = _validate(features, labels, universe)
    fits = _fit_hours(frame)
    snapshots = _snapshots(fits)
    common = (
        frame.feature_eligible
        & np.isfinite(frame[list(FEATURES)]).all(axis=1)
        & frame.asset_class.isin(ASSETS)
    )
    counts = common.groupby([frame.asset_class, frame.clock_ms]).transform("sum")
    frame["common_member_gate"] = common & (counts >= 20)
    outputs, records, factor_records = [], [], []
    for method in METHODS:
        predicted = frame[[*IDENTITY, "common_member_gate"]].copy()
        predicted["coefficient_mode"] = method
        for name in MODELS:
            predicted[f"prediction_{name}"] = np.nan
        for (asset, clock), group in frame.groupby(["asset_class", "clock_ms"], sort=True):
            if asset not in ASSETS or clock < REFITS[0]:
                continue
            cutoff = (
                REFITS[0] if method == "fixed" else next(r for r in reversed(REFITS) if r <= clock)
            )
            state = snapshots[str(cutoff)]["groups"][asset]
            known = group[group.common_member_gate].sort_values("symbol")
            ranks = (
                ((known[list(FEATURES)].rank(method="average") - 1) / (len(known) - 1) - 0.5)
                if len(known) >= 20
                else None
            )
            row = {
                "asset_class": asset,
                "clock_ms": int(clock),
                "day": pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                "coefficient_mode": method,
                "refit_ms": cutoff,
                "common_members": len(known),
                "models": {},
                "contrasts": {},
            }
            for name, columns in MODELS.items():
                beta = state["frozen_training_coefficients"][name]
                scores = np.full(len(known), np.nan)
                if ranks is not None and beta is not None:
                    with np.errstate(over="ignore", invalid="ignore"):
                        scores = sum(ranks[f].to_numpy() * beta[f] for f in columns)
                    if not np.isfinite(scores).all():
                        scores = np.full(len(known), np.nan)
                predicted.loc[known.index, f"prediction_{name}"] = scores
                row["models"][name] = _metric(known, scores)
            for name, (baseline, candidate) in CONTRASTS.items():
                left, right = row["models"][baseline], row["models"][candidate]
                row["contrasts"][name] = (
                    left["mse"] - right["mse"] if left is not None and right is not None else None
                )
            records.append(row)
            if method == "fixed":
                valid = np.isfinite(known.target_240.to_numpy())
                factor_records.append(
                    {
                        "asset_class": asset,
                        "clock_ms": int(clock),
                        "day": row["day"],
                        "factor_rank_ic": {
                            name: regression.correlation(
                                known.loc[valid, name], known.loc[valid, "target_240"], rank=True
                            )
                            if valid.sum() >= 20
                            else None
                            for name in NAMES
                        },
                    }
                )
        outputs.append(predicted)
    days = (
        pd.date_range(
            pd.Timestamp(REFITS[0], unit="ms", tz="UTC"),
            pd.Timestamp(END, unit="ms", tz="UTC"),
            inclusive="left",
            freq="D",
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )
    groups, tests = {}, []
    for method in METHODS:
        for asset in ASSETS:
            part = [
                r for r in records if r["coefficient_mode"] == method and r["asset_class"] == asset
            ]
            comparisons = {
                name: regression.daily_stat(part, lambda r, n=name: r["contrasts"][n], days)
                for name in CONTRASTS
            }
            tests.extend(comparisons.values())
            groups[f"{method}:{asset}"] = {
                "comparisons": comparisons,
                "model_metrics": {
                    name: {
                        metric: regression.daily_stat(
                            part,
                            lambda r, n=name, m=metric: (
                                r["models"][n][m] if r["models"][n] is not None else None
                            ),
                            days,
                        )
                        for metric in ("mse", "ic", "rank_ic")
                    }
                    for name in MODELS
                },
                "scheduled_hours": len(part),
                "common_feature_hours": sum(r["common_members"] >= 20 for r in part),
            }
    factor_results = {}
    for asset in ASSETS:
        part = [r for r in factor_records if r["asset_class"] == asset]
        factor_results[asset] = {
            name: regression.daily_stat(part, lambda r, n=name: r["factor_rank_ic"][n], days)
            for name in NAMES
        }
        tests.extend(factor_results[asset].values())
    adjusted = regression.bh_adjust(
        [
            value["p_asymptotic_two_sided"] if value["p_asymptotic_two_sided"] is not None else 1.0
            for value in tests
        ]
    )
    for value, q in zip(tests, adjusted, strict=True):
        p = value["p_asymptotic_two_sided"]
        value["bh_q_all_24_declared_tests"] = q if p is not None else None
        value["passes_exploratory_screen"] = p is not None and value["mean"] > 0 and q <= 0.05
    result = {
        "schema": "ta-extended-funding-regression-result.v1",
        "groups": groups,
        "raw_factor_rank_ic": factor_results,
        "declared_tests": len(tests),
        "bh_adjustment_family_size": len(tests),
        "valid_p_value_tests": sum(v["p_asymptotic_two_sided"] is not None for v in tests),
        "source_rows": len(frame),
        "prediction_rows": sum(map(len, outputs)),
        "orders_enabled": False,
        "leader_labels_read": False,
        "independent_oos": False,
        "auxiliary_quantile_diagnostics_complete": False,
        "economic_accounts_complete": False,
        "stable_profitability_proven": False,
        "production_eligible": False,
    }
    return pd.concat(outputs, ignore_index=True), result, snapshots, records, fits


def run(source, output):
    """先冻结实际依赖，再读两表；预测与未来标签分别保存。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("续检输出不能覆盖或包含原小时来源")
    verify_receipts(source)
    parent = read(source / "protocol.json")
    if read(source / "result.json").get("status") not in {"COMPLETED", "COMPLETED_WITH_UNKNOWN"}:
        raise ValueError("小时来源未成功完成，不能消费失败或部分面板")
    if (
        parent.get("schema") != "ta-historical-funding-panel.v1"
        or parent["start_ms"] != START
        or parent["end_ms"] != END
        or tuple(parent["features"]) != FEATURES
        or parent.get("study_plan_sha256") != sha(PLAN)
        or any(
            parent[key] is not False
            for key in ("orders_enabled", "leader_labels_read", "production_eligible")
        )
    ):
        raise ValueError("续检只接受固定两月纯公开完整小时来源")
    used = (
        Path(__file__),
        Path(linear_family.__file__),
        Path(regression.__file__),
        Path(__file__).parent / "funding_study.py",
        Path(__file__).parent / "artifacts.py",
        Path(__file__).parent / "regression_run.py",
        Path(__file__).parent / "factors/funding.py",
        Path(__file__).parent / "__main__.py",
        PLAN,
    )
    frozen = {str(path.resolve()): sha(path) for path in used}
    identities = {
        str(source / name): sha(source / name)
        for name in (
            "protocol.json",
            "result.json",
            "receipts.json",
            "features.parquet",
            "labels.parquet",
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-extended-funding-study.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "source": str(source),
            "start_ms": START,
            "end_ms": END,
            "refit_times_ms": REFITS,
            "universe": parent["universe"],
            "models": MODELS,
            "contrasts": CONTRASTS,
            "features": FEATURES,
            "feature_columns_loaded": FEATURE_COLUMNS,
            "label_columns_loaded": LABEL_COLUMNS,
            "horizon_minutes": 240,
            "label_delay_ms": LABEL_DELAY,
            "source_sha256": frozen,
            "input_identities": identities,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
            "independent_oos": False,
            "runtime": {
                "python": sys.version,
                "libraries": {
                    n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")
                },
            },
        },
    )
    try:
        for index, path in enumerate(used):
            target = output / "source" / f"{index:02}_{path.name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
        features = pd.read_parquet(source / "features.parquet", columns=list(FEATURE_COLUMNS))
        labels = pd.read_parquet(source / "labels.parquet", columns=list(LABEL_COLUMNS))
        predictions, result, snapshots, hourly, fits = analyze(features, labels, parent["universe"])
        predictions.to_parquet(output / "predictions.parquet", index=False)
        for name, value in (
            ("training.json", snapshots),
            ("hourly_results.json", hourly),
            ("fit_diagnostics.json", fits),
        ):
            save(output / name, value)
        verify_receipts(source)
        if any(sha(path) != value for path, value in (identities | frozen).items()):
            raise ValueError("计算期间实际输入或执行依赖发生变化")
        result["protocol_sha256"] = sha(output / "protocol.json")
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
