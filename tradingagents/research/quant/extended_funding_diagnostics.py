"""固定十二分数的两月资金费诊断；不选择赢家、执行账户或重报显著性发现。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import sys
import time
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from . import extended_funding_study as study, factor_diagnostics, regression
from .artifacts import read, save, sha, verify_receipts

START, END = study.REFITS[0], study.END
HOUR = study.HOUR
ASSETS, FEATURES, IDENTITY = study.ASSETS, study.FEATURES, study.IDENTITY
MODES = study.METHODS
MODEL_NAMES = tuple(study.MODELS)
RAW_NAMES = tuple(name for name in FEATURES if name.startswith("funding_"))
PREDICTION_COLUMNS = (
    *IDENTITY,
    "coefficient_mode",
    "common_member_gate",
    *(f"prediction_{name}" for name in MODEL_NAMES),
)
COMMON_MASK = "diagnostic_common_gate"
SCOPE = "fixed_46_day_retrospective_extension_previously_selected_family"
SCORES = tuple(
    {"id": name, "kind": "raw_regular_funding_factor", "column": name, "direction": 1}
    for name in RAW_NAMES
) + tuple(
    {
        "id": f"{mode}__{name}",
        "kind": "frozen_cross_section_relative_price_return_prediction",
        "mode": mode,
        "model": name,
        "column": f"prediction_{name}",
        "direction": 1,
    }
    for mode in MODES
    for name in MODEL_NAMES
)


def validate_inputs(features, labels, predictions, universe):
    """完整原双表与每种预测方式先双射，因果人口只由当时特征决定。"""
    base = study._validate(features, labels, universe)
    if set(predictions.columns) != set(PREDICTION_COLUMNS) or predictions.columns.has_duplicates:
        raise ValueError("诊断预测只接受固定身份、两方式、共同门及五模型白名单")
    if (
        set(predictions.coefficient_mode) != set(MODES)
        or predictions[list(IDENTITY)].isna().any().any()
    ):
        raise ValueError("预测方式缺失或身份未知")
    if (
        not pd.api.types.is_bool_dtype(predictions.common_member_gate)
        or predictions.common_member_gate.isna().any()
    ):
        raise ValueError("预测共同门必须为明确布尔值")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in predictions.clock_ms
    ):
        raise ValueError("预测身份时钟必须为整数毫秒")
    for name in MODEL_NAMES:
        values = predictions[f"prediction_{name}"]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or np.isinf(values.to_numpy(dtype=float)).any()
        ):
            raise ValueError("预测须为有限实数或NaN")
    raw_common = (
        base.feature_eligible
        & np.isfinite(base[list(FEATURES)]).all(axis=1)
        & base.asset_class.isin(ASSETS)
    )
    counts = raw_common.groupby([base.asset_class, base.clock_ms]).transform("sum")
    base["common_member_gate"] = raw_common & (counts >= 20)
    expected = base[[*IDENTITY, "common_member_gate"]]
    for mode in MODES:
        part = predictions[predictions.coefficient_mode == mode]
        if len(part) != len(base) or part.duplicated(list(IDENTITY)).any():
            raise ValueError("每种预测方式必须完整保留909个原定小时身份")
        compared = expected.merge(
            part,
            on=list(IDENTITY),
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_expected", "_actual"),
        )
        if (
            not compared._merge.eq("both").all()
            or not compared.common_member_gate_expected.eq(compared.common_member_gate_actual).all()
        ):
            raise ValueError("预测身份或共同人口与原特征不一致")
        for name in MODEL_NAMES:
            if (
                compared.loc[~compared.common_member_gate_expected, f"prediction_{name}"]
                .notna()
                .any()
            ):
                raise ValueError("共同人口之外出现未声明预测")
    return base, predictions


def scope_identities(base):
    """UNKNOWN保留在完整范围证据中，不能混入两种已知资产诊断。"""
    frame = base[(base.clock_ms >= START) & (base.clock_ms < END)].copy()
    identities = frame[[*IDENTITY, "feature_eligible", "common_member_gate"]].copy()
    identities["diagnostic_asset_status"] = np.where(
        frame.asset_class.isin(ASSETS), "DECLARED_ASSET", "EXCLUDED_UNKNOWN_ASSET_CLASS"
    )
    frame[COMMON_MASK] = np.where(frame.common_member_gate, 1.0, np.nan)
    return frame[frame.asset_class.isin(ASSETS)].copy(), identities


def evaluate_score(base, predictions, score):
    """每次只形成一个分数的成员对象，未来标签不参与分位或持久性。"""
    if score not in SCORES:
        raise ValueError("分数不属于事前固定十二项")
    columns = [*IDENTITY, *FEATURES, "target_240", COMMON_MASK]
    frame = base[columns].copy()
    column = score["column"]
    if "mode" in score:
        part = predictions.loc[predictions.coefficient_mode == score["mode"], [*IDENTITY, column]]
        frame = frame.merge(
            part, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True
        )
        if not frame._merge.eq("both").all():
            raise ValueError("固定分数缺少已核共同身份")
        frame = frame.drop(columns="_merge")
    result = factor_diagnostics.evaluate(
        frame,
        score_column=column,
        target_column="target_240",
        calendar_ms=tuple(range(START, END, HOUR)),
        scope=SCOPE,
        common_features=(*FEATURES, COMMON_MASK),
        direction=1,
        asset_classes=ASSETS,
        frequency_ms=HOUR,
        lag_ms=(HOUR, 24 * HOUR),
        minimum_members=20,
    )
    # 原诊断器内部保留HAC实现；此适配只输出描述性日序列与ICIR，不重发p或t。
    for asset in result["summary"].values():
        for information in asset["information"].values():
            daily = information["daily"]
            hac = daily.pop("hac_mean")
            daily["daily_values"] = hac["daily_values"]
            daily["scheduled_days"] = len(hac["daily_values"])
    result["score"] = score
    result["new_p_values_emitted"] = False
    result["raw_ic_is_not_an_additional_independent_discovery"] = True
    result["existing_24_test_family_unchanged"] = True
    return result


def summarize(result):
    """保留全部状态和标签计数；不把未知、常量或无预测改成零表现。"""
    output = {}
    for asset in ASSETS:
        rows = [row for row in result["sections"] if row["asset_class"] == asset]
        output[asset] = {
            "statistics": result["summary"][asset],
            "section_status_counts": dict(Counter(row["status"] for row in rows)),
            "ic_status_counts": dict(Counter(row["ic_status"] for row in rows)),
            "scheduled_periods": len(rows),
            "scheduled_member_rows": sum(row["scheduled_members"] for row in rows),
            "common_member_rows": sum(row["common_feature_members"] for row in rows),
            "known_label_member_rows": sum(row["label_known_members"] for row in rows),
            "quantile_status_counts": dict(
                Counter(value["status"] for row in rows for value in row["quantiles"].values())
            ),
            "persistence_status_counts": {
                str(lag): dict(Counter(row["persistence"][str(lag)]["status"] for row in rows))
                for lag in (HOUR, 24 * HOUR)
            },
        }
    return {"score": result["score"], "assets": output}


def _stable(identities):
    """只强锁实际依赖和实际输入，后台无关代码不属于执行身份。"""
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("诊断期间实际输入或执行来源改变")


def run(panel, source, output):
    """先冻结来源与十二方向，再物理投影白名单并逐分数输出。"""
    panel, source, output = (Path(path).resolve() for path in (panel, source, output))
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (panel, source)):
        raise ValueError("独立诊断不能覆盖或包含既有研究")
    verify_receipts(panel)
    verify_receipts(source)
    if len(SCORES) != 12 or len(RAW_NAMES) != 2 or len(MODEL_NAMES) != 5:
        raise ValueError("诊断必须保留固定十二个分数")
    if (
        read(panel / "result.json").get("status") not in {"COMPLETED", "COMPLETED_WITH_UNKNOWN"}
        or (source / "failure.json").exists()
    ):
        raise ValueError("失败来源不能作为完整诊断输入")
    source_result = read(source / "result.json")
    if (
        source_result.get("schema") != "ta-extended-funding-regression-result.v1"
        or source_result.get("declared_tests") != 24
        or source_result.get("protocol_sha256") != sha(source / "protocol.json")
    ):
        raise ValueError("续检预测没有成功完成的完整来源记录")
    parent, study_protocol = read(panel / "protocol.json"), read(source / "protocol.json")
    if (
        parent.get("schema") != "ta-historical-funding-panel.v1"
        or study_protocol.get("schema") != "ta-extended-funding-study.v1"
        or study_protocol.get("source") != str(panel)
    ):
        raise ValueError("诊断只接受同一小时双表及固定历史续检预测")
    if any(
        protocol.get(key) is not False
        for protocol in (parent, study_protocol)
        for key in ("orders_enabled", "leader_labels_read", "production_eligible")
    ):
        raise ValueError("诊断来源必须为纯公开离线研究")
    for protocol in (parent, study_protocol):
        if (
            protocol.get("start_ms") != study.START
            or protocol.get("end_ms") != END
            or tuple(protocol.get("features", ())) != FEATURES
        ):
            raise ValueError("固定月份或九项共同特征不同")
    if (
        study_protocol.get("universe") != parent.get("universe")
        or study_protocol.get("models") != {key: list(value) for key, value in study.MODELS.items()}
        or study_protocol.get("refit_times_ms") != list(study.REFITS)
        or study_protocol.get("horizon_minutes") != 240
        or study_protocol.get("label_delay_ms") != 241 * study.MINUTE
    ):
        raise ValueError("固定全集、五模型或240分钟标签语义不同")
    plan_path, plan_sha = str(study.PLAN.resolve()), sha(study.PLAN)
    if (
        parent.get("study_plan_sha256") != plan_sha
        or study_protocol.get("source_sha256", {}).get(plan_path) != plan_sha
    ):
        raise ValueError("主计划与双表及预测冻结计划不同")
    panel_ids = {
        str(panel / name): sha(panel / name)
        for name in ("receipts.json", "protocol.json", "features.parquet", "labels.parquet")
    }
    if any(
        study_protocol.get("input_identities", {}).get(name) != identity
        for name, identity in panel_ids.items()
    ):
        raise ValueError("预测未绑定当前同一双表身份")
    inputs = (
        panel_ids
        | {str(panel / "result.json"): sha(panel / "result.json")}
        | {
            str(source / name): sha(source / name)
            for name in ("receipts.json", "protocol.json", "predictions.parquet", "result.json")
        }
    )
    files = [
        Path(__file__),
        Path(study.__file__),
        Path(factor_diagnostics.__file__),
        Path(regression.__file__),
        Path(__file__).with_name("artifacts.py"),
        Path(__file__).with_name("funding_study.py"),
        Path(__file__).with_name("linear_family.py"),
        Path(__file__).with_name("regression_run.py"),
        Path(__file__).parent / "factors/funding.py",
        study.PLAN,
        Path(__file__).parent / "plans/factor_diagnostics_plan.md",
    ]
    frozen = {str(path.resolve()): sha(path) for path in files}
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-extended-funding-diagnostics.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "panel_root": str(panel),
            "study_root": str(source),
            "universe": parent["universe"],
            "scores": SCORES,
            "start_ms": START,
            "end_ms": END,
            "frequency_ms": HOUR,
            "lag_ms": [HOUR, 24 * HOUR],
            "asset_classes": ASSETS,
            "common_features": FEATURES,
            "scope": SCOPE,
            "minimum_members": 20,
            "target_column": "target_240",
            "diagnostics_version": factor_diagnostics.VERSION,
            "source_sha256": frozen,
            "input_identities": inputs,
            "feature_columns_loaded": study.FEATURE_COLUMNS,
            "label_columns_loaded": study.LABEL_COLUMNS,
            "prediction_columns_loaded": PREDICTION_COLUMNS,
            "sequential_score_release": True,
            "new_p_values_emitted": False,
            "bh_applied": False,
            "new_significance_tests_declared": False,
            "existing_24_test_family_unchanged": True,
            "directions_optimized": False,
            "independent_oos": False,
            "costs_or_funding_deducted": False,
            "economic_account_evaluated": False,
            "economic_gate_changed": False,
            "leader_labels_read": False,
            "orders_enabled": False,
            "production_eligible": False,
            "runtime": {
                "python": sys.version,
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow")
                },
            },
        },
    )
    try:
        for index, path in enumerate(files):
            target = output / "source" / f"{index:02}_{path.name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
        features = pd.read_parquet(panel / "features.parquet", columns=list(study.FEATURE_COLUMNS))
        labels = pd.read_parquet(panel / "labels.parquet", columns=list(study.LABEL_COLUMNS))
        predictions = pd.read_parquet(
            source / "predictions.parquet", columns=list(PREDICTION_COLUMNS)
        )
        base, predictions = validate_inputs(features, labels, predictions, parent["universe"])
        del features, labels
        base, identities = scope_identities(base)
        identities.to_parquet(output / "scope_identities.parquet", index=False)
        coverage = {
            "scheduled_scope_rows": len(identities),
            "scheduled_scope_clocks": len(range(START, END, HOUR)),
            "diagnostic_asset_status_counts": identities.diagnostic_asset_status.value_counts().to_dict(),
            "asset_class_member_rows": identities.asset_class.value_counts().to_dict(),
            "common_member_rows": int(identities.common_member_gate.sum()),
        }
        save(output / "coverage.json", coverage)
        del identities
        predictions = predictions[
            (predictions.clock_ms >= START)
            & (predictions.clock_ms < END)
            & predictions.asset_class.isin(ASSETS)
        ].copy()
        summaries = []
        (output / "scores").mkdir()
        for score in SCORES:
            result = evaluate_score(base, predictions, score)
            save(output / "scores" / (score["id"] + ".json"), result)
            summaries.append(summarize(result))
            del result
            gc.collect()
            print({"completed_score": score["id"], "scheduled_scores": len(SCORES)}, flush=True)
        _stable(inputs | frozen)
        save(output / "summary.json", summaries)
        result = {
            "status": "COMPLETED_WITH_UNKNOWN",
            "scores": len(SCORES),
            "asset_classes": ASSETS,
            "scheduled_scope_clocks": len(range(START, END, HOUR)),
            "known_asset_sections_per_score": len(ASSETS) * len(range(START, END, HOUR)),
            "models_called": False,
            "new_p_values_emitted": False,
            "economic_account_evaluated": False,
            "orders_enabled": False,
        }
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "reason": str(error)[:800],
                "numeric_conclusions_allowed": False,
            },
        )
        result = {
            "status": "FAILED_INTEGRITY",
            "numeric_conclusions_allowed": False,
            "models_called": False,
            "orders_enabled": False,
        }
        save(output / "result.json", result)
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
    """只接受显式封存根和新输出，不调用网络或模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args.panel, args.source, args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] != "FAILED_INTEGRITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
