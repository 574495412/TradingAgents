"""冻结控制项基准和逐因子剔除模型，只衡量预测增量，不产生订单。"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.regression import (
    CONTROLS,
    FACTORS,
    FEATURES,
    bh_adjust,
    correlation,
    daily_stat,
    diagnostic_spread,
    ranked,
)

MODELS = {
    "controls_only": CONTROLS,
    "full": FEATURES,
    **{f"without_{f}": tuple(n for n in FEATURES if n != f) for f in FACTORS},
}
CONTRASTS = ("controls_only",) + tuple(f"without_{factor}" for factor in FACTORS)


def fit_all(frame, target):
    """全部模型共用原七特征人口、标签行和标准化，不能靠删因子扩大覆盖。"""
    if frame["symbol"].duplicated().any():
        raise ValueError("增量回归出现重复合约")
    known = frame[np.isfinite(frame[list(FEATURES)]).all(axis=1)].sort_values("symbol")
    transformed = ranked(known)
    valid = np.isfinite(known[target]).to_numpy()
    fits, failures = {}, {}
    for name, columns in MODELS.items():
        if transformed is None or valid.sum() < 20:
            failures[name] = "INSUFFICIENT_COMMON_MEMBERS"
            continue
        matrix = np.column_stack(
            [np.ones(int(valid.sum())), transformed.loc[valid, list(columns)].to_numpy()]
        )
        if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
            failures[name] = "RANK_DEFICIENT_FIXED_DESIGN"
            continue
        beta = np.linalg.lstsq(matrix, known.loc[valid, target].to_numpy(), rcond=None)[0]
        fits[name] = {column: float(value) for column, value in zip(columns, beta[1:], strict=True)}
    return known, transformed, fits, failures


def predictions(known, transformed, target, coefficients):
    """在当前特征上应用固定训练斜率，预测和选组都先于标签筛选。"""
    if transformed is None or coefficients is None:
        return None
    scores = sum(transformed[column].to_numpy() * value for column, value in coefficients.items())
    valid = np.isfinite(known[target]).to_numpy()
    if valid.sum() < 20:
        return None
    observed = known.loc[valid, target].to_numpy()
    observed = observed - observed.mean()
    errors = (scores[valid] - observed) ** 2
    return {
        "members": int(valid.sum()),
        "sse": float(errors.sum()),
        "mse": float(errors.mean()),
        "null_sse": float(observed @ observed),
        "rank_ic": correlation(scores[valid], observed, rank=True),
        "spread": diagnostic_spread(known, scores, target, 1),
    }


def evaluate(samples, protocol, parent_result=None):
    """先从训练截面估计七模型，再固定全部后段预测，不能用验证标签改系数。"""
    records, contexts = [], {}
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        for horizon in (20, 60):
            phase = frame[f"phase_{horizon}"].iloc[0]
            if phase == "excluded_boundary":
                continue
            known, transformed, fits, failures = fit_all(frame, f"target_{horizon}")
            row = {
                "asset_class": asset,
                "clock_ms": int(clock),
                "horizon_minutes": horizon,
                "phase": phase,
                "day": pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                "known_feature_members": len(known),
                "hourly_fits": fits,
                "fit_failures": failures,
                "predictions": {},
                "contrasts": {},
            }
            records.append(row)
            contexts[asset, int(clock), horizon] = known, transformed
    groups, tests = {}, []
    for asset in sorted(samples["asset_class"].unique()):
        for horizon in (20, 60):
            rows = [
                r for r in records if r["asset_class"] == asset and r["horizon_minutes"] == horizon
            ]
            train = [
                r for r in rows if r["phase"] == "training" and len(r["hourly_fits"]) == len(MODELS)
            ]
            coefficients = {
                name: {
                    column: float(np.mean([r["hourly_fits"][name][column] for r in train]))
                    for column in columns
                }
                if train
                else None
                for name, columns in MODELS.items()
            }
            key = f"{asset}:{horizon}m"
            if parent_result is not None:
                expected = parent_result["groups"][key]["training_mean_coefficients"]
                if (
                    coefficients["full"] is None
                    or expected is None
                    or any(
                        not np.isclose(value, expected[f], rtol=1e-10, atol=1e-12)
                        for f, value in coefficients["full"].items()
                    )
                ):
                    raise ValueError("完整模型训练系数不能精确复现首轮回归")
            for row in rows:
                known, transformed = contexts[asset, row["clock_ms"], horizon]
                row["predictions"] = {
                    name: predictions(known, transformed, f"target_{horizon}", beta)
                    for name, beta in coefficients.items()
                }
                full = row["predictions"]["full"]
                for name in CONTRASTS:
                    baseline = row["predictions"][name]
                    ready = (
                        full is not None
                        and baseline is not None
                        and full["members"] == baseline["members"]
                    )
                    row["contrasts"][name] = {
                        "mse_improvement": baseline["mse"] - full["mse"] if ready else None,
                        "sse_improvement": baseline["sse"] - full["sse"] if ready else None,
                        "rank_ic_improvement": full["rank_ic"] - baseline["rank_ic"]
                        if ready and full["rank_ic"] is not None and baseline["rank_ic"] is not None
                        else None,
                    }
            phases = {}
            for phase, bounds in protocol["phases"].items():
                part = [r for r in rows if r["phase"] == phase]
                days = list(
                    pd.date_range(
                        bounds["start"], bounds["end"], inclusive="left", freq="D"
                    ).strftime("%Y-%m-%d")
                )
                model_scores = {}
                for name in MODELS:
                    valid = [
                        r["predictions"][name] for r in part if r["predictions"][name] is not None
                    ]
                    denominator = sum(v["null_sse"] for v in valid)
                    model_scores[name] = {
                        "prediction_cross_sections": len(valid),
                        "prediction_members": sum(v["members"] for v in valid),
                        "r_squared_vs_zero": 1 - sum(v["sse"] for v in valid) / denominator
                        if denominator > 0
                        else None,
                        "rank_ic": daily_stat(
                            part,
                            lambda r, n=name: (
                                r["predictions"][n]["rank_ic"] if r["predictions"][n] else None
                            ),
                            days,
                        ),
                        "diagnostic_spread": {
                            m: daily_stat(
                                part,
                                lambda r, n=name, metric=m: (
                                    r["predictions"][n]["spread"][metric]
                                    if r["predictions"][n]
                                    else None
                                ),
                                days,
                            )
                            for m in ("gross", "base_cost_sensitivity", "stress_cost_sensitivity")
                        },
                    }
                if parent_result is not None:
                    expected_r2 = parent_result["groups"][key]["phases"][phase][
                        "frozen_training_model"
                    ]["excess_return_r_squared_vs_zero"]
                    if not np.isclose(
                        model_scores["full"]["r_squared_vs_zero"],
                        expected_r2,
                        rtol=1e-8,
                        atol=1e-10,
                    ):
                        raise ValueError("完整模型后段预测不能复现首轮R²")
                comparisons = {}
                for name in CONTRASTS:
                    paired = [
                        r for r in part if r["contrasts"][name]["mse_improvement"] is not None
                    ]
                    null = sum(r["predictions"]["full"]["null_sse"] for r in paired)
                    statistics = daily_stat(
                        part, lambda r, n=name: r["contrasts"][n]["mse_improvement"], days
                    )
                    comparisons[name] = {
                        "paired_cross_sections": len(paired),
                        "mse_improvement": statistics,
                        "r_squared_improvement_same_rows": sum(
                            r["contrasts"][name]["sse_improvement"] for r in paired
                        )
                        / null
                        if null > 0
                        else None,
                        "rank_ic_improvement": daily_stat(
                            part, lambda r, n=name: r["contrasts"][n]["rank_ic_improvement"], days
                        ),
                    }
                    tests.append((key, phase, name, statistics["p_asymptotic_two_sided"]))
                phases[phase] = {
                    "scheduled_cross_sections": len(part),
                    "model_scores": model_scores,
                    "comparisons": comparisons,
                    "fit_failures": dict(
                        Counter(
                            name + ":" + reason
                            for r in part
                            for name, reason in r["fit_failures"].items()
                        )
                    ),
                }
            groups[key] = {
                "training_common_fit_hours": len(train),
                "frozen_training_coefficients": coefficients,
                "phases": phases,
            }
    for (key, phase, name, _), q in zip(tests, bh_adjust([t[-1] for t in tests]), strict=True):
        groups[key]["phases"][phase]["comparisons"][name]["mse_improvement"][
            "bh_q_all_declared_tests"
        ] = q
    return {
        "schema": "cz-factor-incremental-result.v1",
        "groups": groups,
        "hourly_records": records,
        "declared_incremental_tests": len(tests),
        "valid_p_value_tests": sum(v[-1] is not None for v in tests),
        "all_history_previously_viewed": True,
        "parent_full_model_reproduced": parent_result is not None,
        "economic_account_evaluated": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }


def run(source, output):
    """增量设计、输入和源码在计算之前冻结，保留所有七模型而非只保存赢家。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    verify_receipts(source)
    parent_protocol, parent_result = read(source / "protocol.json"), read(source / "result.json")
    plan = source_path("profit_factor_incremental_plan.md")
    paths = {
        Path(__file__).resolve(),
        plan,
        Path(ranked.__code__.co_filename).resolve(),
        Path(read.__code__.co_filename).resolve(),
        Path(verify_receipts.__code__.co_filename).resolve(),
    }
    identities = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "cz-factor-incremental-protocol.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "phases": parent_protocol["phases"],
        "parent_receipts_sha256": sha(source / "receipts.json"),
        "models": MODELS,
        "contrasts": CONTRASTS,
        "source_sha256": identities,
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("pandas", "numpy", "pyarrow")},
        },
        "all_data_previously_used": True,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    for path in paths:
        target = output / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    result = evaluate(pd.read_parquet(source / "hourly_samples.parquet"), protocol, parent_result)
    save(output / "hourly_results.json", result.pop("hourly_records"))
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("增量回归期间源码或冻结计划发生变化")
    verify_receipts(source)
    result["protocol_sha256"] = sha(output / "protocol.json")
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(
        {
            "groups": len(result["groups"]),
            "declared_incremental_tests": result["declared_incremental_tests"],
            "parent_full_model_reproduced": result["parent_full_model_reproduced"],
            "production_eligible": False,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
