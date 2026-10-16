"""固定模型族的共同人口回归，训练之后不重新拟合或挑选方向。"""

from collections import Counter

import numpy as np
import pandas as pd

from .regression import bh_adjust, correlation, daily_stat, diagnostic_spread


def fit_section(frame, target, features, models):
    """标准化只看完整已知特征；缺未来标签不能改变成员名次。"""
    if frame["symbol"].duplicated().any():
        raise ValueError("共同截面存在重复合约")
    known = frame[np.isfinite(frame[list(features)]).all(axis=1)].sort_values("symbol")
    transformed = (
        (known[list(features)].rank(method="average") - 1) / (len(known) - 1) - 0.5
        if len(known) >= 20
        else None
    )
    valid = np.isfinite(known[target]).to_numpy()
    fits, failures = {}, {}
    for name, columns in models.items():
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
        fits[name] = dict(zip(columns, map(float, beta[1:]), strict=True))
    return known, transformed, fits, failures


def evaluate(samples, protocol):
    """固定全部模型、比较、期限和阶段，所有检验一起进行探索性BH校正。"""
    features, models = tuple(protocol["features"]), protocol["models"]
    contrasts, horizons = protocol["contrasts"], protocol["horizons_minutes"]
    if len(set(features)) != len(features) or any(
        f.startswith(("target_", "phase_")) for f in features
    ):
        raise ValueError("未来标签、阶段或重复列不能作为模型特征")
    if any(not set(columns) <= set(features) for columns in models.values()):
        raise ValueError("模型使用了未声明特征")
    bounds = {
        name: tuple(int(pd.Timestamp(v).value // 1_000_000) for v in (part["start"], part["end"]))
        for name, part in protocol["phases"].items()
    }
    records, contexts = [], {}
    for (asset, clock), frame in samples.groupby(["asset_class", "clock_ms"], sort=True):
        for horizon in horizons:
            phases = frame[f"phase_{horizon}"]
            if phases.nunique() != 1:
                raise ValueError("同一时点的训练与验证阶段混用")
            phase = phases.iloc[0]
            expected = next(
                (
                    name
                    for name, (start, end) in bounds.items()
                    if start <= clock and clock + (horizon + 1) * 60_000 < end
                ),
                "excluded_boundary",
            )
            if phase != expected:
                raise ValueError("阶段标签与完整预测窗口的固定日历不同")
            if phase == "excluded_boundary":
                continue
            known, transformed, fits, failures = fit_section(
                frame, f"target_{horizon}", features, models
            )
            key = asset, int(clock), horizon
            contexts[key] = known, transformed
            records.append(
                {
                    "asset_class": asset,
                    "clock_ms": int(clock),
                    "horizon_minutes": horizon,
                    "phase": phase,
                    "day": pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat(),
                    "scheduled_members": len(frame),
                    "known_feature_members": len(known),
                    "known_target_members": int(np.isfinite(known[f"target_{horizon}"]).sum()),
                    "fits": fits,
                    "failures": failures,
                }
            )
    groups, tests = {}, []
    for asset in sorted(samples["asset_class"].unique()):
        for horizon in horizons:
            subset = [
                r for r in records if r["asset_class"] == asset and r["horizon_minutes"] == horizon
            ]
            training = [
                r for r in subset if r["phase"] == "training" and len(r["fits"]) == len(models)
            ]
            betas = {
                name: {f: float(np.mean([r["fits"][name][f] for r in training])) for f in columns}
                if training
                else None
                for name, columns in models.items()
            }
            for row in subset:
                known, ranks = contexts[asset, row["clock_ms"], horizon]
                target = f"target_{horizon}"
                valid = np.isfinite(known[target]).to_numpy()
                row["factor_ic"] = {
                    f: correlation(known.loc[valid, f], known.loc[valid, target], rank=True)
                    if valid.sum() >= 20
                    else None
                    for f in protocol["scored_factors"]
                }
                row["predictions"] = {}
                for name, beta in betas.items():
                    value = None
                    if beta is not None and ranks is not None:
                        scores = sum(ranks[f].to_numpy() * v for f, v in beta.items())
                        if valid.sum() >= 20:
                            y = known.loc[valid, target].to_numpy()
                            y = y - y.mean()
                            errors = (scores[valid] - y) ** 2
                            value = {
                                "members": int(valid.sum()),
                                "sse": float(errors.sum()),
                                "mse": float(errors.mean()),
                                "null_sse": float(y @ y),
                                "rank_ic": correlation(scores[valid], y, rank=True),
                                "spread": diagnostic_spread(known, scores, target, 1),
                            }
                    row["predictions"][name] = value
                row["contrasts"] = {}
                for name, pair in contrasts.items():
                    baseline, candidate = (row["predictions"][n] for n in pair)
                    ready = baseline is not None and candidate is not None
                    row["contrasts"][name] = {
                        "mse_improvement": baseline["mse"] - candidate["mse"] if ready else None,
                        "sse_improvement": baseline["sse"] - candidate["sse"] if ready else None,
                    }
            phase_results = {}
            for phase, bounds in protocol["phases"].items():
                rows = [r for r in subset if r["phase"] == phase]
                days = list(
                    pd.date_range(
                        bounds["start"], bounds["end"], inclusive="left", freq="D"
                    ).strftime("%Y-%m-%d")
                )
                model_results = {}
                for name in models:
                    values = [
                        r["predictions"][name] for r in rows if r["predictions"][name] is not None
                    ]
                    null = sum(v["null_sse"] for v in values)
                    model_results[name] = {
                        "cross_sections": len(values),
                        "r_squared_vs_zero": 1 - sum(v["sse"] for v in values) / null
                        if null > 0
                        else None,
                        "rank_ic": daily_stat(
                            rows,
                            lambda r, n=name: (
                                r["predictions"][n]["rank_ic"] if r["predictions"][n] else None
                            ),
                            days,
                        ),
                        "spread": {
                            m: daily_stat(
                                rows,
                                lambda r, n=name, m=m: (
                                    r["predictions"][n]["spread"][m]
                                    if r["predictions"][n]
                                    else None
                                ),
                                days,
                            )
                            for m in ("gross", "base_cost_sensitivity", "stress_cost_sensitivity")
                        },
                    }
                comparisons = {}
                for name in contrasts:
                    comparisons[name] = daily_stat(
                        rows, lambda r, n=name: r["contrasts"][n]["mse_improvement"], days
                    )
                    tests.append(comparisons[name])
                factor_results = {
                    f: daily_stat(rows, lambda r, f=f: r["factor_ic"][f], days)
                    for f in protocol["scored_factors"]
                }
                tests.extend(factor_results.values())
                phase_results[phase] = {
                    "scheduled_cross_sections": len(rows),
                    "models": model_results,
                    "comparisons": comparisons,
                    "factor_rank_ic": factor_results,
                    "fit_failures": dict(
                        Counter(n + ":" + v for r in rows for n, v in r["failures"].items())
                    ),
                }
            groups[f"{asset}:{horizon}m"] = {
                "training_common_fit_hours": len(training),
                "frozen_training_coefficients": betas,
                "phases": phase_results,
            }
    p_values = [r["p_asymptotic_two_sided"] for r in tests]
    count_unknown = protocol.get("bh_count_unknown_in_declared_family", False)
    # 新设计把未知检验计入声明家族，计算时取保守占位1，输出仍保留未知。
    adjusted = bh_adjust([1.0 if p is None else p for p in p_values] if count_unknown else p_values)
    for result, p, q in zip(tests, p_values, adjusted, strict=True):
        result["bh_q_all_declared_tests"] = q if p is not None else None
    return {
        "schema": "ta-fixed-linear-family-result.v1",
        "groups": groups,
        "hourly_records": records,
        "declared_tests": len(tests),
        "valid_p_value_tests": sum(r["p_asymptotic_two_sided"] is not None for r in tests),
        "bh_adjustment_family_size": len(tests)
        if count_unknown
        else sum(p is not None for p in p_values),
        "independent_oos": False,
        "economic_account_evaluated": False,
        "orders_enabled": False,
        "stable_profitability_proven": False,
        "production_eligible": False,
    }
