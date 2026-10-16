"""不调用评分实现，重算截面名次、IC、OLS与训练冻结预测。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.artifacts import read, save, sha

FACTORS = ("return_3m", "return_20m", "return_60m", "flow_3m", "flow_12m")
FEATURES = FACTORS + ("volatility_60m", "log_quote_volume_60m")


def ranks(x):
    """用唯一值计数直接计算平均名次，与被核验模块的pandas名次实现独立。"""
    _, inverse, counts = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    return (first + (counts + 1) / 2)[inverse]


def corr(x, y):
    """中心化内积计算相关系数，常量保持未知。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float((x @ x) * (y @ y)))
    return float(x @ y) / denominator if denominator > 0 else None


def equal(actual, expected):
    """只容许浮点线性代数的舍入，不容许用未知代零。"""
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError("独立回归核验的未知口径不同")
    elif not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("独立回归核验数值不同")


def run(source, output):
    """覆盖全部小时统计，并核验原始文件、完整数据集和跨阶段边界。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    receipts = read(source / "receipts.json")
    if {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()} - {
        "receipts.json"
    } != set(receipts):
        raise ValueError("回归封存文件集合不同")
    if any(sha(source / filename) != value for filename, value in receipts.items()):
        raise ValueError("回归封存文件字节不同")
    if any(
        sha(filename) != value for filename, value in read(source / "input_identities.json").items()
    ):
        raise ValueError("回归原始分钟来源发生变化")
    protocol, result, hourly = (
        read(source / name) for name in ("protocol.json", "result.json", "hourly_results.json")
    )
    data = pd.read_parquet(source / "hourly_samples.parquet")
    if data.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("评分数据的合约小时重复")
    scheduled = {
        (asset, int(clock), h)
        for (asset, clock), frame in data.groupby(["asset_class", "clock_ms"])
        for h in (20, 60)
        if frame[f"phase_{h}"].iloc[0] != "excluded_boundary"
    }
    observed = {(r["asset_class"], r["clock_ms"], r["horizon_minutes"]) for r in hourly}
    if len(observed) != len(hourly) or observed != scheduled:
        raise ValueError("评分删除或重复了预定截面")
    lookup = {(r["asset_class"], r["clock_ms"], r["horizon_minutes"]): r for r in hourly}
    independent_fits, successes = {}, 0
    for (asset, clock), frame in data.groupby(["asset_class", "clock_ms"], sort=True):
        known = frame[np.isfinite(frame[list(FEATURES)]).all(axis=1)].sort_values("symbol")
        transformed = (
            np.column_stack([(ranks(known[f]) - 1) / (len(known) - 1) - 0.5 for f in FEATURES])
            if len(known) >= 20
            else None
        )
        for horizon in (20, 60):
            key, target = (asset, int(clock), horizon), f"target_{horizon}"
            if key not in lookup:
                continue
            record = lookup[key]
            bounds = protocol["phases"][record["phase"]]
            exit_ms = int(clock) + (horizon + 1) * 60_000
            if (
                not pd.Timestamp(bounds["start"]).value // 1_000_000 <= clock
                or not exit_ms < pd.Timestamp(bounds["end"]).value // 1_000_000
            ):
                raise ValueError("未来标签跨越阶段边界")
            valid = np.isfinite(known[target]).to_numpy()
            if record["known_feature_members"] != len(known) or record[
                "known_target_members"
            ] != int(valid.sum()):
                raise ValueError("因子评分漏报缺失成员")
            if transformed is None or valid.sum() < 20:
                if record["factor_statistics"] or record["multivariate"] is not None:
                    raise ValueError("有效成员不足却生成统计")
                continue
            y, x = known.loc[valid, target].to_numpy(), transformed[valid]
            for index, factor in enumerate(FACTORS):
                stats = record["factor_statistics"][factor]
                equal(stats["pearson_ic"], corr(known.loc[valid, factor], y))
                equal(stats["rank_ic"], corr(ranks(x[:, index]), ranks(y)))
                centered, centered_y = x[:, index] - x[:, index].mean(), y - y.mean()
                denominator = float(centered @ centered)
                equal(
                    stats["coefficient"],
                    float(centered @ centered_y) / denominator if denominator > 0 else None,
                )
            design = np.column_stack([np.ones(len(y)), x])
            if np.linalg.matrix_rank(design) < design.shape[1]:
                if record["multivariate"] is not None:
                    raise ValueError("奇异矩阵被偷偷删除列")
            else:
                beta = np.linalg.pinv(design) @ y
                for name, value in zip(("intercept",) + FEATURES, beta, strict=True):
                    equal(record["multivariate"][name], float(value))
                independent_fits[key] = dict(zip(FEATURES, beta[1:], strict=True))
                successes += 1
    for group, value in result["groups"].items():
        asset, horizon_text = group.split(":")
        horizon = int(horizon_text[:-1])
        fitted = [
            beta
            for (a, clock, h), beta in independent_fits.items()
            if a == asset and h == horizon and lookup[a, clock, h]["phase"] == "training"
        ]
        for factor in FEATURES:
            equal(
                value["training_mean_coefficients"][factor]
                if value["training_mean_coefficients"]
                else None,
                float(np.mean([b[factor] for b in fitted])) if fitted else None,
            )
    output.mkdir(parents=True, exist_ok=False)
    proof = {
        "schema": "cz-factor-regression-independent-audit.v1",
        "source_receipts_sha256": sha(source / "receipts.json"),
        "hourly_cross_sections_verified": len(hourly),
        "multivariate_fits_verified": successes,
        "scheduled_missingness_preserved": True,
        "training_coefficients_rebuilt_independently": True,
        "future_label_phase_boundaries_verified": True,
        "sample_features_rebuilt_from_raw": False,
        "economic_account_evaluated": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
    }
    save(output / "result.json", proof)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
