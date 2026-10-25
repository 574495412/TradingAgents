"""执行原生选中的H02固定回归，不把分位标签当可成交账户利润。"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import read, save, sha, verify_receipts
from .factor_diagnostics import evaluate as diagnose
from .factors.position_tilt import (
    A_COLUMN,
    FEATURES,
    HOUR,
    LABEL_DELAY,
    MINUTE,
    NAME,
    P_COLUMN,
    TARGET,
    minute_features,
    tilt,
)
from .regression import bh_adjust, hac_mean
from .ridge import fit_path, predict

ROOT = Path(__file__).resolve().parents[3]
START = 1_790_553_600_000
CUTOFF = 1_791_158_400_000
END = 1_792_022_400_000
DAY = 86_400_000
HYPOTHESIS = "think_alpha_hypothesis_02"
ASSETS = ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL")
COMMON = (*FEATURES, NAME)
INPUTS = {
    "panel": (
        "basis_oi_panel_20261017_v001",
        "3dfb87e126e5dc8d77cc116ea0dc72a2f0db667535ae7eb395db0ac54c8111eb",
    ),
    "prices": (
        "basis_oi_prices_20261017_v001",
        "4b9468e977944ce6d2ec59a78b87defb56dc145edef5b0f61b5bc47a28f2a921",
    ),
    "native": (
        "native_oi_review_20261017_v001",
        "3c6cf5717e3ba725b202931477afa9383f1d0c3701764974fefaa7e69c82734a",
    ),
    "audit": (
        "basis_oi_panel_audit_20261017_v001",
        "b11ecbee9350d808ff3f94c36d3e6bb83653a148a28cd548c8a1cea54162913c",
    ),
}


def models(frame, *, baseline_features=FEATURES, factor_name=NAME, extra_model_specs=()):
    """固定λ=1配对拟合；新增家族显式声明主效应，不改H02默认设计。"""
    baseline_features = tuple(baseline_features)
    common = (*baseline_features, factor_name)
    specs = (
        ("baseline_common", baseline_features, common),
        ("augmented_common", common, common),
        ("baseline_own", baseline_features, baseline_features),
        *extra_model_specs,
    )
    if len({spec[0] for spec in specs}) != len(specs):
        raise ValueError("模型名称重复，不能覆盖主对照")
    return {
        name: fit_path(
            frame,
            features=features,
            common_features=common,
            target=TARGET,
            refit_ms=CUTOFF,
            training_start_ms=START,
            label_delay_ms=LABEL_DELAY,
            penalties=(1.0,),
        )
        for name, features, common in specs
    }


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None


def evaluate(
    samples,
    *,
    baseline_features=FEATURES,
    factor_name=NAME,
    extra_model_specs=(),
    extra_diagnostic_scores=(),
    scope="H02_20261005_20261015_previously_viewed_history",
    coefficient_key="h02_standardized_coefficient",
    own_score_name="raw_tilt_own",
    require_positive_spread=True,
    require_concentration_survival=False,
):
    """共享已登记的十日配对评分；不同假说必须显式传入特征和失效条件。"""
    baseline_features = tuple(baseline_features)
    common = (*baseline_features, factor_name)
    evaluated, fits, summaries, primary, diagnostics = [], {}, {}, [], {}
    calendar = list(range(CUTOFF, END, HOUR))
    for asset in ASSETS:
        full = samples[samples.asset_class.eq(asset)].copy()
        fits[asset] = models(
            full,
            baseline_features=baseline_features,
            factor_name=factor_name,
            extra_model_specs=extra_model_specs,
        )
        part = full[full.clock_ms.ge(CUTOFF)].copy()
        for name, record in fits[asset].items():
            part[name] = predict(part, record, 1.0)
        paired = (
            np.isfinite(part[list(common)].to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(part[TARGET])
            & np.isfinite(part.baseline_common)
            & np.isfinite(part.augmented_common)
        )
        part["paired_label_known"] = paired
        part["paired_mse_improvement"] = np.where(
            paired,
            (part[TARGET] - part.baseline_common) ** 2
            - (part[TARGET] - part.augmented_common) ** 2,
            np.nan,
        )
        part["day_ms"] = part.clock_ms // DAY * DAY
        daily = []
        for day in range(CUTOFF, END, DAY):
            group = part[part.day_ms.eq(day)]
            daily.append(
                {
                    "day_ms": day,
                    "scheduled": len(group),
                    "paired": int(group.paired_label_known.sum()),
                    "mse_improvement": _mean(group.paired_mse_improvement),
                }
            )
        statistic = hac_mean(
            [np.nan if r["mse_improvement"] is None else r["mse_improvement"] for r in daily], lag=3
        )
        primary.append({"asset_class": asset, "statistic": statistic, "daily": daily})
        diagnostics[asset] = {}
        for score, population in (
            (factor_name, common),
            ("baseline_common", common),
            ("augmented_common", common),
            (own_score_name, (factor_name,)),
            *((score, common) for score in extra_diagnostic_scores),
        ):
            frame = part.copy()
            if score == own_score_name:
                frame[score] = frame[factor_name]
            diagnostics[asset][score] = diagnose(
                frame,
                score_column=score,
                target_column=TARGET,
                calendar_ms=calendar,
                scope=scope,
                common_features=population,
                asset_classes=(asset,),
            )
        # 同日先在合约内求和，再除以原日配对成员数，贡献和等于主日均误差改善。
        totals = part.groupby("day_ms").paired_mse_improvement.count().replace(0, np.nan)
        known_days = sum(row["mse_improvement"] is not None for row in daily)
        part["daily_equal_weight_contribution"] = (
            part.paired_mse_improvement / part.day_ms.map(totals) / max(known_days, 1)
        )
        contributions = part.groupby("symbol").daily_equal_weight_contribution.sum(min_count=1)
        positive = contributions[contributions > 0].sort_values(ascending=False, kind="stable")
        largest = positive.index[0] if len(positive) else None
        without = []
        for day in range(CUTOFF, END, DAY):
            group = part[part.day_ms.eq(day) & part.symbol.ne(largest)]
            without.append(_mean(group.paired_mse_improvement))
        raw = diagnostics[asset][factor_name]["summary"][asset]
        coefficient = fits[asset]["augmented_common"]["fits"][0]["coefficients"]
        summaries[asset] = {
            "scheduled_rows": len(part),
            "baseline_feature_rows": int(
                np.isfinite(part[list(baseline_features)]).all(axis=1).sum()
            ),
            "common_feature_rows": int(np.isfinite(part[list(common)]).all(axis=1).sum()),
            "paired_label_rows": int(paired.sum()),
            "known_primary_days": known_days,
            "baseline_own_mse": _mean((part[TARGET] - part.baseline_own) ** 2),
            "baseline_common_mse": _mean(
                (part.loc[paired, TARGET] - part.loc[paired, "baseline_common"]) ** 2
            ),
            "augmented_common_mse": _mean(
                (part.loc[paired, TARGET] - part.loc[paired, "augmented_common"]) ** 2
            ),
            "cash_zero_mse_common": _mean(part.loc[paired, TARGET] ** 2),
            coefficient_key: coefficient.get(factor_name) if coefficient else None,
            "raw_mean_rank_ic": raw["information"]["rank_ic"]["daily"]["hac_mean"]["mean"],
            "raw_q5_minus_q1_mean": raw["spreads"]["raw_mean_return"]["q5_minus_q1"]["mean"],
            "largest_positive_contributor": largest,
            "largest_positive_contribution": float(positive.iloc[0]) if len(positive) else None,
            "without_largest_positive_contributor_daily_mse_improvement": _mean(without),
            "unknown_or_incomplete_paths_with_known_label": int(
                (np.isfinite(part[TARGET]) & ~part.label_path_complete).sum()
            ),
        }
        if extra_model_specs:
            summaries[asset]["all_model_errors"] = {
                name: {
                    "feature_population_rows": int(
                        np.isfinite(part[record["common_features"]]).all(axis=1).sum()
                    ),
                    "prediction_rows": int(np.isfinite(part[name]).sum()),
                    "label_pair_rows": int(
                        (np.isfinite(part[name]) & np.isfinite(part[TARGET])).sum()
                    ),
                    "mse": _mean((part[TARGET] - part[name]) ** 2),
                }
                for name, record in fits[asset].items()
            }
        evaluated.append(part)
    p_values = [r["statistic"]["p_asymptotic_two_sided"] for r in primary]
    q_values = bh_adjust([p if p is not None else 1.0 for p in p_values])
    for row, q in zip(primary, q_values, strict=True):
        row["bh_family_size"] = 2
        row["unknown_p_counted_as_one"] = row["statistic"]["p_asymptotic_two_sided"] is None
        row["q"] = q
        summary = summaries[row["asset_class"]]
        checks = {
            "positive_increment": row["statistic"]["mean"],
            "positive_coefficient": summary[coefficient_key],
            "positive_rank_ic": summary["raw_mean_rank_ic"],
        }
        if require_positive_spread:
            checks["positive_quintile_spread"] = summary["raw_q5_minus_q1_mean"]
        if require_concentration_survival:
            checks["positive_without_largest_contributor"] = summary[
                "without_largest_positive_contributor_daily_mse_improvement"
            ]
        row["mechanism_checks"] = {k: None if v is None else bool(v > 0) for k, v in checks.items()}
        row["exploratory_predictive_screen_passed"] = bool(
            all(v is True for v in row["mechanism_checks"].values()) and q <= 0.05
        )
    return pd.concat(evaluated, ignore_index=True), fits, summaries, primary, diagnostics


def _validate_sources(roots, *, expected_inputs=INPUTS, expected_hypothesis=HYPOTHESIS):
    """验证封存字节、原生选择与人口，不能用正在写入的面板。"""
    identities = {}
    for name, root in roots.items():
        verify_receipts(root)
        checksum = sha(root / "receipts.json")
        expected = expected_inputs[name][1]
        if expected is not None and checksum != expected:
            raise ValueError("来源不等于预先绑定的封存版本：" + name)
        identities[str(root / "receipts.json")] = checksum
    panel, prices = (read(roots[k] / "protocol.json") for k in ("panel", "prices"))
    native, audit = (read(roots[k] / "result.json") for k in ("native", "audit"))
    if (
        native["selected_hypothesis_ids"] != [expected_hypothesis]
        or native["orders_enabled"] is not False
        or audit["status"] != "VERIFIED_FULL_GRID"
        or audit["panel_receipts_sha256"] != expected_inputs["panel"][1]
        or panel["start_ms"] != START
        or panel["end_exclusive_ms"] != END
        or prices["start_ms"] != START
        or prices["end_ms"] != END
        or panel["metrics_delay_ms"] != 10 * MINUTE
        or set(panel["symbols"]) != set(prices["universe"])
        or len(prices["universe"]) != 748
        or {v["asset_class"] for v in prices["universe"].values()} != set(ASSETS)
    ):
        raise ValueError("来源时钟、人口或原生假说选择不兼容")
    return identities, prices["universe"]


def _seal(output):
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )


def run(output):
    """冻结代码和计划后逐品种计算，失败版本保留且不覆盖。"""
    output = Path(output).resolve()
    roots = {k: ROOT / "results/research" / v[0] for k, v in INPUTS.items()}
    package = ROOT / "tradingagents"
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (*roots.values(), package)
    ):
        raise ValueError("输出不可覆盖或包含来源及代码")
    identities, universe = _validate_sources(roots)
    paths = sorted(
        set(package.rglob("*.py")) | set((package / "research/quant/plans").glob("*.md"))
    )
    frozen = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {
        "schema": "ta-native-position-tilt-study.v1",
        "hypothesis_id": HYPOTHESIS,
        "created_ms": time.time_ns() // 1_000_000,
        "research_owner": "TradingAgents",
        "start_ms": START,
        "cutoff_ms": CUTOFF,
        "end_exclusive_ms": END,
        "label_information_delay_ms": LABEL_DELAY,
        "label_completed_clock_mapping": "open(t+62m)/open(t+2m)-1",
        "features": list(FEATURES),
        "common_features": list(COMMON),
        "lambda": 1.0,
        "refit_count_per_asset": 1,
        "primary_bh_family_size": 2,
        "hac_lag_days": 3,
        "input_identities": identities,
        "source_sha256": frozen,
        "universe": universe,
        "historical_archive_rest_mapping_proven": False,
        "historical_first_publication_proven": False,
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "economic_account_evaluated": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
        "runtime": {
            "python": sys.version,
            **{n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")},
        },
    }
    save(output / "protocol.json", protocol)
    try:
        for p in paths:
            target = output / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            if sha(target) != frozen[str(p)]:
                raise ValueError("源码副本与执行前身份不同")
        metrics = pq.ParquetFile(roots["panel"] / "panel_hourly.parquet")
        prices = pq.ParquetFile(roots["prices"] / "minutes.parquet")
        groups = {}
        symbol_column = prices.schema_arrow.names.index("symbol")
        for index in range(prices.num_row_groups):
            stats = prices.metadata.row_group(index).column(symbol_column).statistics
            if stats is None or stats.min != stats.max or stats.min not in universe:
                raise ValueError("价格行组必须有唯一已登记合约")
            groups.setdefault(stats.min, []).append(index)
        if set(groups) != set(universe) or metrics.num_row_groups != 748:
            raise ValueError("面板行组没有完整保留人口")
        samples = []
        for i, symbol in enumerate(sorted(universe), 1):
            hourly = metrics.read_row_group(
                i - 1,
                columns=[
                    "symbol",
                    "clock_ms",
                    "raw_T_ms",
                    "requested_metrics_raw_T_ms",
                    P_COLUMN,
                    A_COLUMN,
                    P_COLUMN + "_value_status",
                    A_COLUMN + "_value_status",
                ],
            ).to_pandas()
            if (
                set(hourly.symbol) != {symbol}
                or hourly.clock_ms.tolist() != list(range(START, END, HOUR))
                or not hourly.requested_metrics_raw_T_ms.eq(hourly.clock_ms - 10 * MINUTE).all()
                or not hourly.loc[hourly.raw_T_ms.notna(), "raw_T_ms"]
                .eq(hourly.loc[hourly.raw_T_ms.notna(), "requested_metrics_raw_T_ms"])
                .all()
            ):
                raise ValueError("小时面板不满足原生固定时钟")
            hourly[NAME] = tilt(hourly)
            columns = ["symbol", "clock_ms", "bar_open_ms"]
            for name in (
                "contract_1m_open",
                "contract_1m_close",
                "quote_volume",
                "taker_buy_quote_volume",
            ):
                columns += [name, name + "_value_status"]
            minutes = prices.read_row_groups(groups[symbol], columns=columns).to_pandas()
            if (
                set(minutes.symbol) != {symbol}
                or minutes.clock_ms.tolist() != list(range(START, END, MINUTE))
                or not minutes.bar_open_ms.eq(minutes.clock_ms - MINUTE).all()
            ):
                raise ValueError("分钟面板必须保留完整完成钟及原开盘钟")
            features = minute_features(minutes.set_index("clock_ms"))
            frame = hourly[["symbol", "clock_ms", NAME]].merge(
                features,
                left_on="clock_ms",
                right_index=True,
                validate="one_to_one",
                how="left",
            )
            frame["asset_class"] = universe[symbol]["asset_class"]
            frame["label_information_end_ms"] = frame.clock_ms + LABEL_DELAY
            samples.append(frame)
            if i % 50 == 0 or i == len(universe):
                print({"prepared_symbols": i, "total": len(universe)}, flush=True)
        sample = pd.concat(samples, ignore_index=True)
        if len(sample) != 305184 or sample.duplicated(["symbol", "clock_ms"]).any():
            raise ValueError("最终样本身份与全人口不守恒")
        pq.write_table(
            pa.Table.from_pandas(sample, preserve_index=False), output / "samples.parquet"
        )
        predictions, fits, summaries, primary, diagnostics = evaluate(sample)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), output / "predictions.parquet"
        )
        save(output / "coefficients.json", fits)
        save(output / "primary_tests.json", primary)
        for asset, scores in diagnostics.items():
            for score, result in scores.items():
                save(output / f"diagnostic_{asset}_{score}.json", result)
        predictions.groupby(["asset_class", "symbol"]).daily_equal_weight_contribution.sum(
            min_count=1
        ).to_csv(output / "symbol_contributions.csv")
        final_identities, _ = _validate_sources(roots)
        if final_identities != identities:
            raise ValueError("执行期间输入收据身份发生变化")
        if any(sha(Path(p)) != expected for p, expected in frozen.items()):
            raise ValueError("执行期间源码或计划发生变化")
        result = {
            "schema": "ta-native-position-tilt-study.v1.result",
            "status": "COMPLETED_DESCRIPTIVE_ONLY",
            "samples": len(sample),
            "validation_rows": len(predictions),
            "factor_known": int(np.isfinite(sample[NAME]).sum()),
            "summary": summaries,
            "primary_tests": primary,
            "predictive_screen_passed_groups": sum(
                r["exploratory_predictive_screen_passed"] for r in primary
            ),
            "elapsed_seconds": time.monotonic() - started,
            "economic_account_evaluated": False,
            "economic_increment_demonstrated": False,
            "historical_archive_rest_mapping_proven": False,
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        }
        save(output / "result.json", result)
    except Exception as error:
        save(
            output / "failure.json",
            {"status": "FAILED", "type": type(error).__name__, "message": str(error)},
        )
        _seal(output)
        raise
    _seal(output)
    return result


def main(argv=None):
    """此命令不调用模型、网络或交易接口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output)
    print({k: result[k] for k in ("status", "samples", "predictive_screen_passed_groups")})


if __name__ == "__main__":
    main()
