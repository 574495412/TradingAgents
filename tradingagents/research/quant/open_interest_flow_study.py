"""执行原生H03交互增量筛查，经济账户须另有完整成交协议。"""

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

from .artifacts import read, save, sha
from .factors.open_interest_flow import FLOW, GROWTH, NAME, OI, components
from .factors.position_tilt import FEATURES, HOUR, LABEL_DELAY, MINUTE, TARGET
from .position_tilt_study import (
    CUTOFF,
    END,
    INPUTS as PARENT_INPUTS,
    ROOT,
    START,
    _seal,
    _validate_sources,
    evaluate,
)

HYPOTHESIS = "think_alpha_hypothesis_03"
BASELINE = (*FEATURES, FLOW, GROWTH)
COMMON = (*BASELINE, NAME)
INPUTS = {
    **PARENT_INPUTS,
    "native": (
        "native_oi_post_h02_review_20261017_v001",
        "2d82d29c1a9ed2b60899258172ca72b6453324a36918c010db8fe10ad3cc4145",
    ),
    "controls": (
        "position_tilt_20261017_v002",
        "6fe6bdb6c66ea0f6fa8e3026a0f967587d1118062273f4e3fb2a55b065d02db7",
    ),
    "controls_audit": (
        "position_tilt_audit_20261017_v003",
        "1ea03fdc1a71eb4f200a706e4b22816779ed1c339805b56b298678934bf838bf",
    ),
}
MODEL_SPECS = (
    ("controls_common", FEATURES, COMMON),
    ("controls_own", FEATURES, FEATURES),
    ("augmented_own", COMMON, COMMON),
)


def score(samples):
    """B3只与含主效应的B2作主检验，B1保留为补充对照。"""
    return evaluate(
        samples,
        baseline_features=BASELINE,
        factor_name=NAME,
        extra_model_specs=MODEL_SPECS,
        extra_diagnostic_scores=("controls_common",),
        scope="H03_20261005_20261015_previously_viewed_history",
        coefficient_key="h03_standardized_coefficient",
        own_score_name="raw_interaction_own",
        require_positive_spread=False,
        require_concentration_survival=True,
    )


def run(output):
    """来源首末核验，复用已独立重建的控制项，新增交互从纯来源计算。"""
    roots = {k: ROOT / "results/research" / name for k, (name, _) in INPUTS.items()}
    output = Path(output).resolve()
    package = ROOT / "tradingagents"
    if any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (*roots.values(), package)
    ):
        raise ValueError("新输出不能覆盖或包含来源与代码")
    identities, universe = _validate_sources(
        roots, expected_inputs=INPUTS, expected_hypothesis=HYPOTHESIS
    )
    control_protocol = read(roots["controls"] / "protocol.json")
    control_audit = read(roots["controls_audit"] / "result.json")
    audit_inputs = read(roots["controls_audit"] / "protocol.json")["input_receipts"]
    if (
        control_audit["status"] != "VERIFIED_FULL_FEATURES_LABELS_REGRESSION_PRIMARY_TESTS"
        or audit_inputs[str(roots["controls"])] != INPUTS["controls"][1]
        or any(
            control_protocol["input_identities"].get(str(roots[k] / "receipts.json"))
            != INPUTS[k][1]
            for k in ("panel", "prices")
        )
        or control_protocol["label_information_delay_ms"] != LABEL_DELAY
        or control_protocol["start_ms"] != START
        or control_protocol["cutoff_ms"] != CUTOFF
        or control_protocol["end_exclusive_ms"] != END
    ):
        raise ValueError("复用控制项缺少同源全量独立验证")
    paths = sorted(
        set(package.rglob("*.py")) | set((package / "research/quant/plans").glob("*.md"))
    )
    frozen = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {
        "schema": "ta-native-open-interest-flow-study.v1",
        "hypothesis_id": HYPOTHESIS,
        "created_ms": time.time_ns() // 1_000_000,
        "research_owner": "TradingAgents",
        "input_identities": identities,
        "source_sha256": frozen,
        "universe": universe,
        "start_ms": START,
        "cutoff_ms": CUTOFF,
        "end_exclusive_ms": END,
        "baseline_features": list(BASELINE),
        "common_features": list(COMMON),
        "model_meanings": {
            "controls_common": "B1_common",
            "controls_own": "B1_own",
            "baseline_common": "B2_common",
            "baseline_own": "B2_own",
            "augmented_common": "B3_common",
            "augmented_own": "B3_own",
        },
        "formula": "S=max(log(O(rawT=t-10m))-log(O(rawT=t-70m)),0)*F_60m_ending_t-10m",
        "metrics_visibility_assumption_ms": 10 * MINUTE,
        "label_information_delay_ms": LABEL_DELAY,
        "label_completed_clock_mapping": "open(t+62m)/open(t+2m)-1",
        "lambda": 1.0,
        "refits_per_asset": 1,
        "primary_bh_family_size": 2,
        "primary_contrast": "B2_common_minus_B3_common_squared_error",
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "historical_first_publication_proven": False,
        "oi_economic_clock_proven": False,
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
                raise ValueError("源副本与运行前身份不同")
        controls = pq.read_table(
            roots["controls"] / "samples.parquet",
            columns=[
                "symbol",
                "asset_class",
                "clock_ms",
                *FEATURES,
                TARGET,
                "label_path_complete",
                "label_information_end_ms",
            ],
        ).to_pandas()
        if (
            len(controls) != 305184
            or controls.duplicated(["symbol", "clock_ms"]).any()
            or not controls.label_information_end_ms.eq(controls.clock_ms + LABEL_DELAY).all()
        ):
            raise ValueError("控制项必须保留全748×408小时身份")
        control_groups = dict(iter(controls.groupby("symbol", sort=True)))
        if set(control_groups) != set(universe):
            raise ValueError("控制项未保留人口")
        metrics = pq.ParquetFile(roots["panel"] / "panel_hourly.parquet")
        prices = pq.ParquetFile(roots["prices"] / "minutes.parquet")
        groups, symbol_column = {}, prices.schema_arrow.names.index("symbol")
        for rg in range(prices.num_row_groups):
            stats = prices.metadata.row_group(rg).column(symbol_column).statistics
            if stats is None or stats.min != stats.max or stats.min not in universe:
                raise ValueError("价格行组身份不唯一")
            groups.setdefault(stats.min, []).append(rg)
        if set(groups) != set(universe) or metrics.num_row_groups != 748:
            raise ValueError("原来源缺少人口行组")
        frames = []
        for number, symbol in enumerate(sorted(universe), 1):
            hourly = metrics.read_row_group(
                number - 1,
                columns=[
                    "symbol",
                    "clock_ms",
                    "raw_T_ms",
                    "requested_metrics_raw_T_ms",
                    OI,
                    OI + "_value_status",
                ],
            ).to_pandas()
            minute = (
                prices.read_row_groups(
                    groups[symbol],
                    columns=[
                        "clock_ms",
                        "quote_volume",
                        "quote_volume_value_status",
                        "taker_buy_quote_volume",
                        "taker_buy_quote_volume_value_status",
                    ],
                )
                .to_pandas()
                .set_index("clock_ms")
            )
            if (
                set(hourly.symbol) != {symbol}
                or hourly.clock_ms.tolist() != list(range(START, END, HOUR))
                or minute.index.tolist() != list(range(START, END, MINUTE))
            ):
                raise ValueError("两来源必须保持完整精确日历")
            values = components(hourly, minute)
            frame = control_groups[symbol].sort_values("clock_ms").copy()
            if frame.clock_ms.tolist() != hourly.clock_ms.tolist() or set(frame.asset_class) != {
                universe[symbol]["asset_class"]
            }:
                raise ValueError("控制项日历或资产分类改变")
            for name in (FLOW, GROWTH, NAME):
                frame[name] = values[name].to_numpy()
            frames.append(frame)
            if number % 100 == 0 or number == len(universe):
                print({"prepared_symbols": number, "total": len(universe)}, flush=True)
        samples = pd.concat(frames, ignore_index=True)
        pq.write_table(
            pa.Table.from_pandas(samples, preserve_index=False), output / "samples.parquet"
        )
        predictions, fits, summaries, primary, diagnostics = score(samples)
        pq.write_table(
            pa.Table.from_pandas(predictions, preserve_index=False), output / "predictions.parquet"
        )
        save(output / "coefficients.json", fits)
        save(output / "primary_tests.json", primary)
        for asset, scores in diagnostics.items():
            for name, record in scores.items():
                save(output / f"diagnostic_{asset}_{name}.json", record)
        predictions.groupby(["asset_class", "symbol"]).daily_equal_weight_contribution.sum(
            min_count=1
        ).to_csv(output / "symbol_contributions.csv")
        final_identities, _ = _validate_sources(
            roots, expected_inputs=INPUTS, expected_hypothesis=HYPOTHESIS
        )
        if final_identities != identities or any(
            sha(Path(p)) != checksum for p, checksum in frozen.items()
        ):
            raise ValueError("运行期间输入或执行依赖发生变化")
        result = {
            "schema": "ta-native-open-interest-flow-study.v1.result",
            "status": "COMPLETED_PREDICTIVE_SCREEN_ONLY",
            "samples": len(samples),
            "validation_rows": len(predictions),
            "factor_known": int(np.isfinite(samples[NAME]).sum()),
            "factor_known_zero": int(samples[NAME].eq(0).sum()),
            "summary": summaries,
            "primary_tests": primary,
            "predictive_screen_passed_groups": sum(
                r["exploratory_predictive_screen_passed"] for r in primary
            ),
            "elapsed_seconds": time.monotonic() - started,
            "economic_account_evaluated": False,
            "economic_increment_demonstrated": False,
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
    """纯离线回归，不调用模型、公开网络或交易接口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output)
    print({k: result[k] for k in ("status", "samples", "predictive_screen_passed_groups")})


if __name__ == "__main__":
    main()
