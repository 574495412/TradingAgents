"""冻结三个价格响应因子，并从完整公开分钟构建共同人口线性模型研究。"""

import argparse
import importlib.metadata
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read, save, sha, verify_receipts
from .factors.response import NAMES, calculate
from .linear_family import evaluate
from .regression import CONTROLS, FACTORS, FEATURES
from .regression_run import MINUTE, PHASES, features_from_minutes, ms, validated_rows

MODELS = {
    "controls": CONTROLS,
    "legacy": FEATURES,
    **{f"add_{f}": FEATURES + (f,) for f in NAMES},
    "expanded": FEATURES + NAMES,
    **{f"drop_{f}": FEATURES + tuple(n for n in NAMES if n != f) for f in NAMES},
}
CONTRASTS = {
    **{f"add_{f}_vs_legacy": ("legacy", f"add_{f}") for f in NAMES},
    "expanded_vs_legacy": ("legacy", "expanded"),
    **{f"expanded_vs_drop_{f}": (f"drop_{f}", "expanded") for f in NAMES},
}
HORIZONS = (20, 60, 240)


def phase_at(clock, horizon):
    """延迟后的目标区间必须严格位于同一固定阶段。"""
    for name, bounds in PHASES.items():
        if ms(bounds["start"]) <= clock and clock + (horizon + 1) * MINUTE < ms(bounds["end"]):
            return name
    return "excluded_boundary"


def run(source, output):
    """先保存设计、输入身份和源码，再计算任何本轮新增未来收益或统计。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("新研究不能覆盖或包含原来源")
    verify_receipts(source)
    parent, raw_ids = read(source / "protocol.json"), read(source / "input_identities.json")
    if parent["phases"] != PHASES or any(
        parent[k] is not False for k in ("orders_enabled", "leader_labels_read")
    ):
        raise ValueError("本轮只沿用原固定的纯公开人口与研究阶段")
    contracts = parent["universe"]
    if len(contracts) != 748:
        raise ValueError("不能缩减原748合约全集")
    paths = [
        Path(__file__).resolve(),
        Path(calculate.__code__.co_filename).resolve(),
        Path(evaluate.__code__.co_filename).resolve(),
        Path(features_from_minutes.__code__.co_filename).resolve(),
        Path(sha.__code__.co_filename).resolve(),
        Path(__file__).with_name("regression.py"),
        Path(__file__).parent / "plans/market_response_plan.md",
    ]
    frozen = {str(p): sha(p) for p in paths}
    bindings = {
        str(source / name): sha(source / name)
        for name in ("protocol.json", "input_identities.json", "receipts.json")
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "ta-price-response-study.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "research_owner": "TradingAgents",
        "phases": PHASES,
        "universe": contracts,
        "features": FEATURES + NAMES,
        "scored_factors": FACTORS + NAMES,
        "models": MODELS,
        "contrasts": CONTRASTS,
        "horizons_minutes": HORIZONS,
        "source_sha256": frozen,
        "parent_identities": bindings,
        "training_common_model_hours_only": True,
        "hourly_sampling": True,
        "four_hour_targets_overlap": True,
        "hac_lag_days": 3,
        "base_turnover_cost_fraction": 0.0016,
        "stress_turnover_cost_fraction": 0.0030,
        "diagnostic_spreads_are_not_account_returns": True,
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "historical_arrival_times_observed": False,
        "historical_eligibility_proven": False,
        "economic_account_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")},
        },
    }
    save(output / "protocol.json", protocol)
    for path in paths:
        target = output / "source" / path.relative_to(Path(__file__).parents[2])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    sources = {}
    for filename, expected in raw_ids.items():
        path = Path(filename).resolve()
        if path.parent.name not in contracts or path.suffix != ".json":
            raise ValueError("原始分钟不属于已声明合约")
        sources.setdefault(path.parent.name, []).append((path, expected))
    if set(sources) != set(contracts):
        raise ValueError("原始分钟身份没有覆盖全部固定成员")
    calendar = np.arange(ms("2026-09-28T00:01:00Z"), ms("2026-10-15T00:00:00Z") + 1, MINUTE)
    hours = np.arange(
        ms(PHASES["training"]["start"]), ms(PHASES["reserved_later_day"]["end"]), 60 * MINUTE
    )
    samples, profiles, verified = [], [], {}
    for number, symbol in enumerate(sorted(contracts), 1):
        rows, seen = [], set()
        for path, expected in sorted(sources[symbol]):
            if sha(path) != expected:
                raise ValueError("封存公开原始分钟发生变化")
            verified[str(path)] = expected
            rows.extend(validated_rows(read(path), symbol, seen))
        frame = (
            pd.DataFrame(
                rows, columns=["clock_ms", "open", "close", "quote_volume", "taker_buy_quote"]
            )
            .set_index("clock_ms")
            .reindex(calendar)
        )
        featured = pd.concat([features_from_minutes(frame), calculate(frame)], axis=1)
        for horizon in HORIZONS:
            featured[f"target_{horizon}"] = (
                frame["open"].shift(-(horizon + 2)) / frame["open"].shift(-2) - 1
            )
        featured = featured.reindex(hours)
        contract = contracts[symbol]
        eligible = (hours > contract["onboard_ms"]) & (
            (contract["delivery_ms"] == 0) | (hours < contract["delivery_ms"])
        )
        featured.loc[~eligible, list(FEATURES + NAMES)] = np.nan
        featured["symbol"], featured["asset_class"], featured["clock_ms"] = (
            symbol,
            contract["asset_class"],
            hours,
        )
        for horizon in HORIZONS:
            featured[f"phase_{horizon}"] = [phase_at(int(clock), horizon) for clock in hours]
            featured.loc[
                featured[f"phase_{horizon}"] == "excluded_boundary", f"target_{horizon}"
            ] = np.nan
        profiles.append(
            {
                "symbol": symbol,
                "scheduled_hours": len(hours),
                "raw_rows": len(rows),
                "legacy_complete_hours": int(
                    np.isfinite(featured[list(FEATURES)]).all(axis=1).sum()
                ),
                "common_complete_hours": int(
                    np.isfinite(featured[list(FEATURES + NAMES)]).all(axis=1).sum()
                ),
                "new_factor_known_hours": {f: int(np.isfinite(featured[f]).sum()) for f in NAMES},
            }
        )
        samples.append(featured.reset_index(drop=True))
        if number % 100 == 0:
            print({"prepared_symbols": number, "scheduled_symbols": 748}, flush=True)
    panel = pd.concat(samples, ignore_index=True)
    if panel.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("新因子面板重复合约小时")
    panel.to_parquet(output / "hourly_samples.parquet", index=False)
    save(
        output / "coverage.json",
        {"symbols": profiles, "sample_rows": len(panel), "raw_files_verified": len(verified)},
    )
    save(output / "input_identities.json", verified)
    result = evaluate(panel, protocol)
    save(output / "hourly_results.json", result.pop("hourly_records"))
    verify_receipts(source)
    if any(sha(p) != v for p, v in (frozen | bindings).items()):
        raise ValueError("研究期间源码或输入绑定发生变化")
    result.update(
        protocol_sha256=sha(output / "protocol.json"),
        sample_sha256=sha(output / "hourly_samples.parquet"),
    )
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(
        {
            "sample_rows": len(panel),
            "groups": len(result["groups"]),
            "declared_tests": result["declared_tests"],
            "production_eligible": False,
        },
        flush=True,
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
