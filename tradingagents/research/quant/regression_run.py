"""从封存公开分钟生成小时因子面板，冻结后执行预测回归与评分。"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.artifacts import read, save, sha
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.regression import FEATURES, evaluate

MINUTE = 60_000
PHASES = {
    "training": {"start": "2026-09-29T00:00:00+00:00", "end": "2026-10-06T00:00:00+00:00"},
    "retrospective_validation": {
        "start": "2026-10-06T00:00:00+00:00",
        "end": "2026-10-14T00:00:00+00:00",
    },
    "reserved_later_day": {
        "start": "2026-10-14T00:00:00+00:00",
        "end": "2026-10-15T00:00:00+00:00",
    },
}


def ms(value):
    return int(pd.Timestamp(value).value // 1_000_000)


def label_phase(clock, horizon):
    """目标退出必须严格早于阶段终点，训练标签不能读入验证或保留日。"""
    for name, bounds in PHASES.items():
        if ms(bounds["start"]) <= clock and clock + (horizon + 1) * MINUTE < ms(bounds["end"]):
            return name
    return "excluded_boundary"


def features_from_minutes(frame):
    """输入须为连续日历网格，缺一分钟就让其涉及的完整特征窗口未知。"""
    if frame.index.has_duplicates or (
        len(frame) > 1 and not (np.diff(frame.index.to_numpy()) == MINUTE).all()
    ):
        raise ValueError("因子分钟网格重复或压缩了缺失时间")
    close = np.log(frame["close"])
    returns = close.diff()
    values = pd.DataFrame(index=frame.index)
    for length in (3, 20, 60):
        values[f"return_{length}m"] = close - close.shift(length)
    for length in (3, 12):
        quote = frame["quote_volume"].rolling(length, min_periods=length).sum()
        buyer = frame["taker_buy_quote"].rolling(length, min_periods=length).sum()
        values[f"flow_{length}m"] = 2 * buyer / quote.where(quote > 0) - 1
    values["volatility_60m"] = returns.rolling(60, min_periods=60).std(ddof=1)
    values["log_quote_volume_60m"] = np.log1p(
        frame["quote_volume"].rolling(60, min_periods=60).sum()
    )
    complete = (
        frame[["close", "quote_volume", "taker_buy_quote"]]
        .notna()
        .all(axis=1)
        .rolling(61, min_periods=61)
        .sum()
        == 61
    )
    values.loc[~complete, list(FEATURES)] = np.nan
    # 原文按收盘边界索引：t+一分钟开盘对应边界t+两分钟那一行。
    for horizon in (20, 60):
        entry = frame["open"].shift(-2)
        exit_price = frame["open"].shift(-(horizon + 2))
        values[f"target_{horizon}"] = exit_price / entry - 1
    return values


def validated_rows(raw, symbol, seen):
    """逐条核验时钟、价格和金额；同一收盘只许一份原始分钟。"""
    if raw["symbol"] != symbol:
        raise ValueError("原分钟档案合约身份不同")
    result = []
    for row in raw["rows"]:
        if (
            len(row) != 12
            or type(row[0]) is not int
            or type(row[6]) is not int
            or row[0] % MINUTE
            or row[6] != row[0] + MINUTE - 1
        ):
            raise ValueError("原分钟字段或完整结束时间不合法")
        opening, high, low, close, volume, quote, buyer = (
            float(row[k]) for k in (1, 2, 3, 4, 5, 7, 10)
        )
        if (
            not np.isfinite([opening, high, low, close, volume, quote, buyer]).all()
            or min(opening, high, low, close) <= 0
            or min(volume, quote, buyer) < 0
        ):
            raise ValueError("原分钟存在非有限、负数量或非正价格")
        if not low <= min(opening, close) <= max(
            opening, close
        ) <= high or buyer > quote + 1e-8 * max(1, quote):
            raise ValueError("原分钟高低价或买方成交金额不守恒")
        closing = row[0] + MINUTE
        if closing in seen:
            raise ValueError("原分钟存在重复合约时间键")
        seen.add(closing)
        result.append((closing, opening, close, quote, buyer))
    return result


def run(panel, later, output):
    """在任何标签计算前保存协议，所有品种均输出预定小时行并保留缺失。"""
    panel, later, output = Path(panel).resolve(), Path(later).resolve(), Path(output).resolve()
    manifest = read(panel / "manifest.json")
    holdout = read(later / "raw_manifest.json")
    if manifest["leader_labels_read"] is not False or manifest["orders_enabled"] is not False:
        raise ValueError("回归只读取不含带单标签的公开面板")
    symbols = sorted(manifest["contracts"])
    if len(symbols) != 748 or {r["symbol"] for r in holdout["records"]} != set(symbols):
        raise ValueError("回归全集与事前固定清单不同")
    paths = {
        Path(__file__).resolve(),
        Path(evaluate.__code__.co_filename).resolve(),
        Path(read.__code__.co_filename).resolve(),
    }
    plan = source_path("profit_factor_regression_plan.md")
    paths.add(plan)
    identities = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "cz-factor-regression-study.v1",
        "created_ms": time.time_ns() // 1_000_000,
        "phases": PHASES,
        "universe": manifest["contracts"],
        "factors_and_controls": list(FEATURES),
        "panel_manifest_sha256": sha(panel / "manifest.json"),
        "later_raw_manifest_sha256": sha(later / "raw_manifest.json"),
        "later_boundary_receipt_sha256": sha(later / "boundary_receipt.json"),
        "source_sha256": identities,
        "plan_sha256": sha(plan),
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("pandas", "numpy", "pyarrow")},
        },
        "hourly_cross_sections": True,
        "minimum_members": 20,
        "horizons_minutes": [20, 60],
        "feature_time": "COMPLETED_MINUTES_ONLY",
        "entry_delay_minutes": 1,
        "historical_arrival_times_observed": False,
        "historical_eligibility_proven": False,
        "posthoc_download_not_live_oos": True,
        "reserved_later_day_previously_unused_for_returns": holdout["returns_evaluated"] is False,
        "no_dependent_index_or_positioning_subset_used": True,
        "no_composite_score_or_universal_pass_threshold": True,
        "economic_account_evaluated": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "protocol.json", protocol)
    for path in paths:
        target = output / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    old = {}
    for filename, identity in manifest["source_hashes"].items():
        path = Path(filename)
        if path.parent.parent.name == "raw" and path.suffix == ".json":
            old[path.parent.name, path.stem] = path, identity
    new = {r["symbol"]: r for r in holdout["records"]}
    days = list(pd.date_range("2026-09-28", "2026-10-14", freq="D").strftime("%Y-%m-%d"))
    calendar = np.arange(
        ms("2026-09-28T00:01:00Z"), ms("2026-10-15T00:00:00Z") + 1, MINUTE, dtype=np.int64
    )
    hours = np.arange(
        ms(PHASES["training"]["start"]),
        ms(PHASES["reserved_later_day"]["end"]),
        60 * MINUTE,
        dtype=np.int64,
    )
    samples, raw_hashes, profiles = [], {}, []
    for index, symbol in enumerate(symbols):
        rows, seen, absent = [], set(), []
        for day in days:
            if day == "2026-10-14":
                record = new[symbol]
                path, expected = Path(record["path"]), record["sha256"]
            elif (symbol, day) in old:
                path, expected = old[symbol, day]
            else:
                absent.append(day)
                continue
            if sha(path) != expected:
                raise ValueError("原分钟输入与封存身份不同")
            raw_hashes[str(path)] = expected
            rows.extend(validated_rows(read(path), symbol, seen))
        values = (
            pd.DataFrame(
                rows, columns=["clock_ms", "open", "close", "quote_volume", "taker_buy_quote"]
            )
            .set_index("clock_ms")
            .reindex(calendar)
        )
        features = features_from_minutes(values).reindex(hours)
        contract = manifest["contracts"][symbol]
        eligible = (hours > contract["onboard_ms"]) & (
            (contract["delivery_ms"] == 0) | (hours < contract["delivery_ms"])
        )
        features.loc[~eligible, list(FEATURES)] = np.nan
        features["symbol"], features["asset_class"] = symbol, contract["asset_class"]
        features["clock_ms"] = hours
        for horizon in (20, 60):
            features[f"phase_{horizon}"] = [label_phase(int(clock), horizon) for clock in hours]
            features.loc[
                features[f"phase_{horizon}"] == "excluded_boundary", f"target_{horizon}"
            ] = np.nan
        samples.append(features.reset_index(drop=True))
        profiles.append(
            {
                "symbol": symbol,
                "asset_class": contract["asset_class"],
                "raw_rows": len(rows),
                "absent_day_files": absent,
                "scheduled_hours": len(hours),
                "eligible_hours": int(eligible.sum()),
                "complete_feature_hours": int(
                    np.isfinite(features[list(FEATURES)]).all(axis=1).sum()
                ),
                "target_20_known": int(np.isfinite(features["target_20"]).sum()),
                "target_60_known": int(np.isfinite(features["target_60"]).sum()),
            }
        )
        if index % 100 == 99:
            print(f"原文与小时特征：{index + 1}/{len(symbols)}。", flush=True)
    data = pd.concat(samples, ignore_index=True)
    if data.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("回归样本存在重复合约小时")
    data.to_parquet(output / "hourly_samples.parquet", index=False)
    save(output / "input_identities.json", raw_hashes)
    save(
        output / "coverage.json",
        {
            "symbols": profiles,
            "sample_rows": len(data),
            "raw_files_verified": len(raw_hashes),
            "duplicate_symbol_hours": 0,
            "asset_class_members": dict(
                Counter(c["asset_class"] for c in manifest["contracts"].values())
            ),
        },
    )
    result = evaluate(data, protocol)
    hourly = result.pop("hourly_records")
    save(output / "hourly_results.json", hourly)
    result["protocol_sha256"] = sha(output / "protocol.json")
    result["sample_sha256"] = sha(output / "hourly_samples.parquet")
    result["coverage_sha256"] = sha(output / "coverage.json")
    if any(sha(path) != identity for path, identity in identities.items()):
        raise ValueError("回归期间源码或冻结计划发生变化")
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(
        {
            "sample_rows": len(data),
            "raw_files_verified": len(raw_hashes),
            "groups": len(result["groups"]),
            "declared_rank_ic_tests": result["declared_rank_ic_tests"],
            "production_eligible": False,
        },
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--later-day", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.panel, args.later_day, args.output)


if __name__ == "__main__":
    main()
