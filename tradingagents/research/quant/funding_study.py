"""把完整已结算普通资金费按固定日历加入价格响应回归，不推断未来现金。"""

import argparse
import importlib.metadata
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read, save, sha, verify_receipts
from .backtesting.search import funding_source_path, load_funding
from .factors.funding import NAMES, calculate
from .linear_family import evaluate
from .regression import CONTROLS, FACTORS, FEATURES
from .regression_run import PHASES, ms

MODELS = {
    "controls": CONTROLS,
    "legacy": FEATURES,
    **{f"add_{f}": FEATURES + (f,) for f in NAMES},
    "expanded": FEATURES + NAMES,
}
CONTRASTS = {
    **{f"add_{f}_vs_legacy": ("legacy", f"add_{f}") for f in NAMES},
    "expanded_vs_legacy": ("legacy", "expanded"),
    **{f"expanded_vs_add_{f}": (f"add_{f}", "expanded") for f in NAMES},
}


def regular_events(root, capture, manifest):
    """逐叶验证请求覆盖和原文类别；特殊股息只从预测输入移除。"""
    first, last = ms(capture["start"]), ms(capture["end"])
    cursor, kinds = first, Counter()
    for leaf in sorted(manifest["leaves"], key=lambda r: r["start_ms"]):
        if leaf["start_ms"] != cursor or leaf["end_ms"] < cursor:
            raise ValueError("资金费请求区间有缺口、重叠或倒序")
        for row in read(funding_source_path(root, leaf)):
            if not leaf["start_ms"] <= row["fundingTime"] <= leaf["end_ms"]:
                raise ValueError("资金费事件越出其请求叶区间")
        cursor = leaf["end_ms"] + 1
    if cursor != last:
        raise ValueError("资金费请求区间未完整覆盖固定日期")
    events = load_funding(root)
    raw = read(root / "funding_events.json")
    types = {}
    for row in raw:
        kind = row.get("rateType")
        if kind not in {"Regular", "Special"}:
            raise ValueError("未声明资金费事件类型不能进入模型")
        types[row["symbol"], row["fundingTime"]] = kind
        kinds[kind] += 1
    return [e for e in events if types[e["symbol"], e["time_ms"]] == "Regular"], dict(kinds)


def run(source, funding_root, output):
    """先冻结公式与输入，再计算216项共同人口检验，保留所有未知行。"""
    source, funding_root, output = (Path(p).resolve() for p in (source, funding_root, output))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, funding_root)):
        raise ValueError("新实验不能覆盖或包含来源")
    verify_receipts(source)
    parent = read(source / "protocol.json")
    capture, manifest = (
        read(funding_root / p) for p in ("capture_plan.json", "funding_manifest.json")
    )
    if (
        parent["phases"] != PHASES
        or len(parent["universe"]) != 748
        or set(capture["funding_symbols"]) != set(parent["universe"])
        or capture["orders_enabled"] is not False
        or capture["leader_labels_read"] is not False
        or parent["orders_enabled"] is not False
        or parent["leader_labels_read"] is not False
    ):
        raise ValueError("只能沿用原完整748合约和纯公开研究阶段")
    package = Path(__file__).resolve().parents[2]
    sources = sorted(
        set(package.rglob("*.py")) | set((Path(__file__).parent / "plans").glob("*.md"))
    )
    sources.append(Path(__file__).parent / "requirements-forward.txt")
    frozen = {str(p): sha(p) for p in sources}
    identities = {str(source / "receipts.json"): sha(source / "receipts.json")}
    identities.update(
        {
            str(funding_root / p): sha(funding_root / p)
            for p in ("capture_plan.json", "funding_manifest.json", "funding_events.json")
        }
    )
    identities.update(
        {str(funding_source_path(funding_root, p)): p["sha256"] for p in manifest["leaves"]}
    )
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "ta-settled-funding-study.v1",
        "account_family": "settled_funding",
        "created_ms": time.time_ns() // 1_000_000,
        "research_owner": "TradingAgents",
        "universe": parent["universe"],
        "phases": PHASES,
        "features": FEATURES + NAMES,
        "scored_factors": FACTORS + NAMES,
        "models": MODELS,
        "contrasts": CONTRASTS,
        "bh_count_unknown_in_declared_family": True,
        "horizons_minutes": (20, 60, 240),
        "source_sha256": frozen,
        "input_identities": identities,
        "funding_information_delay_ms": 300_000,
        "funding_coverage_start_ms": ms(capture["start"]),
        "funding_coverage_end_ms": ms(capture["end"]) - 1,
        "feature_event_types": ["Regular"],
        "account_event_types": ["Regular", "Special"],
        "account_slippage_bps": [3, 10, 11],
        "target_is_price_return_only": True,
        "historical_first_publication_time_proven": False,
        "all_history_previously_viewed": True,
        "independent_oos": False,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
        "runtime": {
            "python": sys.version,
            "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")},
        },
    }
    save(output / "protocol.json", protocol)
    for p in sources:
        target = output / "source" / p.relative_to(package.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    events, type_counts = regular_events(funding_root, capture, manifest)
    by_symbol = defaultdict(list)
    for event in events:
        if event["symbol"] in parent["universe"]:
            by_symbol[event["symbol"]].append(event)
    samples = pd.read_parquet(
        source / "hourly_samples.parquet",
        columns=[
            "symbol",
            "asset_class",
            "clock_ms",
            *FEATURES,
            *[f"{prefix}_{h}" for h in (20, 60, 240) for prefix in ("target", "phase")],
        ],
    )
    if samples.duplicated(["symbol", "clock_ms"]).any() or set(samples.symbol) != set(
        parent["universe"]
    ):
        raise ValueError("封存价格面板的成员或小时身份不同")
    prepared, coverage = [], []
    for symbol, frame in samples.groupby("symbol", sort=True):
        frame = frame.sort_values("clock_ms").set_index("clock_ms")
        computed = calculate(
            frame.index, by_symbol[symbol], ms(capture["start"]), ms(capture["end"]) - 1
        )
        joined = frame.join(computed)
        complete = np.isfinite(joined[list(FEATURES + NAMES)]).all(axis=1)
        coverage.append(
            {
                "symbol": symbol,
                "scheduled_hours": len(frame),
                "regular_events": len(by_symbol[symbol]),
                "common_complete_hours": int(complete.sum()),
                "known_factor_hours": {f: int(np.isfinite(joined[f]).sum()) for f in NAMES},
            }
        )
        prepared.append(joined.reset_index())
    panel = pd.concat(prepared, ignore_index=True)
    panel.to_parquet(output / "hourly_samples.parquet", index=False)
    save(
        output / "coverage.json",
        {
            "symbols": coverage,
            "sample_rows": len(panel),
            "event_type_counts_before_universe_filter": type_counts,
            "regular_events_inside_universe": sum(map(len, by_symbol.values())),
        },
    )
    save(output / "input_identities.json", identities)
    result = evaluate(panel, protocol)
    save(output / "hourly_results.json", result.pop("hourly_records"))
    verify_receipts(source)
    if any(sha(p) != value for p, value in (frozen | identities).items()):
        raise ValueError("计算期间源码或输入身份变化")
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
            "rows": len(panel),
            "declared_tests": result["declared_tests"],
            "production_eligible": False,
        },
        flush=True,
    )
    return result


def main(argv=None):
    """独立命令或统一量化入口均使用同一固定研究设计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "funding-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source, args.funding_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
