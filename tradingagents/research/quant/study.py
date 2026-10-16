"""在 TradingAgents 内独立重算封存因子面板，保留原统计和历史使用边界。"""

import importlib.metadata
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read, save, sha, verify_receipts
from .regression import FEATURES, evaluate


def validate_panel(samples, protocol):
    """拒绝重叠阶段、重复截面、阶段混用和训练标签穿越；不压缩缺失样本。"""
    if any(protocol.get(k) is not False for k in ("leader_labels_read", "orders_enabled")):
        raise ValueError("量化面板必须声明未读带单标签且未启用订单")
    phases = protocol["phases"]
    if "training" not in phases or not phases:
        raise ValueError("缺少冻结训练区间")
    bounds = {}
    for name, interval in phases.items():
        start, end = (pd.Timestamp(interval[k]) for k in ("start", "end"))
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("阶段必须使用明确时区和正向区间")
        bounds[name] = (start.value // 1_000_000, end.value // 1_000_000)
    ordered = sorted(bounds.values())
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("冻结训练与验证阶段不能重叠")
    train_end = bounds["training"][1]
    if any(start < train_end for name, (start, _) in bounds.items() if name != "training"):
        raise ValueError("验证区间必须位于训练之后")
    required = set(FEATURES) | {"symbol", "asset_class", "clock_ms"}
    required |= {f"{prefix}_{h}" for h in (20, 60) for prefix in ("target", "phase")}
    if not required <= set(samples) or samples.empty:
        raise ValueError("量化面板缺少预定特征、标签或成员")
    if samples[["symbol", "asset_class", "clock_ms"]].isna().any().any():
        raise ValueError("合约、资产类别和时间键不能缺失")
    if samples.duplicated(["symbol", "clock_ms"]).any():
        raise ValueError("重复合约小时不能进入回归")
    clocks = samples["clock_ms"].to_numpy()
    if not np.isfinite(clocks).all() or (clocks % 3_600_000 != 0).any():
        raise ValueError("回归面板必须使用完整UTC小时")
    for horizon in (20, 60):
        column = f"phase_{horizon}"
        if not set(samples[column]) <= set(phases) | {"excluded_boundary"}:
            raise ValueError("面板出现未登记阶段")
        if (samples.groupby(["asset_class", "clock_ms"])[column].nunique() != 1).any():
            raise ValueError("同一截面不能混用训练与验证标签")
        # 阶段由固定时钟推导，不能把亏损时点改名为边界后排除。
        expected = np.full(len(samples), "excluded_boundary", dtype=object)
        for name, (start, end) in bounds.items():
            expected[(clocks >= start) & (clocks + (horizon + 1) * 60_000 < end)] = name
        if not (samples[column].to_numpy() == expected).all():
            raise ValueError("面板阶段与固定时钟边界不同")


def run(source, output):
    """来源只读；事前固定本次源码和输入身份，不把重算变成新样本外。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("新研究输出不能覆盖或包含原来源")
    verify_receipts(source)
    protocol = read(source / "protocol.json")
    samples = pd.read_parquet(source / "hourly_samples.parquet")
    validate_panel(samples, protocol)
    package = Path(__file__).parent
    identities = {
        str(p.relative_to(package)): sha(p)
        for p in package.rglob("*")
        if p.suffix in {".py", ".md"}
    }
    source_receipt = sha(source / "receipts.json")
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "ta-quant-regression-replay.v1",
            "created_ms": time.time_ns() // 1_000_000,
            "research_owner": "TradingAgents",
            "phases": protocol["phases"],
            "parent_receipts_sha256": source_receipt,
            "source_sha256": identities,
            "sample_sha256": sha(source / "hourly_samples.parquet"),
            "runtime": {
                "python": sys.version,
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow")
                },
            },
            "all_history_previously_viewed": True,
            "independent_oos": False,
            "leader_labels_read": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    for name in identities:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / name).read_bytes())
    # 固定面板随结果保存，使增量检验和独立复核可以直接接续，原来源仍只读。
    for name in ("hourly_samples.parquet", "input_identities.json", "coverage.json"):
        shutil.copyfile(source / name, output / name)
    result = evaluate(samples, protocol)
    save(output / "hourly_results.json", result.pop("hourly_records"))
    verify_receipts(source)
    if sha(source / "receipts.json") != source_receipt or any(
        sha(package / name) != value for name, value in identities.items()
    ):
        raise ValueError("计算期间输入封存或研究源码发生变化")
    result.update(
        research_owner="TradingAgents",
        independent_oos=False,
        all_history_previously_viewed=True,
        protocol_sha256=sha(output / "protocol.json"),
    )
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    return result
