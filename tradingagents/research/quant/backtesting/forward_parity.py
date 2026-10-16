"""将独立前向特征实现与封存价格面板核对，不计算收益或历史可用性。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.research.quant.backtesting.forward_inputs import (
    HISTORY_BARS,
    Bars,
    price_ready,
    snapshot,
)
from tradingagents.research.quant.backtesting.search import COLUMNS
from tradingagents.research.quant.backtesting.signals import MINUTE
from tradingagents.research.quant.reference import market_features as market
from tradingagents.research.quant.strategy_audit import read, save, sha


def run(panel, output, at):
    """输入时刻在计算前固定，只使用价格原文，不读取账户或标签。"""
    clock = int(at.value // 1_000_000)
    if at.tzinfo is None or clock % MINUTE:
        raise ValueError("核对时刻必须是有时区的完整分钟")
    manifest = read(panel / "manifest.json")
    if (
        manifest.get("orders_enabled") is not False
        or manifest.get("leader_labels_read") is not False
    ):
        raise ValueError("特征核对只接受纯公开封存面板")
    earliest = clock - (HISTORY_BARS - 1) * MINUTE
    days = {
        pd.Timestamp(t - MINUTE, unit="ms", tz="UTC").date().isoformat() for t in (earliest, clock)
    }
    raw_paths = [
        Path(p)
        for p in manifest["source_hashes"]
        if "/raw/" in p and any(p.endswith(f"/{d}.json") for d in days)
    ]
    expected_path = panel / f"candidate_{at.strftime('%Y%m%d')}.parquet"
    expected_hash = next(
        r["sha256"] for r in manifest["daily_files"] if r["path"] == expected_path.name
    )
    if sha(expected_path) != expected_hash:
        raise ValueError("封存特征面板身份变化")
    sources = [
        Path(__file__).resolve(),
        Path(snapshot.__code__.co_filename).resolve(),
        Path(market.__file__).resolve(),
        Path(read.__code__.co_filename).resolve(),
        *Path(market.__file__).parent.joinpath("factors").glob("*.py"),
    ]
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        {
            "schema": "cz-profit-forward-feature-parity.v1",
            "at": at.isoformat(),
            "panel_manifest_sha256": sha(panel / "manifest.json"),
            "expected_parquet_sha256": sha(expected_path),
            "source_files": {str(p): sha(p) for p in sources},
            "raw_sources": {str(p): manifest["source_hashes"][str(p)] for p in raw_paths},
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-10,
            "historical_feature_comparison_only": True,
            "historic_receipt_time_claimed": False,
            "profitability_evaluated": False,
            "orders_enabled": False,
            "leader_labels_read": False,
        },
    )
    bars = Bars(manifest["symbols"])
    for path in raw_paths:
        checksum = sha(path)
        if checksum != manifest["source_hashes"][str(path)]:
            raise ValueError("封存原始分钟身份变化")
        value = read(path)
        received = time.time_ns() // 1_000_000
        for row in value["rows"]:
            if earliest <= row[6] + 1 <= clock:
                bars.add(
                    value["symbol"],
                    row,
                    received_ms=received,
                    source_sha256=checksum,
                    completed_verified=True,
                )
    frame, status = snapshot(
        bars, manifest["contracts"], closing=clock, decision_ms=time.time_ns() // 1_000_000
    )
    if frame is None:
        save(
            output / "result.json",
            {
                "compared": False,
                "input_status": status,
                "profitability_evaluated": False,
                "orders_enabled": False,
            },
        )
        raise ValueError("原历史在该时刻不满足前向共同分钟完整性，不能核对完整截面")
    expected = (
        pd.read_parquet(expected_path, columns=COLUMNS, filters=[("bar_time", "==", at)])
        .droplevel("bar_time")
        .sort_index()
    )
    frame = frame.sort_index()
    if not frame.index.equals(expected.index):
        raise ValueError("独立前向实现的固定集合发生变化")
    columns = [c for c in COLUMNS if c not in ("asset_class", "eligible", "observed")]
    numeric = {}
    for c in columns:
        left, right = frame[c].to_numpy(float), expected[c].to_numpy(float)
        match = np.isclose(left, right, equal_nan=True, rtol=1e-10, atol=1e-12)
        finite = np.isfinite(left) & np.isfinite(right)
        numeric[c] = {
            "matched": int(match.sum()),
            "maximum_absolute_difference": float(np.max(np.abs(left[finite] - right[finite])))
            if finite.any()
            else None,
        }
    identities = all(frame[c].equals(expected[c]) for c in ("asset_class", "eligible", "observed"))
    result = {
        "members": len(frame),
        "numeric": numeric,
        "identities_match": identities,
        "price_ready_symbols_identical": price_ready(frame) == price_ready(expected),
        "all_numeric_match": all(v["matched"] == len(frame) for v in numeric.values()),
        "historic_receipt_time_claimed": False,
        "profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(output / "result.json", result)
    (output / "source").mkdir()
    for number, p in enumerate(sources):
        (output / "source" / f"{number:02d}_{p.name}").write_bytes(p.read_bytes())
    for p in sources:
        if sha(p) != read(output / "protocol.json")["source_files"][str(p)]:
            raise ValueError("特征核对过程中源码变化")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--at", required=True)
    args = parser.parse_args()
    run(args.panel.resolve(), args.output.resolve(), pd.Timestamp(args.at))


if __name__ == "__main__":
    main()
