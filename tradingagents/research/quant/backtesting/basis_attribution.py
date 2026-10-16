"""封存后分解合约变动、指数变动与基差修复，仅作解释，不回写交易筛选。"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import zipfile
from pathlib import Path

import pandas as pd

from tradingagents.research.quant.strategy_audit import read, save, sha


def components(perp_entry, perp_exit, index_entry, index_exit, direction):
    """对数分解是精确的报价变化恒等式，不是可交易指数腿的现金收益。"""
    if direction not in (-1, 1) or any(
        not math.isfinite(v) or v <= 0 for v in (perp_entry, perp_exit, index_entry, index_exit)
    ):
        raise ValueError("解释分解必须使用真实的有限正报价")
    contract = direction * math.log(perp_exit / perp_entry)
    index = direction * math.log(index_exit / index_entry)
    basis = direction * math.log((perp_exit / index_exit) / (perp_entry / index_entry))
    if not math.isclose(contract, index + basis, abs_tol=1e-12):
        raise AssertionError("报价变化分解不守恒")
    return {
        "contract_signed_log_bps": contract * 10000,
        "index_signed_log_bps": index * 10000,
        "basis_signed_log_bps": basis * 10000,
    }


class Quotes:
    """使用同一分钟真正的指数开盘；预定期末才使用对应收盘，不能混用边界。"""

    def __init__(self, root):
        self.paths = {}
        for record in read(Path(root) / "manifest.json")["records"]:
            if record["status"] == "VERIFIED":
                if sha(record["path"]) != record["sha256"]:
                    raise ValueError("指数分解来源发生变化")
                self.paths[record["symbol"], record["day"]] = record["path"]
        self.cache = {}

    def at(self, symbol, clock, closing=False):
        raw_open = clock - 60000 if closing else clock
        key = symbol, pd.Timestamp(raw_open, unit="ms", tz="UTC").date().isoformat()
        if key not in self.paths:
            return None
        if key not in self.cache:
            with zipfile.ZipFile(self.paths[key]) as archive:
                text = archive.read(archive.namelist()[0]).decode()
            values = {}
            for row in csv.reader(io.StringIO(text)):
                if row and row[0].isdigit():
                    values[int(row[0])] = float(row[1]), float(row[4])
            self.cache[key] = values
        prices = self.cache[key].get(raw_open)
        return prices[1 if closing else 0] if prices else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = read(args.run / "result.json")
    quotes = Quotes(args.reference)
    summaries = {}
    for name in (
        "basis_candidate_20m_seed_3.0bps",
        "basis_candidate_20m_seed_10.0bps",
        "basis_sparse_control_20m_seed_3.0bps",
        "basis_sparse_control_20m_seed_10.0bps",
    ):
        path = args.run / f"{name}.json.gz"
        with gzip.open(path, "rt") as stream:
            account = __import__("json").load(stream)
        rows, missing = [], []
        for trade in account["trades"]:
            i1 = quotes.at(trade["symbol"], trade["entry_ms"])
            i2 = quotes.at(
                trade["symbol"], trade["exit_ms"], closing=trade["reason"] == "scheduled_end"
            )
            if i1 is None or i2 is None:
                missing.append(trade["cycle_id"])
                continue
            direction, slip = trade["direction"], account["slippage_fraction"]
            p1, p2 = (
                trade["entry_price"] / (1 + direction * slip),
                trade["exit_price"] / (1 - direction * slip),
            )
            rows.append(
                {
                    "cycle_id": trade["cycle_id"],
                    "symbol": trade["symbol"],
                    "entry_ms": trade["entry_ms"],
                    "exit_ms": trade["exit_ms"],
                    "actual_net_pnl_usdt": trade["net_pnl"],
                    **components(p1, p2, i1, i2, direction),
                }
            )
        summaries[name] = {
            "original_metrics": result["accounts"][name]["full"],
            "rows": rows,
            "missing_reference_cycles": missing,
            "mean_components_bps": {
                k: sum(row[k] for row in rows) / len(rows) if rows else None
                for k in ("contract_signed_log_bps", "index_signed_log_bps", "basis_signed_log_bps")
            },
            "basis_component_positive_cycles": sum(r["basis_signed_log_bps"] > 0 for r in rows),
        }
    save(
        args.output / "analysis.json",
        {
            "cases": summaries,
            "policy_not_modified": True,
            "post_ledger_explanation_only": True,
            "source_result_sha256": sha(args.run / "result.json"),
            "source_code_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    main()
