"""在原退出时刻诊断同期推送盘口，不改库存、不补成交、不重算策略收益。"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from tradingagents.dataflows.research.live_capture import MINUTE
from tradingagents.research.quant.backtesting.microstructure import source_available
from tradingagents.research.quant.backtesting.microstructure_pairing import (
    ReceivedMessages,
    verify_receipts,
)
from tradingagents.research.quant.backtesting.paper_account import PaperAccount
from tradingagents.research.quant.backtesting.paper_inputs import number
from tradingagents.research.quant.backtesting.paper_live_audit import require_loaded_sources
from tradingagents.research.quant.backtesting.segmented_chain import segment_entries
from tradingagents.research.quant.paths import PROJECT_ROOT
from tradingagents.research.quant.strategy_audit import read, save, sha


def quote_conditions(value, epoch, *, now, due, direction, quantity):
    """只检验原数量、退出边界后的新鲜公开盘口，最新小盘口不能退回旧的大盘口。"""
    if type(now) is not int or type(due) is not int or due > now or direction not in {-1, 1}:
        raise ValueError("退出诊断时钟或方向不合法")
    required = number(quantity, positive=True)
    side = "bid" if direction == 1 else "ask"
    error = source_available(value, epoch, now, now, quote=True)
    if error is None and now - value["received_ms"] > 5000:
        error = "QUOTE_RECEIPT_TOO_OLD"
    if error is None and (value["received_ms"] < due or value["transaction_ms"] < due):
        error = "QUOTE_PRECEDES_ORIGINAL_EXIT_DUE"
    displayed = number(value[side + "_qty"]) if error is None else None
    if error is None and displayed < required:
        error = "ORIGINAL_QUANTITY_EXCEEDS_DISPLAYED_BEST_SIZE"
    return {
        "market_quote_conditions_met": error is None,
        "reason": error,
        "side": side,
        "required_original_quantity": str(required),
        "displayed_side_quantity": str(displayed) if displayed is not None else None,
        "selected_quote": value,
        "known_epoch": epoch,
        "real_fill_proven": False,
        "orders_enabled": False,
    }


def exit_attempts(frames, protocol, expected):
    """从相同原账户完整重放退出尝试，成功与失败均保留，最终状态必须完全相同。"""
    models = {
        name: PaperAccount(
            protocol["symbols"], int(name), period_end_ms=protocol["simulation_end_ms"]
        )
        for name in ("3", "10")
    }
    for frame in frames:
        now = frame["actual_ms"]
        for name, model in models.items():
            fills_before, declines_before = len(model.fills), len(model.declines)
            state = model.apply(frame)
            closes = [f for f in state["fills"][fills_before:] if f["action"] == "CLOSE"]
            declines = [
                d
                for d in state["declines"][declines_before:]
                if d["reason"] in {"EXIT_QUOTE_UNAVAILABLE", "EXIT_DISPLAYED_CAPACITY_INSUFFICIENT"}
            ]
            for event in closes + declines:
                position = (
                    event if event.get("action") == "CLOSE" else state["positions"][event["symbol"]]
                )
                due = [(position["entry_ms"] + 20 * MINUTE, 1, "time")]
                due.extend(
                    (position[key], priority, reason)
                    for key, priority, reason in (
                        ("risk_due_ms", 0, "risk"),
                        ("profit_due_ms", 2, "profit"),
                    )
                    if key in position
                )
                if protocol["simulation_end_ms"] is not None:
                    due.append((protocol["simulation_end_ms"], 0, "scheduled_end"))
                stamp, _, cause = min(due)
                if stamp > now:
                    raise ValueError("原账户在退出权限之前产生退出尝试")
                yield {
                    "cost_account": name,
                    "symbol": position["symbol"],
                    "episode": position["episode"],
                    "cycle_id": position["cycle_id"],
                    "actual_ms": now,
                    "canonical_close_ms": frame["canonical_close_ms"],
                    "phase": frame.get("phase", "decision"),
                    "exit_due_ms": stamp,
                    "exit_due_reason": cause,
                    "direction": position["direction"],
                    "original_quantity": position["quantity"],
                    "original_outcome": "SIMULATED_CLOSE"
                    if event.get("action") == "CLOSE"
                    else event["reason"],
                    "original_quote_body_sha256": frame["quotes"]["body_sha256"],
                }
    for name, model in models.items():
        if model.checkpoint() != expected[name]:
            raise ValueError("退出尝试诊断改变了原账户或遗漏了原输入")


def run(source, audited, captures, output):
    """只对同一原库存检查报价条件；满足条件也不能推断另一账户路径或收益。"""
    source, audited, output = Path(source), Path(audited), Path(output)
    verify_receipts(source)
    verify_receipts(audited)
    protocol, checkpoint = read(source / "protocol.json"), read(source / "checkpoint.json")
    binding, audited_result = (
        read(audited / "run_binding.json"),
        read(audited / "audit/result.json"),
    )
    snapshot = read(source / "prefix_snapshot.json")
    if (
        binding["exit_code"] != 0
        or binding["audited_snapshot_receipts_sha256"] != sha(source / "receipts.json")
        or binding["audit_receipts_sha256"] != sha(audited / "audit/receipts.json")
    ):
        raise ValueError("退出诊断的独立财务审计未绑定当前前缀")
    if (
        audited_result["account_frames_verified"] != checkpoint["frame_count"]
        or audited_result["source_records_verified"] != snapshot["source_prefix"]["event_count"]
    ):
        raise ValueError("退出诊断的原文与账户未完整核验")
    require_loaded_sources(protocol)
    runtime = {
        "python": sys.version,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("requests", "websockets", "pandas", "numpy")
        },
    }
    if runtime != protocol["runtime"]:
        raise ValueError("退出诊断运行库与原研究冻结版本不同")
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if (name == "tradingagents" or name.startswith("tradingagents.")) and getattr(
            module, "__file__", None
        ):
            paths.add(Path(module.__file__).resolve())
    identities = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    package = PROJECT_ROOT
    for path in paths:
        target = output / "helper_sources" / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    save(
        output / "protocol.json",
        {
            "schema": "cz-original-inventory-exit-quote-probe.v1",
            "source_sha256": identities,
            "runtime": runtime,
            "financial_snapshot_receipts_sha256": sha(source / "receipts.json"),
            "financial_audit_receipts_sha256": sha(audited / "receipts.json"),
            "observer_receipts_sha256": {str(p): sha(Path(p) / "receipts.json") for p in captures},
            "maximum_quote_age_ms": 5000,
            "quote_transaction_and_receipt_after_original_due_required": True,
            "purpose": "SAME_ORIGINAL_INVENTORY_QUOTE_CONDITIONS_ONLY",
            "orders_enabled": False,
        },
    )
    rows = []
    with TemporaryDirectory(prefix="cz-exit-quote-probe-") as directory:
        messages = ReceivedMessages(
            captures, Path(directory) / "observations.db", sha(source / "protocol.json")
        )
        try:
            frames = (
                wrapper["frame"] for _, wrapper, _ in segment_entries(source / "account_frames")
            )
            for attempt in exit_attempts(frames, protocol, checkpoint["accounts"]):
                value, epoch = messages.latest(
                    "quote", attempt["symbol"], attempt["actual_ms"], attempt["actual_ms"]
                )
                diagnosis = quote_conditions(
                    value,
                    epoch,
                    now=attempt["actual_ms"],
                    due=attempt["exit_due_ms"],
                    direction=attempt["direction"],
                    quantity=attempt["original_quantity"],
                )
                rows.append({**attempt, "quote_diagnostic": diagnosis})
            sources = [
                {"root": s["root"], "receipt_sha256": s["receipt_sha256"], "audit": s["audit"]}
                for s in messages.sources
            ]
        finally:
            messages.close()
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("退出诊断期间源码发生变化")
    summary = {}
    for name in ("3", "10"):
        selected = [r for r in rows if r["cost_account"] == name]
        summary[name] = {
            "original_exit_attempts": len(selected),
            "distinct_original_episodes": len({r["episode"] for r in selected}),
            "original_outcomes": dict(Counter(r["original_outcome"] for r in selected)),
            "quote_conditions_met": sum(
                r["quote_diagnostic"]["market_quote_conditions_met"] for r in selected
            ),
            "diagnostic_reasons": dict(
                Counter(
                    r["quote_diagnostic"]["reason"]
                    for r in selected
                    if r["quote_diagnostic"]["reason"]
                )
            ),
            "outcome_by_quote_condition": dict(
                Counter(
                    r["original_outcome"]
                    + ":"
                    + ("MET" if r["quote_diagnostic"]["market_quote_conditions_met"] else "NOT_MET")
                    for r in selected
                )
            ),
        }
    result = {
        "schema": "cz-original-inventory-exit-quote-probe-result.v1",
        "protocol_sha256": sha(output / "protocol.json"),
        "summary": summary,
        "attempts": rows,
        "observer_sources": sources,
        "original_account_replay_exact": True,
        "original_inventory_preserved": True,
        "alternative_account_replayed": False,
        "counterfactual_pnl_computed": False,
        "new_alpha_proven": False,
        "real_fills_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-prefix", type=Path, required=True)
    parser.add_argument("--financial-audit", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.financial_prefix.resolve(),
        args.financial_audit.resolve(),
        [p.resolve() for p in args.capture],
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
