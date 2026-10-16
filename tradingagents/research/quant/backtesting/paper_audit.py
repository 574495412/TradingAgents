"""从原始报价、成交和资金费独立重建现金，不调用账户记账方法。"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext

from tradingagents.dataflows.research.live_capture import MINUTE, digest
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS
from tradingagents.research.quant.backtesting.paper_inputs import (
    contract_rules,
    funding_window,
    number,
    quote,
)


def equal(actual, expected, label):
    """只容许十进制运算的末位舍入，不容许漏费或替换来源。"""
    if abs(number(actual) - expected) > Decimal("1e-30"):
        raise ValueError(f"独立财务重建不一致：{label}")


def audit(frames, checkpoint):
    """独立核对每帧含浮亏权益、数量预算、成交来源和逐类型资金费。"""
    with localcontext() as context:
        context.prec = 40
        return _audit(frames, checkpoint)


def _audit(frames, checkpoint):
    protocol = checkpoint["protocol"]
    if (
        protocol["hypothesis"] != HYPOTHESIS
        or protocol["execution_version"] != "received_quote_paper_v1"
        or protocol["initial_equity"] != "1000"
        or protocol["position_fraction"] != "0.125"
        or protocol["fee_bps_each_side"] != 5
        or protocol["slippage_bps_each_side"] not in (3, 10)
        or protocol["maximum_positions"] != 4
        or protocol["maximum_gross_equity_ratio"] != 1
        or protocol["hold_ms"] != 20 * MINUTE
        or protocol["take_profit_fraction"] != "0.01"
        or protocol["maximum_drawdown"] != "0.08"
        or protocol["orders_enabled"] is not False
        or protocol["production_eligible"] is not False
    ):
        raise ValueError("财务审计协议与固定账户不一致")
    cash = peak = Decimal("1000")
    drawdown = Decimal(0)
    fee_rate, slip = Decimal("0.0005"), Decimal(protocol["slippage_bps_each_side"]) / 10000
    inventory, cycles, seen, payments, used_episodes = {}, {}, {}, [], set()
    halted, previous_decision, previous_closing = False, None, None
    required_errors = set()
    fills, cursor, head, signal_frames = checkpoint["fills"], 0, None, {}
    signal_requirements = Counter(f["signal_ms"] for f in fills if f["action"] == "OPEN")
    completed = 0
    for index, frame in enumerate(frames):
        if index >= len(checkpoint["curve"]):
            raise ValueError("权益曲线与完整输入帧数量不一致")
        completed = index + 1
        now = frame["actual_ms"]
        if frame.get("source_restart") or (
            previous_closing is not None and frame["canonical_close_ms"] > previous_closing + MINUTE
        ):
            required_errors.add("OBSERVATION_GAP")
        if signal_requirements[now]:
            signal_frames[now] = frame
        head = digest({"sequence": index, "previous_sha256": head, "frame_sha256": digest(frame)})
        marks = {s: number(v["row"][4], positive=True) for s, v in frame["marks"].items()}
        for raw in frame.get("funding", []):
            window = funding_window(raw, now)
            for event in window["events"]:
                key = event["symbol"], event["time_ms"], event["rate_type"]
                economics = event["rate"], event["mark_price"]
                if key in seen:
                    if seen[key] != economics:
                        raise ValueError("资金费原文存在经济修订")
                    continue
                seen[key] = economics
                owners = [
                    p
                    for p in cycles.values()
                    if p["symbol"] == key[0] and p["entry_ms"] < key[1] <= p.get("exit_ms", now)
                ]
                if len(owners) > 1:
                    raise ValueError("独立重建出现重叠库存")
                if owners:
                    owner = owners[0]
                    payment = (
                        -owner["direction"]
                        * owner["quantity"]
                        * number(event["rate"])
                        * number(event["mark_price"])
                    )
                    cash += payment
                    owner["funding"] += payment
                    payments.append({**event, "cycle_id": owner["cycle_id"], "cash": str(payment)})
                    if previous_decision is not None and key[1] <= previous_decision:
                        halted = True
                        required_errors.add("FUNDING_ARRIVED_AFTER_A_FINANCIAL_DECISION")
        # 风险曲线也观察成交前权益，避免只核对成交后而漏掉已触及的回撤。
        if not set(inventory) - set(marks):
            before = cash + sum(
                (
                    p["direction"] * p["quantity"] * (marks[s] - p["price"])
                    for s, p in inventory.items()
                ),
                Decimal(0),
            )
            peak = max(peak, before)
            drawdown = max(drawdown, 1 - before / peak)
            halted = halted or drawdown >= Decimal("0.08") or before <= 0
        else:
            required_errors.add("OPEN_POSITION_MARK_UNAVAILABLE")
        ending = protocol["period_end_ms"]
        halted = halted or (ending is not None and now >= ending)
        for symbol, p in inventory.items():
            if halted:
                p.setdefault("risk_due_ms", (now + 2 * MINUTE - 1) // MINUTE * MINUTE)
            if symbol in marks and p["direction"] * (marks[symbol] / p["price"] - 1) >= Decimal(
                "0.01"
            ):
                p.setdefault("profit_due_ms", (now + 2 * MINUTE - 1) // MINUTE * MINUTE)
        while cursor < len(fills):
            fill = fills[cursor]
            action = fill["action"]
            stamp = fill["entry_ms"] if action == "OPEN" else fill["exit_ms"]
            if stamp > now:
                break
            if stamp != now:
                raise ValueError("成交不属于当前真实接收帧")
            symbol, direction, qty = (
                fill["symbol"],
                fill["direction"],
                number(fill["quantity"], positive=True),
            )
            if symbol not in protocol["symbols"] or direction not in (-1, 1):
                raise ValueError("成交合约或方向错误")
            if action == "OPEN":
                if frame.get("phase", "decision") != "decision":
                    raise ValueError("仅退出帧不能开仓")
                if set(inventory) - set(marks) or halted:
                    raise ValueError("风险停机或缺失估值时不得开仓")
                signal = signal_frames.get(fill["signal_ms"])
                recheck = frame["evidence"][symbol]
                due = (fill["signal_ms"] + 2 * MINUTE - 1) // MINUTE * MINUTE
                if (
                    not signal
                    or signal["evidence"][symbol]["direction"] != direction
                    or fill["scheduled_boundary_ms"] != due
                    or fill["recheck_ms"] != recheck["actual_decision_ms"]
                    or not due <= fill["recheck_ms"] <= now < due + MINUTE
                    or recheck["direction"] != direction
                    or symbol in inventory
                    or len(inventory) >= 4
                    or fill["episode"] in used_episodes
                ):
                    raise ValueError("入场时序、复核、活动唯一性或共享容量错误")
                used_episodes.add(fill["episode"])
                q = quote(frame["quotes"], symbol, now, not_before_ms=fill["recheck_ms"])
                side = "ask" if direction == 1 else "bid"
                price = q[side] * (1 + direction * slip)
                rules = contract_rules(frame["contracts"], symbol, now)
                if any(qty < lo or qty > hi or qty % step for lo, hi, step in rules["lots"]):
                    raise ValueError("成交数量违反合约限制")
                if qty * min(marks[symbol], price) < rules["minimum_notional"]:
                    raise ValueError("成交低于最低金额")
                eq = cash + sum(
                    (
                        p["direction"] * p["quantity"] * (marks[s] - p["price"])
                        for s, p in inventory.items()
                    ),
                    Decimal(0),
                )
                if qty * price > eq * Decimal("0.125"):
                    raise ValueError("入场超过当时共享权益预算")
                fee = qty * price * fee_rate
                equal(fill["entry_price"], price, "入场价")
                equal(fill["entry_fee"], fee, "入场费")
                cash -= fee
                p = {
                    "symbol": symbol,
                    "cycle_id": fill["cycle_id"],
                    "entry_ms": now,
                    "direction": direction,
                    "quantity": qty,
                    "price": price,
                    "entry_fee": fee,
                    "funding": Decimal(0),
                }
                inventory[symbol] = cycles[fill["cycle_id"]] = p
                gross = sum((v["quantity"] * marks[s] for s, v in inventory.items()), Decimal(0))
                after = cash + sum(
                    (
                        v["direction"] * v["quantity"] * (marks[s] - v["price"])
                        for s, v in inventory.items()
                    ),
                    Decimal(0),
                )
                if gross > after + Decimal("1e-30"):
                    raise ValueError("入场超过一倍权益敞口")
                if fill["rules_source_sha256"] != rules["source_sha256"]:
                    raise ValueError("数量限制来源被替换")
                signal_requirements[fill["signal_ms"]] -= 1
                if not signal_requirements[fill["signal_ms"]]:
                    del signal_frames[fill["signal_ms"]]
            elif action == "CLOSE":
                p = inventory.pop(symbol)
                if (
                    p["cycle_id"] != fill["cycle_id"]
                    or qty != p["quantity"]
                    or direction != p["direction"]
                ):
                    raise ValueError("平仓不属于独立重建的真实模拟库存")
                due = [(p["entry_ms"] + 20 * MINUTE, 1, "time")]
                due.extend(
                    (p[k], priority, reason)
                    for k, priority, reason in (
                        ("risk_due_ms", 0, "risk"),
                        ("profit_due_ms", 2, "profit"),
                    )
                    if k in p
                )
                if ending is not None:
                    due.append((ending, 0, "scheduled_end"))
                available, _, reason = min(due)
                if now < available or fill["reason"] != reason:
                    raise ValueError("退出原因或可执行时间错误")
                if now >= available + MINUTE:
                    required_errors.add("EXIT_DELAY_EXCEEDED_ONE_MINUTE")
                q = quote(frame["quotes"], symbol, now, not_before_ms=available)
                side = "bid" if direction == 1 else "ask"
                price = q[side] * (1 - direction * slip)
                fee = qty * price * fee_rate
                gross = direction * qty * (price - p["price"])
                equal(fill["exit_price"], price, "退出价")
                equal(fill["exit_fee"], fee, "退出费")
                equal(fill["gross_pnl"], gross, "价差损益")
                cash += gross - fee
                p.update(exit_ms=now, exit_price=price, exit_fee=fee, gross_pnl=gross)
            else:
                raise ValueError("未知成交动作")
            if qty > q[f"{side}_qty"] or fill["quote_source_sha256"] != q["source_sha256"]:
                raise ValueError("成交超过公开显示容量或报价来源被替换")
            cursor += 1
        equity = (
            None
            if set(inventory) - set(marks)
            else cash
            + sum(
                (
                    p["direction"] * p["quantity"] * (marks[s] - p["price"])
                    for s, p in inventory.items()
                ),
                Decimal(0),
            )
        )
        curve = checkpoint["curve"][index]
        if (
            curve["actual_ms"] != now
            or curve["canonical_close_ms"] != frame["canonical_close_ms"]
            or curve["positions"] != len(inventory)
        ):
            raise ValueError("权益曲线时钟或库存数量错误")
        equal(curve["cash"], cash, "逐帧现金")
        if equity is None:
            if curve["equity"] is not None:
                raise ValueError("缺失估值不能补成已知权益")
        else:
            equal(curve["equity"], equity, "逐帧含浮亏权益")
            peak = max(peak, equity)
            drawdown = max(drawdown, 1 - equity / peak)
        previous_decision = now
        previous_closing = frame["canonical_close_ms"]
    if len(checkpoint["curve"]) != completed:
        raise ValueError("权益曲线与完整输入帧数量不一致")
    if (
        cursor != len(fills)
        or checkpoint["event_count"] != completed
        or checkpoint["event_head"] != head
    ):
        raise ValueError("成交或输入校验链前缀不完整")
    if payments != checkpoint["payments"]:
        raise ValueError("资金费付款账本不完整或被修改")
    if not required_errors <= {v["reason"] for v in checkpoint["quality_errors"]}:
        raise ValueError("财务时序或数据质量失败被删除")
    equal(checkpoint["cash"], cash, "最终现金")
    equal(checkpoint["max_dd"], drawdown, "含浮亏回撤")
    closed = {p["cycle_id"]: p for p in cycles.values() if "exit_ms" in p}
    if set(closed) != {t["cycle_id"] for t in checkpoint["trades"]} or set(inventory) != set(
        checkpoint["positions"]
    ):
        raise ValueError("期末库存或完整交易集合不一致")
    for trade in checkpoint["trades"]:
        p = closed[trade["cycle_id"]]
        net = p["gross_pnl"] - p["entry_fee"] - p["exit_fee"] + p["funding"]
        equal(trade["net_pnl"], net, "完整交易净损益")
        equal(trade["funding"], p["funding"], "逐交易资金费")
    return {
        "frames_verified": completed,
        "fills_verified": cursor,
        "funding_payments_verified": len(payments),
        "closed_trades_verified": len(closed),
        "cash": str(cash),
        "maximum_observed_drawdown": str(drawdown),
        "real_fills_proven": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
    }
