"""从封存原文独立重算观察器覆盖；不调用采集器或其消息解析函数。"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from pathlib import Path

from tradingagents.dataflows.research.live_capture import body, digest
from tradingagents.research.quant.backtesting.paper_store import entries
from tradingagents.research.quant.backtesting.raw_chain import raw_entries
from tradingagents.research.quant.backtesting.segmented_chain import segments
from tradingagents.research.quant.strategy_audit import read, save, sha

ROUTES = {
    "quote": ("wss://fstream.binance.com/public/stream", "@bookTicker", "bookTicker"),
    "aggregate_trade": ("wss://fstream.binance.com/market/stream", "@aggTrade", "aggTrade"),
}


def relative_source(value):
    """源码档案保留包内完整路径，不能仅凭同名文件通过身份核验。"""
    parts = Path(value).parts
    namespace = "tradingagents" if "tradingagents" in parts else "cz"
    if namespace not in parts:
        raise ValueError("观察器源码不在登记研究包内")
    return Path(*parts[parts.index(namespace) :])


def payload(value, envelope, role, symbols, received):
    """独立字段检查区分空盘口和可用盘口，普通成交量允许为零。"""
    if not isinstance(value, dict) or value.get("s") not in symbols:
        raise ValueError("行情不属于预定全集")
    if "st" in value and (type(value["st"]) is not int or value["st"] != 1):
        raise ValueError("行情不是USDT合约")
    if (
        envelope.get("stream") != value["s"].lower() + ROUTES[role][1]
        or value.get("e") != ROUTES[role][2]
    ):
        raise ValueError("流名、行情与路由不一致")
    if (
        any(type(value.get(k)) is not int for k in ("T", "E"))
        or not 0 <= value["T"] <= value["E"] <= received + 1000
    ):
        raise ValueError("行情事件时间不合法")
    fields = ("b", "a", "B", "A") if role == "quote" else ("p", "q")
    numbers = [Decimal(value[key]) for key in fields]
    if any(not n.is_finite() or n < 0 for n in numbers):
        raise ValueError("行情价格或数量非有限或为负")
    if role == "quote":
        if (
            type(value.get("u")) is not int
            or value["u"] < 0
            or (min(numbers[:2]) > 0 and numbers[0] > numbers[1])
        ):
            raise ValueError("盘口更新序号或方向不合法")
        return value["s"], min(numbers) > 0
    if (
        min(numbers) <= 0
        or any(type(value.get(k)) is not int or value[k] < 0 for k in ("a", "f", "l"))
        or value["f"] > value["l"]
        or type(value.get("m")) is not bool
    ):
        raise ValueError("聚合成交身份或数量不合法")
    if "nq" in value:
        normal = Decimal(value["nq"])
        if not normal.is_finite() or not 0 <= normal <= numbers[1]:
            raise ValueError("聚合成交普通数量不守恒")
    return value["s"], True


def audit(root):
    """验证终态、全部字节与接收包装，输出覆盖和未知而不输出账户收益。"""
    root = Path(root)
    receipts = read(root / "receipts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"receipts.json"}
    if set(receipts) != actual:
        raise ValueError("观察器封存文件集合发生变化")
    for name, expected in receipts.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("观察器封存文件身份发生变化")
    protocol = read(root / "protocol.json")
    prefix_only = (root / "prefix_snapshot.json").exists()
    result = read(root / ("prefix_snapshot.json" if prefix_only else "result.json"))
    if prefix_only and (
        result["schema"] != "cz-public-research-prefix-snapshot.v1"
        or result["kind"] != "observer"
        or result["protocol_sha256"] != sha(root / "protocol.json")
        or result["original_study_terminally_sealed"] is not False
        or result["capture_end_reached"] is not False
    ):
        raise ValueError("观察器前缀被误标为完整终态")
    symbols = protocol["symbols"]
    if (
        protocol["schema"] != "cz-public-microstructure-observer.v1"
        or len(symbols) != 220
        or symbols != sorted(set(symbols))
        or protocol["source_routes"] != {k: list(v[:2]) for k, v in ROUTES.items()}
        or any(
            protocol.get(k) is not False
            for k in (
                "account_state_read_or_written",
                "can_emit_trade_signals",
                "production_eligible",
                "orders_enabled",
            )
        )
    ):
        raise ValueError("观察器协议或权限发生变化")
    source = {
        str(relative_source(p)): expected for p, expected in protocol["source_sha256"].items()
    }
    for name, expected in source.items():
        if sha(root / "source" / name) != expected:
            raise ValueError("观察器源码档案不同")
    plan_sha = protocol.get("predeclared_plan_sha256")
    if plan_sha is not None and plan_sha != (
        source.get("tradingagents/research/quant/plans/profit_factor_microstructure_plan.md")
        or source.get("cz/research/profit_factor_microstructure_plan.md")
    ):
        raise ValueError("观察器预先声明计划与档案不同")
    for function in (body, digest, entries, segments, raw_entries, read, sha):
        path = Path(function.__code__.co_filename).resolve()
        if sha(path) != source.get(str(relative_source(path))):
            raise ValueError("已加载原文字节核验器与观察器冻结源码不同")
    totals = {role: Counter() for role in ROUTES}
    observed = {role: set() for role in ROUTES}
    ages = {role: Counter() for role in ROUTES}
    epochs, errors, clock, clock_sha = {}, [], None, None
    clock_intervals, maximum_received = [], 0
    count, head = 0, None
    for number, record, current_head in raw_entries(root / "public_events"):
        head = current_head
        count = number + 1
        kind, received = record["kind"], record.get("received_ms")
        if type(received) is not int or received < 0:
            raise ValueError("观察器事件缺少实际接收时间")
        maximum_received = max(maximum_received, received)
        if kind == "clock":
            if record.get("credentials_used") is not False:
                raise ValueError("观察器时钟出现认证权限")
            try:
                server = body(record).get("serverTime") if record.get("status") == 200 else None
                good = (
                    record["local_clock_consistent"]
                    and type(server) is int
                    and record["requested_ms"] - 1000 <= server <= received + 1000
                )
            except (ValueError, AttributeError, KeyError):
                good = False
            if good:
                if clock is None or maximum_received > clock["received_ms"] + 120_000:
                    clock_intervals.append(
                        {"known_from_ms": maximum_received, "known_through_ms": received + 120_000}
                    )
                else:
                    clock_intervals[-1]["known_through_ms"] = max(
                        clock_intervals[-1]["known_through_ms"], received + 120_000
                    )
                clock, clock_sha = record, digest(record)
            continue
        if kind == "observer_clock_unknown":
            if clock is not None:
                clock_intervals[-1]["known_through_ms"] = min(
                    clock_intervals[-1]["known_through_ms"], received - 1
                )
            clock = None
            continue
        if kind in {"public_clock_retry", "observer_failure", "observer_stop_requested"}:
            continue
        role, generation = record.get("source_role"), record.get("generation")
        if role not in ROUTES or not isinstance(generation, str) or not generation:
            raise ValueError("观察器事件缺少来源与物理连接")
        key = role, generation
        if kind == "observer_connection_started":
            if key in epochs or record["url"] != ROUTES[role][0]:
                raise ValueError("物理连接身份重复或端点不同")
            epochs[key] = {
                "started_ms": received,
                "confirmed_ms": None,
                "end_ms": None,
                "requests": {},
                "awaiting": set(),
                "acknowledged": [],
                "pings": set(),
                "observed_through_ms": received,
            }
            totals[role]["connections"] += 1
            continue
        epoch = epochs.get(key)
        if epoch is None or epoch["end_ms"] is not None or received < epoch["started_ms"]:
            raise ValueError("事件不在已知物理连接内")
        if (
            kind in {"observer_message", "observer_pong_ack"}
            and received < protocol["capture_end_ms"]
        ):
            epoch["observed_through_ms"] = max(epoch["observed_through_ms"], received)
        if kind == "observer_subscription_request":
            request = record["request"]
            identity = request["id"]
            if (
                identity in epoch["requests"]
                or request["method"] != "SUBSCRIBE"
                or not 1 <= len(request["params"]) <= 100
            ):
                raise ValueError("订阅请求重复或超出批量范围")
            epoch["requests"][identity] = request["params"]
            epoch["awaiting"].add(identity)
        elif kind == "observer_subscription_confirmed":
            requested = [p for values in epoch["requests"].values() for p in values]
            expected = [s.lower() + ROUTES[role][1] for s in symbols]
            if (
                epoch["awaiting"]
                or epoch["confirmed_ms"] is not None
                or requested != expected
                or record["members"] != 220
            ):
                raise ValueError("全集订阅确认缺少完整唯一回应")
            epoch["confirmed_ms"] = received
        elif kind == "observer_connection_ended":
            epoch["end_ms"] = received
            if record["registered_end_reached"] != (received >= protocol["capture_end_ms"]):
                raise ValueError("连接终点声明与真实接收时刻不同")
            totals[role]["connection_errors"] += record["error"] is not None
        elif kind == "observer_ping_sent":
            if received in epoch["pings"]:
                raise ValueError("主动PING身份重复")
            epoch["pings"].add(received)
        elif kind == "observer_pong_ack":
            if record["sent_ms"] not in epoch["pings"] or record["sent_ms"] > received:
                raise ValueError("PONG没有对应的先前PING")
            epoch["pings"].remove(record["sent_ms"])
            totals[role]["pong_acks"] += 1
        elif kind == "observer_payload_rejected":
            totals[role]["rejection_records"] += 1
        elif kind == "observer_message":
            expected_clock = clock is not None and 0 <= received - clock["received_ms"] <= 120_000
            if (
                type(record["clock_verified"]) is not bool
                or record["clock_verified"] != expected_clock
                or record["clock_record_sha256"] != clock_sha
                or record["subscription_confirmed_ms"] != epoch["confirmed_ms"]
                or record["after_capture_end"] != (received >= protocol["capture_end_ms"])
            ):
                raise ValueError("原文接收包装被后来时钟、订阅或截止时间改写")
            if record["after_capture_end"]:
                totals[role]["after_end_messages"] += 1
                continue
            try:
                envelope = body(record)
                if isinstance(envelope, dict) and "id" in envelope:
                    identity = envelope["id"]
                    if (
                        envelope.get("result", "missing") is not None
                        or identity not in epoch["awaiting"]
                    ):
                        raise ValueError("订阅回应失败或重复")
                    epoch["awaiting"].remove(identity)
                    epoch["acknowledged"].append(identity)
                    continue
                if not isinstance(envelope, dict):
                    raise ValueError("市场原文不是组合消息")
                symbol, ready = payload(envelope.get("data"), envelope, role, symbols, received)
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                errors.append(
                    {"record_sha256": digest(record), "source_role": role, "error": str(exc)}
                )
                continue
            totals[role]["market_messages"] += 1
            totals[role]["clock_verified_messages"] += record["clock_verified"]
            totals[role]["ready_messages"] += ready
            totals[role]["before_subscription_confirmation"] += epoch["confirmed_ms"] is None
            observed[role].add(symbol)
            delay = received - envelope["data"]["E"]
            ages[role]["up_to_5s" if delay <= 5000 else "over_5s"] += 1
        else:
            raise ValueError("观察器出现未登记事件类型")
    if result["source_prefix"] != {"event_count": count, "event_chain_sha256": head}:
        raise ValueError("终态原文前缀与完整日志不同")
    if prefix_only and result["maximum_record_received_ms"] != maximum_received:
        raise ValueError("前缀实际接收水位与完整字节不同")
    if not prefix_only and any(e["end_ms"] is None for e in epochs.values()):
        raise ValueError("终态缺少物理连接关闭记录")
    for role in ROUTES:
        if prefix_only:
            continue
        value = result.get(role, {})
        for field in (
            "connections",
            "market_messages",
            "clock_verified_messages",
            "ready_messages",
            "pong_acks",
            "after_end_messages",
        ):
            if value.get(field, 0) != totals[role][field]:
                raise ValueError("观察器计数与独立原文重建不同")
        if value.get("observed_symbols", []) != sorted(observed[role]):
            raise ValueError("观察器合约覆盖与原文不同")
    if not prefix_only and any(
        result.get(k) is not False
        for k in (
            "strategy_profitability_evaluated",
            "account_state_read_or_written",
            "orders_enabled",
            "production_eligible",
        )
    ):
        raise ValueError("诊断终态越过账户或生产权限")
    if errors and not prefix_only and result["failure"] is None:
        raise ValueError("存在非法原文却声明无失败")
    return {
        "schema": "cz-public-microstructure-observer-audit.v1",
        "source_prefix": result["source_prefix"],
        "receipt_sha256": sha(root / "receipts.json"),
        "predeclared_plan_hash_bound": plan_sha is not None,
        "raw_bytes_verified": True,
        "prefix_only": prefix_only,
        "clock_intervals": [
            {
                **window,
                "known_through_ms": min(
                    window["known_through_ms"], maximum_received, protocol["capture_end_ms"]
                ),
            }
            for window in clock_intervals
            if window["known_from_ms"]
            <= min(window["known_through_ms"], maximum_received, protocol["capture_end_ms"])
        ],
        "capture_end_reached": result["capture_end_reached"],
        "failure": result.get("failure"),
        "payload_errors": errors,
        "sources": {
            role: {
                **dict(totals[role]),
                "observed_members": len(observed[role]),
                "unobserved_members": sorted(set(symbols) - observed[role]),
                "receive_delay_groups": dict(ages[role]),
            }
            for role in ROUTES
        },
        "connection_epochs": [
            {
                "source_role": role,
                "generation": generation,
                **{
                    k: v
                    for k, v in epoch.items()
                    if k in {"started_ms", "confirmed_ms", "end_ms", "observed_through_ms"}
                },
            }
            for (role, generation), epoch in epochs.items()
        ],
        "strategy_profitability_evaluated": False,
        "orders_enabled": False,
        "production_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.capture.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "result.json", result)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(
        args.output / "receipts.json",
        {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    )
    print(
        {
            k: v
            for k, v in result.items()
            if k not in {"connection_epochs", "sources", "payload_errors"}
        }
    )


if __name__ == "__main__":
    main()
