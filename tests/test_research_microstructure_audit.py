"""原文字节、全集确认和接收包装必须分别通过独立检查。"""

from pathlib import Path

import pytest

from tradingagents.dataflows.research.live_capture import canonical
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting import microstructure_audit as module
from tradingagents.research.quant.backtesting.microstructure import parse_message
from tradingagents.research.quant.backtesting.segmented_chain import SegmentedChain


def fixture(root, failure=None, *, prefix=False, generation_prefix="会话"):
    symbols = [f"A{k:03}USDT" for k in range(220)]
    root.mkdir()
    files = {
        Path(f.__code__.co_filename).resolve()
        for f in (
            module.body,
            module.digest,
            module.entries,
            module.segments,
            module.raw_entries,
            module.read,
            module.sha,
            parse_message,
        )
    }
    protocol = {
        "schema": "cz-public-microstructure-observer.v1",
        "symbols": symbols,
        "capture_start_ms": 90_000,
        "capture_end_ms": 120_000,
        "source_routes": {k: list(v[:2]) for k, v in module.ROUTES.items()},
        "source_sha256": {str(p): module.sha(p) for p in files},
        "linked_study_protocol_sha256": "关联研究",
        "account_state_read_or_written": False,
        "can_emit_trade_signals": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    module.save(root / "protocol.json", protocol)
    for file in files:
        target = root / "source" / module.relative_source(file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    clock = received_record(canonical({"serverTime": 99_995}), received_ms=100_000) | {
        "kind": "clock",
        "status": 200,
        "requested_ms": 99_990,
        "local_clock_consistent": True,
        "credentials_used": False,
    }
    records, result = (
        [clock],
        {
            "failure": None,
            "capture_end_reached": True,
            "stopped_ms": 120_002,
            "strategy_profitability_evaluated": False,
            "account_state_read_or_written": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    for role in module.ROUTES:
        generation = generation_prefix + ":" + role
        records.append(
            {
                "kind": "observer_connection_started",
                "received_ms": 100_100,
                "source_role": role,
                "generation": generation,
                "url": module.ROUTES[role][0],
            }
        )
        streams = [s.lower() + module.ROUTES[role][1] for s in symbols]
        if failure == "missing_member" and role == "quote":
            streams[-1] = streams[0]
        records.append(
            {
                "kind": "observer_subscription_request",
                "received_ms": 100_150,
                "source_role": role,
                "generation": generation,
                "request": {"method": "SUBSCRIBE", "id": "回应", "params": streams[:100]},
            }
        )
        for part in (1, 2):
            records.append(
                {
                    "kind": "observer_subscription_request",
                    "received_ms": 100_150,
                    "source_role": role,
                    "generation": generation,
                    "request": {
                        "method": "SUBSCRIBE",
                        "id": "回应" + str(part),
                        "params": streams[part * 100 : (part + 1) * 100],
                    },
                }
            )
        for identity in ("回应", "回应1", "回应2"):
            record = received_record(
                canonical({"result": None, "id": identity}), received_ms=100_200
            )
            record.update(
                kind="observer_message",
                source_role=role,
                generation=generation,
                subscription_confirmed_ms=None,
                clock_verified=True,
                clock_record_sha256=module.digest(clock),
                after_capture_end=False,
            )
            records.append(record)
        records.append(
            {
                "kind": "observer_subscription_confirmed",
                "received_ms": 100_300,
                "source_role": role,
                "generation": generation,
                "members": 220,
            }
        )
        value = {"s": symbols[0], "st": 1, "E": 100_800, "T": 100_790}
        if role == "quote":
            value.update(e="bookTicker", u=1, b="99", a="101", B="2", A="2")
        else:
            value.update(e="aggTrade", a=1, f=1, l=1, p="100", q="2", nq="0", m=False)
        if failure == "future_event" and role == "quote":
            value["E"] = 102_000
        record = received_record(
            canonical({"stream": streams[0], "data": value}), received_ms=100_900
        )
        record.update(
            kind="observer_message",
            source_role=role,
            generation=generation,
            subscription_confirmed_ms=100_300,
            clock_verified=True,
            clock_record_sha256=module.digest(clock),
            after_capture_end=False,
        )
        if failure == "clock_rewrite" and role == "quote":
            record["clock_record_sha256"] = "后来替换的时钟"
        records.append(record)
        result[role] = {
            "connections": 1,
            "market_messages": 1,
            "clock_verified_messages": 1,
            "ready_messages": 1,
            "pong_acks": 0,
            "after_end_messages": 0,
            "observed_symbols": [symbols[0]],
        }
        if role == "quote" and failure == "latest_empty":
            value.update(T=100_900, E=100_950, u=2, b="0", B="0")
            empty = received_record(
                canonical({"stream": streams[0], "data": value}), received_ms=101_000
            )
            empty.update(
                kind="observer_message",
                source_role=role,
                generation=generation,
                subscription_confirmed_ms=100_300,
                clock_verified=True,
                clock_record_sha256=module.digest(clock),
                after_capture_end=False,
            )
            records.append(empty)
            result[role]["market_messages"] += 1
            result[role]["clock_verified_messages"] += 1
    if failure == "clock_gap":
        records.append(
            {"kind": "observer_clock_unknown", "received_ms": 110_000, "error": "人工时钟失败"}
        )
        recovered = received_record(canonical({"serverTime": 111_000}), received_ms=111_001) | {
            "kind": "clock",
            "status": 200,
            "requested_ms": 110_990,
            "local_clock_consistent": True,
            "credentials_used": False,
        }
        records.append(recovered)
    if not prefix:
        for role in module.ROUTES:
            records.append(
                {
                    "kind": "observer_connection_ended",
                    "received_ms": 120_001,
                    "source_role": role,
                    "generation": generation_prefix + ":" + role,
                    "error": None,
                    "registered_end_reached": True,
                }
            )
    journal = SegmentedChain(root / "public_events")
    for record in records:
        journal.add(record)
    result["source_prefix"] = journal.seal()
    journal.close()
    if prefix:
        module.save(
            root / "prefix_snapshot.json",
            {
                "schema": "cz-public-research-prefix-snapshot.v1",
                "kind": "observer",
                "protocol_sha256": module.sha(root / "protocol.json"),
                "source_prefix": result["source_prefix"],
                "original_study_terminally_sealed": False,
                "capture_end_reached": False,
                "maximum_record_received_ms": 100_900,
            },
        )
    else:
        module.save(root / "result.json", result)
    module.save(
        root / "receipts.json",
        {str(p.relative_to(root)): module.sha(p) for p in root.rglob("*") if p.is_file()},
    )


def test_receipt_valid_diagnostic_does_not_invent_missing_member_coverage(tmp_path):
    root = tmp_path / "观察"
    fixture(root)
    result = module.audit(root)
    assert result["raw_bytes_verified"] and result["payload_errors"] == []
    assert all(
        v["observed_members"] == 1 and len(v["unobserved_members"]) == 219
        for v in result["sources"].values()
    )
    assert result["strategy_profitability_evaluated"] is False


@pytest.mark.parametrize("failure", ["missing_member", "clock_rewrite", "future_event"])
def test_hash_valid_but_semantically_false_observation_is_rejected(tmp_path, failure):
    root = tmp_path / "观察"
    fixture(root, failure)
    with pytest.raises(ValueError):
        module.audit(root)


def test_live_prefix_cannot_extend_unknown_physical_end_or_clock_into_the_future(tmp_path):
    root = tmp_path / "前缀"
    fixture(root, prefix=True)
    result = module.audit(root)
    assert result["prefix_only"] is True and result["capture_end_reached"] is False
    assert all(
        v["end_ms"] is None and v["observed_through_ms"] == 100_900
        for v in result["connection_epochs"]
    )
    assert result["clock_intervals"] == [{"known_from_ms": 100_000, "known_through_ms": 100_900}]


def test_clock_recovery_starts_a_new_known_interval_without_repairing_the_gap(tmp_path):
    root = tmp_path / "时钟断层"
    fixture(root, "clock_gap")
    result = module.audit(root)
    assert result["clock_intervals"] == [
        {"known_from_ms": 100_000, "known_through_ms": 109_999},
        {"known_from_ms": 111_001, "known_through_ms": 120_000},
    ]
