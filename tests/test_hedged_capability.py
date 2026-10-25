"""真实两腿能力的语义及实际Parquet拒绝边界；小样本仅为联调。"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tradingagents.research import hedged_capability as module
from tradingagents.research.hedged_capability import (
    HedgedSourceCapability,
    verified_hedged_capabilities,
)
from tradingagents.research.quant.artifacts import read, sha


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def reseal(root, audit):
    path = root / "receipts.json"
    write(path, {p.name: sha(p) for p in root.iterdir() if p.is_file() and p != path})
    record = read(audit / "result.json")
    record["source_panel_receipts_sha256"] = sha(path)
    write(audit / "result.json", record)
    write(audit / "receipts.json", {"result.json": sha(audit / "result.json")})
    return HedgedSourceCapability(
        schema_version="ta-hedged-source-capability.v1",
        panel_root=str(root),
        panel_receipts_sha256=sha(path),
        audit_root=str(audit),
        audit_receipts_sha256=sha(audit / "receipts.json"),
    )


@pytest.fixture
def pair(tmp_path, monkeypatch):
    # 只缩小合成日历，实际声明固定61日及175680行，不消费真实数据。
    start = module.START
    monkeypatch.setattr(module, "END", start + 120000)
    monkeypatch.setattr(module, "ROWS", 4)
    root, audit = tmp_path / "panel", tmp_path / "audit"
    root.mkdir()
    audit.mkdir()
    flags = dict.fromkeys(
        [
            "basis_generated",
            "labels_generated",
            "returns_evaluated",
            "orders_enabled",
            "production_eligible",
        ],
        False,
    )
    write(
        root / "protocol.json",
        {
            **flags,
            "schema": "ta-hedged-major-source-panel.v1",
            "symbols": list(module.SYMBOLS),
            "start_raw_open_ms": start,
            "end_raw_open_exclusive_ms": start + 120000,
            "expected_rows": 4,
            "spot_time_unit": "microseconds",
            "perp_time_unit": "milliseconds",
            "funding_times_rounded": False,
            "leader_labels_read": False,
        },
    )
    write(
        root / "result.json",
        {
            **flags,
            "status": "COMPLETED_SOURCE_PANEL_ONLY",
            "symbols": 2,
            "rows": 4,
            "both_known": 4,
            "funding_events": 2,
            "funding_types": {"Regular": 2},
            "off_whole_minute_funding_events": 2,
        },
    )
    write(root / "coverage.json", {"rows": 4, "both_known": 4})
    rows = []
    for symbol in module.SYMBOLS:
        for clock in (start, start + 60000):
            rows.append(
                {
                    "symbol": symbol,
                    "raw_open_ms": clock,
                    "completed_ms": clock + 60000,
                    "spot_status": "KNOWN_RAW_MINUTE",
                    "perp_status": "KNOWN_RAW_MINUTE",
                    **dict.fromkeys(module.ALIASES.values(), "100.0000"),
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), root / "minutes.parquet")
    write(
        root / "funding_events.json",
        [
            {"symbol": s, "time_ms": start + 1, "rate": "0.00001", "mark": "100", "type": "Regular"}
            for s in module.SYMBOLS
        ],
    )
    write(
        root / "funding_coverage.json",
        [
            {"symbol": s, "month": m, "month_ready_under_source_assumptions": True}
            for s in module.SYMBOLS
            for m in ("2026-08", "2026-09")
        ],
    )
    write(
        audit / "result.json",
        {
            **flags,
            "status": "VERIFIED_FULL_TWO_MARKET_RAW_FIELDS_AND_SETTLEMENTS",
            "rows": 4,
            "raw_market_zip_count": 8,
            "funding_events": 2,
        },
    )
    return root, audit, reseal(root, audit)


def test_actual_footer_calendar_and_summary_projection(pair):
    root, audit, capability = pair
    summary = verified_hedged_capabilities(capability)
    assert summary["rows"] == 4 and summary["unknown_pair_minutes"] == 0
    assert summary["input_aliases"]["spot_minute_close"] == "spot_close"
    assert str(root) not in json.dumps(summary) and str(audit) not in json.dumps(summary)
    assert summary["historical_first_publication_proven"] is False
    assert summary["off_whole_minute_funding_events"] == 2
    assert (
        summary["accounting_only_input_aliases"]["settled_cash_event_mark"] == "funding_events.mark"
    )
    assert summary["accounting_inputs_are_not_predictor_features"] is True
    assert "settled_cash_event_mark" in summary["declared_inputs"]


@pytest.mark.parametrize(
    "damage",
    [
        "units",
        "labels",
        "rounded",
        "funding_type",
        "funding_count",
        "raw_clock",
        "duplicate",
        "float_price",
        "wrong_audit",
        "coverage",
    ],
)
def test_self_consistent_receipts_do_not_bypass_semantic_checks(pair, damage):
    root, audit, _ = pair
    if damage in {"units", "labels", "rounded"}:
        value = read(root / "protocol.json")
        key, update = {
            "units": ("spot_time_unit", "milliseconds"),
            "labels": ("labels_generated", True),
            "rounded": ("funding_times_rounded", True),
        }[damage]
        value[key] = update
        write(root / "protocol.json", value)
    elif damage in {"funding_type", "funding_count"}:
        value = read(root / "funding_events.json")
        if damage == "funding_type":
            value[0]["type"] = "Other"
        else:
            value[0]["time_ms"] -= 1
        write(root / "funding_events.json", value)
    elif damage in {"raw_clock", "duplicate", "float_price"}:
        table = pq.read_table(root / "minutes.parquet").to_pandas()
        if damage == "raw_clock":
            table.loc[0, "completed_ms"] += 1
        elif damage == "duplicate":
            table.loc[1, "raw_open_ms"] = table.loc[0, "raw_open_ms"]
        else:
            table["spot_open"] = table.spot_open.astype(float)
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), root / "minutes.parquet")
    elif damage == "wrong_audit":
        value = read(audit / "result.json")
        value["status"] = "FAILED"
        write(audit / "result.json", value)
    else:
        value = read(root / "coverage.json")
        value["both_known"] = 3
        write(root / "coverage.json", value)
    capability = reseal(root, audit)
    with pytest.raises(ValueError):
        verified_hedged_capabilities(capability)


def test_byte_tampering_rejected_before_summary(pair):
    root, _, capability = pair
    (root / "coverage.json").write_text("{}")
    with pytest.raises(ValueError):
        verified_hedged_capabilities(capability)
