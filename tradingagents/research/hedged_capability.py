"""真实两腿来源的受限能力声明，不以来源就绪授予盈利资格。"""

from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import Field

from .contracts import StrictModel
from .quant.artifacts import read, sha, verify_receipts

START, END, MINUTE = 1785542400000, 1790812800000, 60000
ROWS = 175680
SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALIASES = {
    "minute_open": "perp_open",
    "minute_close": "perp_close",
    "minute_quote_volume": "perp_quote_volume",
    "minute_taker_buy_quote": "perp_taker_buy_quote",
    "spot_minute_open": "spot_open",
    "spot_minute_close": "spot_close",
    "spot_minute_quote_volume": "spot_quote_volume",
    "spot_minute_taker_buy_quote": "spot_taker_buy_quote",
}
FUNDING_INPUTS = ("settled_regular_funding_rate", "settled_regular_funding_time")
ACCOUNTING_INPUTS = {
    "settled_cash_event_rate": "funding_events.rate",
    "settled_cash_event_time": "funding_events.time_ms",
    "settled_cash_event_mark": "funding_events.mark",
    "settled_cash_event_type": "funding_events.type",
}


class HedgedSourceCapability(StrictModel):
    """仅接受已连接并独立审计的真实两腿来源，路径不进入模型上下文。"""

    schema_version: Literal["ta-hedged-source-capability.v1"]
    panel_root: str = Field(min_length=1, max_length=3000)
    panel_receipts_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audit_root: str = Field(min_length=1, max_length=3000)
    audit_receipts_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _root(name, expected):
    root = Path(name)
    if not root.is_absolute() or root.is_symlink() or str(root.resolve()) != str(root):
        raise ValueError("两腿能力必须绑定规范绝对本地目录")
    if sha(root / "receipts.json") != expected:
        raise ValueError("两腿能力收据身份不符")
    verify_receipts(root)
    if (root / "failure.json").exists():
        raise ValueError("失败来源或失败审计不能开放两腿输入")
    return root


def verified_hedged_capabilities(capability):
    """核完整收据、实际全日历和结算类型，返回无路径的固定投影。"""
    import pyarrow.parquet as pq

    root = _root(capability.panel_root, capability.panel_receipts_sha256)
    audit_root = _root(capability.audit_root, capability.audit_receipts_sha256)
    protocol, result, coverage = (
        read(root / n) for n in ("protocol.json", "result.json", "coverage.json")
    )
    audit = read(audit_root / "result.json")
    if (
        protocol.get("schema") != "ta-hedged-major-source-panel.v1"
        or protocol.get("symbols") != list(SYMBOLS)
        or protocol.get("start_raw_open_ms") != START
        or protocol.get("end_raw_open_exclusive_ms") != END
        or protocol.get("expected_rows") != ROWS
        or protocol.get("spot_time_unit") != "microseconds"
        or protocol.get("perp_time_unit") != "milliseconds"
        or protocol.get("funding_times_rounded") is not False
        or result.get("status") != "COMPLETED_SOURCE_PANEL_ONLY"
        or result.get("symbols") != len(SYMBOLS)
        or result.get("rows") != ROWS
        or coverage.get("rows") != ROWS
        or audit.get("status") != "VERIFIED_FULL_TWO_MARKET_RAW_FIELDS_AND_SETTLEMENTS"
        or audit.get("source_panel_receipts_sha256") != capability.panel_receipts_sha256
        or audit.get("rows") != ROWS
        or audit.get("raw_market_zip_count") != 8
        or any(
            data.get(flag) is not False
            for data in (protocol, result, audit)
            for flag in (
                "basis_generated",
                "labels_generated",
                "returns_evaluated",
                "orders_enabled",
                "production_eligible",
            )
        )
        or protocol.get("leader_labels_read") is not False
    ):
        raise ValueError("两腿来源范围、单位、来源纯度或独立审计不兼容")
    parquet = pq.ParquetFile(root / "minutes.parquet")
    required = {
        "symbol": "string",
        "raw_open_ms": "int64",
        "completed_ms": "int64",
        **dict.fromkeys(ALIASES.values(), "string"),
        "spot_status": "string",
        "perp_status": "string",
    }
    types = {f.name: str(f.type) for f in parquet.schema_arrow}
    if parquet.metadata.num_rows != ROWS or any(types.get(k) != v for k, v in required.items()):
        raise ValueError("两腿实际Parquet缺列、类型或行数不合法")
    frame = pq.read_table(
        root / "minutes.parquet",
        columns=["symbol", "raw_open_ms", "completed_ms", "spot_status", "perp_status"],
    ).to_pandas()
    if (
        frame.duplicated(["symbol", "raw_open_ms"]).any()
        or set(frame.symbol) != set(SYMBOLS)
        or not frame.completed_ms.eq(frame.raw_open_ms + MINUTE).all()
    ):
        raise ValueError("两腿实际身份、完成钟或人口不合法")
    for _, group in frame.groupby("symbol", sort=True):
        if group.raw_open_ms.tolist() != list(range(START, END, MINUTE)):
            raise ValueError("两腿实际完整分钟日历不符")
    known = frame.spot_status.eq("KNOWN_RAW_MINUTE") & frame.perp_status.eq("KNOWN_RAW_MINUTE")
    if (
        result.get("both_known") != int(known.sum())
        or coverage.get("both_known") != int(known.sum())
        or not known.any()
    ):
        raise ValueError("两腿共同覆盖或未知计数不守恒")
    events = read(root / "funding_events.json")
    funding_coverage = read(root / "funding_coverage.json")
    if len(events) != result.get("funding_events") or len(events) != audit.get("funding_events"):
        raise ValueError("两腿结算事件数量不一致")
    identities = set()
    for event in events:
        identity = (event["symbol"], event["time_ms"], event["type"])
        if (
            identity in identities
            or event["symbol"] not in SYMBOLS
            or type(event["time_ms"]) is not int
            or not START <= event["time_ms"] < END
            or event["type"] not in {"Regular", "Special"}
        ):
            raise ValueError("结算身份、原毫秒或类型不合法")
        identities.add(identity)
        try:
            rate, mark = Decimal(event["rate"]), Decimal(event["mark"])
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("结算原值不可解释") from error
        if not rate.is_finite() or not mark.is_finite() or mark <= 0:
            raise ValueError("结算原值非有限或mark非正")
    if (
        len(funding_coverage) != 4
        or {(r["symbol"], r["month"]) for r in funding_coverage}
        != {(s, m) for s in SYMBOLS for m in ("2026-08", "2026-09")}
        or any(r.get("month_ready_under_source_assumptions") is not True for r in funding_coverage)
    ):
        raise ValueError("两腿资金费四币月未就绪")
    types = dict(Counter(e["type"] for e in events))
    off_minute = sum(e["time_ms"] % MINUTE != 0 for e in events)
    if (
        types != result.get("funding_types")
        or off_minute != result.get("off_whole_minute_funding_events")
        or {e["symbol"] for e in events} != set(SYMBOLS)
    ):
        raise ValueError("两腿结算类型、毫秒偏移或符号计数不一致")
    return {
        "schema_version": "ta-verified-hedged-source-capabilities.v2",
        "panel_receipts_sha256": capability.panel_receipts_sha256,
        "audit_receipts_sha256": capability.audit_receipts_sha256,
        "symbols": list(SYMBOLS),
        "rows": ROWS,
        "start_raw_open_ms": START,
        "end_raw_open_exclusive_ms": END,
        "both_known": int(known.sum()),
        "unknown_pair_minutes": ROWS - int(known.sum()),
        "input_aliases": dict(ALIASES),
        "declared_inputs": [*ALIASES, *FUNDING_INPUTS, *ACCOUNTING_INPUTS],
        "accounting_only_input_aliases": dict(ACCOUNTING_INPUTS),
        "accounting_inputs_are_not_predictor_features": True,
        "funding_events": len(events),
        "funding_types": types,
        "off_whole_minute_funding_events": off_minute,
        "completed_bar_visibility_assumption": True,
        "settled_rate_visibility_delay_ms_assumption": 300000,
        "funding_cash_requires_actual_inventory_at_original_event_ms": True,
        "funding_complete_under_source_assumptions_only": True,
        "historical_first_publication_proven": False,
        "historical_units_and_tradability_proven": False,
        "basis_generated": False,
        "labels_generated": False,
        "returns_evaluated": False,
        "orders_enabled": False,
        "scope": "真实现货与永续两腿来源，不是指数代理；论文先验固定BTC/ETH，不删除旧748/909研究。",
    }
