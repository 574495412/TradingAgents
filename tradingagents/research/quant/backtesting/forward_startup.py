"""选择已真实收到的完整共同分钟，协调推送与历史预热的交接。"""

from tradingagents.dataflows.research.live_capture import MINUTE, available_rows
from tradingagents.research.quant.backtesting.forward_inputs import normalized_row


def active_members(contracts, closing):
    """只根据已知合约身份确定可交易成员，不能按下载成败缩减。"""
    return [
        s
        for s, c in contracts.items()
        if c["trading"]
        and c["onboard_ms"] <= closing - MINUTE
        and (not c["delivery_ms"] or closing - MINUTE < c["delivery_ms"])
    ]


def common_stream_close(bars, contracts, *, now_ms):
    """部分订阅跨过边界时不能提前选定预热结束点，必须等共同终值齐全。"""
    latest = (now_ms - 1000) // MINUTE * MINUTE
    for closing in range(latest, latest - 3 * MINUTE, -MINUTE):
        members = active_members(contracts, closing)
        if members and all(bars.at(s, closing, now_ms) is not None for s in members):
            return closing, members
    return None, []


def merge_history(bars, records, *, closing, now_ms):
    """跨通道不同终值暂不采用；一致后仍保留首次接收的真实价格与时间。"""
    overlap, conflicts, confirmed = set(), [], set()
    for record in records:
        for row in available_rows(
            record, decision_ms=now_ms, cutoff_ms=closing, clock_verified=True
        ):
            symbol, clock = record["symbol"], row[6] + 1
            previous = bars.at(symbol, clock, now_ms)
            if previous and previous["normalized"] != normalized_row(row):
                conflicts.append(
                    {
                        "symbol": symbol,
                        "closing_ms": clock,
                        "first_source_sha256": previous["source_sha256"],
                        "conflicting_source_sha256": record["body_sha256"],
                    }
                )
                continue
            if previous and clock == closing:
                overlap.add(symbol)
            if previous:
                confirmed.add((symbol, clock))
            bars.add(
                symbol,
                row,
                received_ms=record["received_ms"],
                source_sha256=record["body_sha256"],
                completed_verified=True,
            )
    return {"overlap": overlap, "conflicts": conflicts, "confirmed_keys": confirmed}


def require_confirmed_conflicts(conflicts, confirmation):
    """复核必须覆盖每个冲突分钟，空响应或仅最新一分钟相同都不能通过。"""
    required = {(v["symbol"], v["closing_ms"]) for v in conflicts}
    if confirmation["conflicts"] or not required.issubset(confirmation["confirmed_keys"]):
        raise ValueError("公开终值冲突未得到逐分钟精确复核")
