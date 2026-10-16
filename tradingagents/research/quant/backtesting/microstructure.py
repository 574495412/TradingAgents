"""同期盘口与公开成交印记诊断，不生成策略信号或交易权限。"""

from __future__ import annotations

from decimal import Decimal, localcontext

from tradingagents.dataflows.research.live_capture import body, digest
from tradingagents.research.quant.backtesting.paper_inputs import number

VERSION = "trade-print-diagnostic-v1"
MAXIMUM_PAIR_AGE_MS = 5000


def parse_message(record, symbols):
    """明确区分聚合成交、最优盘口、交易所事件时钟和实际接收时钟。"""
    envelope = body(record)
    if (
        type(record.get("received_ms")) is not int
        or record["received_ms"] < 0
        or type(record.get("clock_verified")) is not bool
        or not isinstance(record.get("generation"), str)
        or not record["generation"]
    ):
        raise ValueError("诊断接收包装缺少明确时钟或连接身份")
    if (
        not isinstance(envelope, dict)
        or "stream" not in envelope
        or not isinstance(envelope.get("data"), dict)
    ):
        raise ValueError("诊断消息缺少组合流身份")
    value = envelope["data"]
    symbol, role = value.get("s"), record["source_role"]
    if symbol not in symbols or (
        "st" in value and (type(value["st"]) is not int or value["st"] != 1)
    ):
        raise ValueError("诊断消息不属于固定USDT合约全集")
    suffix, event = ("@bookTicker", "bookTicker") if role == "quote" else ("@aggTrade", "aggTrade")
    if (
        role not in {"quote", "aggregate_trade"}
        or envelope["stream"] != symbol.lower() + suffix
        or value.get("e") != event
    ):
        raise ValueError("诊断路由、合约与消息类型不同")
    if (
        any(type(value.get(k)) is not int for k in ("E", "T"))
        or not 0 <= value["T"] <= value["E"] <= record["received_ms"] + 1000
    ):
        raise ValueError("诊断事件时钟不合法或晚于接收")
    result = {
        "symbol": symbol,
        "role": role,
        "event_ms": value["E"],
        "transaction_ms": value["T"],
        "received_ms": record["received_ms"],
        "generation": record["generation"],
        "clock_verified": record["clock_verified"],
        "record_sha256": digest(record),
        "body_sha256": record["body_sha256"],
    }
    if role == "quote":
        if type(value.get("u")) is not int or value["u"] < 0:
            raise ValueError("最优盘口更新序号缺失")
        bid, ask, bid_qty, ask_qty = (number(value[k]) for k in ("b", "a", "B", "A"))
        if min(bid, ask, bid_qty, ask_qty) < 0 or (min(bid, ask) > 0 and bid > ask):
            raise ValueError("最优盘口价格、数量或买卖方向异常")
        result.update(
            ready=min(bid, ask, bid_qty, ask_qty) > 0,
            bid=str(bid),
            ask=str(ask),
            bid_qty=str(bid_qty),
            ask_qty=str(ask_qty),
            update_id=value["u"],
        )
    else:
        if (
            any(type(value.get(k)) is not int or value[k] < 0 for k in ("a", "f", "l"))
            or value["f"] > value["l"]
            or type(value.get("m")) is not bool
        ):
            raise ValueError("聚合成交序号或主动方向不合法")
        price, quantity = number(value["p"], positive=True), number(value["q"], positive=True)
        normal = number(value["nq"]) if "nq" in value else None
        if normal is not None and not 0 <= normal <= quantity:
            raise ValueError("聚合成交普通数量不守恒")
        result.update(
            ready=True,
            price=str(price),
            quantity=str(quantity),
            normal_quantity=str(normal) if normal is not None else None,
            aggregate_id=value["a"],
            first_trade_id=value["f"],
            last_trade_id=value["l"],
            buyer_is_maker=value["m"],
            last_all_type_trade_time_proven=False,
        )
    return result


def source_available(value, epoch, data_ms, decision_ms, *, quote):
    """只有端点之前发生、决策之前收到且连接区间覆盖该决策的观测才可配对。"""
    if value is None or epoch is None:
        return "SOURCE_NOT_OBSERVED"
    if not value["clock_verified"] or not value["ready"]:
        return "CLOCK_OR_DISPLAYED_QUOTE_UNKNOWN"
    if (
        value["generation"] != epoch["generation"]
        or not epoch["confirmed_ms"] <= value["received_ms"] <= decision_ms <= epoch["end_ms"]
        or not epoch["confirmed_ms"] <= value["transaction_ms"] <= data_ms
    ):
        return "SOURCE_OUTSIDE_KNOWN_ENDPOINT_OR_EPOCH"
    if value["received_ms"] - value["event_ms"] > MAXIMUM_PAIR_AGE_MS:
        return "SOURCE_RECEIVE_DELAY_EXCEEDED"
    if quote and data_ms - value["transaction_ms"] > MAXIMUM_PAIR_AGE_MS:
        return "QUOTE_TOO_OLD_AT_ENDPOINT"
    return None


def endpoint(
    price,
    index,
    median_basis,
    *,
    symbol,
    activity_id,
    ledger_identity,
    data_ms,
    decision_ms,
    price_received_ms,
    index_received_ms,
    quote=None,
    quote_epoch=None,
    trade=None,
    trade_epoch=None,
):
    """同一端点的代数拆分与年龄代理；未知不补价，拆分不等于因果识别。"""
    if (
        type(data_ms) is not int
        or type(decision_ms) is not int
        or not data_ms + 1000 <= decision_ms
        or any(
            type(t) is not int or not data_ms - 1000 <= t <= decision_ms
            for t in (price_received_ms, index_received_ms)
        )
        or any(not isinstance(v, str) or not v for v in (symbol, activity_id, ledger_identity))
    ):
        raise ValueError("诊断端点早于完整观测或缺少实际决策时刻")
    if any(v is not None and v["symbol"] != symbol for v in (quote, trade)):
        raise ValueError("诊断端点引用了其他合约的原文")
    with localcontext() as context:
        context.prec = 50
        p, i = number(price, positive=True), number(index, positive=True)
        median = number(median_basis) if median_basis is not None else None
        basis = p / i - 1
        quote_error = source_available(quote, quote_epoch, data_ms, decision_ms, quote=True)
        trade_error = source_available(trade, trade_epoch, data_ms, decision_ms, quote=False)
        if trade_error is None and number(trade["price"], positive=True) != p:
            trade_error = "OBSERVED_AGGREGATE_PRICE_DOES_NOT_MATCH_BAR_CLOSE"
        mid_basis = print_component = None
        if quote_error is None:
            mid = (number(quote["bid"], positive=True) + number(quote["ask"], positive=True)) / 2
            mid_basis, print_component = mid / i - 1, (p - mid) / i
            if abs(basis - mid_basis - print_component) > Decimal("1e-45") * max(1, abs(basis)):
                raise ValueError("基差代数拆分不守恒")
        age = data_ms - trade["transaction_ms"] if trade_error is None else None
        return {
            "schema": VERSION,
            "data_ms": data_ms,
            "decision_ms": decision_ms,
            "symbol": symbol,
            "activity_id": activity_id,
            "ledger_identity": ledger_identity,
            "price_received_ms": price_received_ms,
            "index_received_ms": index_received_ms,
            "basis": str(basis),
            "median_basis": str(median) if median is not None else None,
            "residual": str(basis - median) if median is not None else None,
            "mid_basis": str(mid_basis) if mid_basis is not None else None,
            "print_component": str(print_component) if print_component is not None else None,
            "quote_unknown_reason": quote_error,
            "trade_age_unknown_reason": trade_error,
            "observed_aggregate_trade_age_proxy_ms": age,
            "trade_age_group": None
            if age is None
            else "up_to_5s"
            if age <= 5000
            else "5s_to_60s"
            if age <= 60_000
            else "over_60s",
            "quote_record_sha256": quote["record_sha256"] if quote is not None else None,
            "trade_record_sha256": trade["record_sha256"] if trade is not None else None,
            "causal_artifact_proven": False,
            "real_fills_proven": False,
            "orders_enabled": False,
        }


def changes(first, second, direction):
    """单列真实基差、中间价、成交印记与基线变化，不能把基线移动当成收益。"""
    if (
        direction not in {-1, 1}
        or first["data_ms"] >= second["data_ms"]
        or any(first[k] != second[k] for k in ("symbol", "activity_id", "ledger_identity"))
    ):
        raise ValueError("诊断方向或端点顺序不合法")
    with localcontext() as context:
        context.prec = 50
        values = {}
        for name in ("basis", "mid_basis", "print_component", "median_basis", "residual"):
            values[name] = (
                None
                if first[name] is None or second[name] is None
                else str(direction * (number(second[name]) - number(first[name])))
            )
    return {
        "directional_changes": values,
        "account_return_computed": False,
        "causal_artifact_proven": False,
        "orders_enabled": False,
    }
