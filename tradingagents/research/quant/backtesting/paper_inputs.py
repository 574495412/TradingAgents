"""公开模拟账户的成交约束、报价和资金费原文解析，不读取私人账户。"""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows.research.live_capture import BASE, MINUTE, ROUTES, body


def number(value, *, positive=False):
    """财务计算拒绝非有限值；数量与价格另要求严格为正。"""
    result = Decimal(str(value))
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError("公开财务数值不合法")
    return result


def known(record, kind, now, *, maximum_age_ms=None):
    """原文必须已经实际收到；可选陈旧限制按实际接收时间计算。"""
    if (
        record.get("kind") != kind
        or record.get("status") != 200
        or not record.get("local_clock_consistent")
        or record.get("credentials_used") is not False
        or type(record.get("requested_ms")) is not int
        or type(record.get("received_ms")) is not int
        or not record["requested_ms"] <= record["received_ms"] <= now
        or (maximum_age_ms is not None and now - record["received_ms"] > maximum_age_ms)
    ):
        raise ValueError("公开财务输入尚未收到、失败或已经过时")
    return body(record)


def contract_rules(record, symbol, now):
    """保留市场与普通数量约束，不把显示精度当作真实下单步长。"""
    payload = known(record, "exchange", now, maximum_age_ms=5 * MINUTE)
    rows = [r for r in payload["symbols"] if r["symbol"] == symbol]
    if len(rows) != 1:
        raise ValueError("合约身份缺失或重复")
    row = rows[0]
    if (
        row["status"] != "TRADING"
        or row["quoteAsset"] != "USDT"
        or row["marginAsset"] != "USDT"
        or row["contractType"] != "TRADIFI_PERPETUAL"
        or int(row["onboardDate"]) > now
        or (int(row["deliveryDate"]) and now >= int(row["deliveryDate"]))
        or "MARKET" not in row["orderTypes"]
    ):
        raise ValueError("该合约当前没有研究入场资格")
    filters = {v["filterType"]: v for v in row["filters"]}
    if len(filters) != len(row["filters"]):
        raise ValueError("合约限制类型重复")
    lots = []
    for name in ("LOT_SIZE", "MARKET_LOT_SIZE"):
        f = filters[name]
        lo, hi, step = (number(f[k], positive=True) for k in ("minQty", "maxQty", "stepSize"))
        if hi < lo:
            raise ValueError("合约数量范围不合法")
        lots.append((lo, hi, step))
    return {
        "lots": lots,
        "minimum_notional": number(filters["MIN_NOTIONAL"]["notional"], positive=True),
        "source_sha256": record["body_sha256"],
    }


def quote(record, symbol, now, *, not_before_ms):
    """复核之后才请求报价；只提供模拟成交参考，不声称订单真实成交。"""
    value = known(record, "quotes", now, maximum_age_ms=5000)
    rows = value if isinstance(value, list) else [value]
    rows = [r for r in rows if r["symbol"] == symbol]
    if len(rows) != 1 or record["requested_ms"] < not_before_ms:
        raise ValueError("报价身份错误或报价请求早于复核")
    row = rows[0]
    if (
        type(row.get("time")) is not int
        or not now - 5000 <= row["time"] <= record["received_ms"] + 1000
    ):
        raise ValueError("盘口事件时间陈旧或晚于接收")
    bid, ask, bid_qty, ask_qty = (
        number(row[k], positive=True) for k in ("bidPrice", "askPrice", "bidQty", "askQty")
    )
    if ask < bid:
        raise ValueError("盘口买卖价交叉")
    return {
        "bid": bid,
        "ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "received_ms": record["received_ms"],
        "event_ms": row["time"],
        "source_sha256": record["body_sha256"],
    }


def funding_window(record, now):
    """核验明确区间和未截断原文；同钟普通与特别资金费保持独立。"""
    rows = known(record, "funding", now)
    url = urlparse(record["url"])
    if f"{url.scheme}://{url.netloc}" != BASE or url.path != ROUTES["funding"]:
        raise ValueError("资金费来源路径错误")
    query = parse_qs(url.query, strict_parsing=True)
    if set(query) != {"symbol", "startTime", "endTime", "limit"} or any(
        len(v) != 1 for v in query.values()
    ):
        raise ValueError("资金费必须声明完整区间、单一合约与上限")
    symbol = query["symbol"][0]
    start, end, limit = (int(query[k][0]) for k in ("startTime", "endTime", "limit"))
    if (
        record["symbol"] != symbol
        or not 0 <= start <= end <= record["requested_ms"] - 1000
        or limit != 1000
        or not isinstance(rows, list)
        or len(rows) >= limit
    ):
        raise ValueError("资金费区间尚未结束或响应可能已被截断")
    result, seen, previous = [], set(), -1
    for row in rows:
        stamp, kind = row.get("fundingTime"), row.get("rateType")
        if (
            row.get("symbol") != symbol
            or type(stamp) is not int
            or not start <= stamp <= end
            or stamp < previous
            or kind not in {"Regular", "Special"}
        ):
            raise ValueError("资金费身份、顺序、结算时刻或类型不合法")
        key = (stamp, kind)
        if key in seen:
            raise ValueError("同类资金费事件重复")
        seen.add(key)
        previous = stamp
        result.append(
            {
                "symbol": symbol,
                "time_ms": stamp,
                "rate_type": kind,
                "rate": str(number(row["fundingRate"]).normalize()),
                "mark_price": str(number(row["markPrice"], positive=True).normalize()),
                "received_ms": record["received_ms"],
                "source_sha256": record["body_sha256"],
            }
        )
    return {
        "symbol": symbol,
        "start_ms": start,
        "end_ms": end,
        "received_ms": record["received_ms"],
        "requested_ms": record["requested_ms"],
        "source_sha256": record["body_sha256"],
        "events": result,
    }


def covers(windows, start, end):
    """公开已查询区间必须连续覆盖持仓寿命；空事件不代表未来费用为零。"""
    cursor = start
    for window in sorted(windows, key=lambda v: (v["start_ms"], v["end_ms"])):
        if window["start_ms"] > cursor:
            break
        if window["end_ms"] >= cursor:
            cursor = window["end_ms"] + 1
        if cursor > end:
            return True
    return end < start
