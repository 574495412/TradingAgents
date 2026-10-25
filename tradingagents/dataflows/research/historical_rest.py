"""公开资金费重复边界分页的纯核心；原文请求、限流及收据由调用者负责。"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation

LIMIT = 1000
SCHEMA = "ta-public-funding-checkpoint.v1"


def _copy(value):
    """回调与返回值不得共享可变状态，且检查点必须可由严格JSON保存。"""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("NON_JSON_EVENT_OR_CHECKPOINT") from error


def _integer(value, label):
    """布尔和浮点不能隐式成为原始毫秒。"""
    if type(value) is not int:
        raise ValueError(label + "_MUST_BE_INTEGER")
    return value


def _decimal(value, label):
    """有限十进制不经float或默认精度normalize，保留长小数及原毫秒。"""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(label + "_INVALID_DECIMAL")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(label + "_INVALID_DECIMAL") from error
    if not number.is_finite():
        raise ValueError(label + "_NONFINITE")
    return number


def _decimal_key(number):
    """按精确数值统一1e-3与0.001，不受Decimal上下文舍入影响。"""
    if number == 0:
        return [0, "0", 0]
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return [sign, "".join(map(str, digits)), exponent]


def _identity(row):
    """全部原字段都参与身份；费率及标记价仅作不损精度的数值规范化。"""
    value = _copy(row)
    value["fundingRate"] = {"exact_decimal": _decimal_key(_decimal(row["fundingRate"], "RATE"))}
    value["markPrice"] = {"exact_decimal": _decimal_key(_decimal(row["markPrice"], "MARK"))}
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _page(rows, symbol, begin, end, *, cap=LIMIT):
    """同页完全相同行没有eventId时无法解释其真实多重性，必须显式阻断。"""
    if not isinstance(rows, list) or (cap is not None and len(rows) > cap):
        raise ValueError("INVALID_PAGE_OR_OVER_LIMIT")
    result, prior, identities = [], None, set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not {"symbol", "fundingTime", "fundingRate", "markPrice", "rateType"} <= row.keys()
        ):
            raise ValueError("MISSING_FUNDING_FIELDS")
        if not isinstance(row["symbol"], str) or not row["symbol"] or row["symbol"] != symbol:
            raise ValueError("SYMBOL_MISMATCH")
        stamp = _integer(row["fundingTime"], "FUNDING_TIME")
        if not begin <= stamp <= end or (prior is not None and stamp < prior):
            raise ValueError("UNSORTED_OR_OUT_OF_RANGE_PAGE")
        if not isinstance(row["rateType"], str) or row["rateType"] not in {"Regular", "Special"}:
            raise ValueError("UNKNOWN_RATE_TYPE")
        _decimal(row["fundingRate"], "RATE")
        if _decimal(row["markPrice"], "MARK") <= 0:
            raise ValueError("MARK_MUST_BE_POSITIVE")
        identity = _identity(row)
        if identity in identities:
            raise ValueError("UNRESOLVED_MULTIPLICITY")
        identities.add(identity)
        result.append(_copy(row))
        prior = stamp
    return result


def _group(rows):
    """同钟顺序可以改变，但多重集合不能少行、换值或丢Special。"""
    return Counter(_identity(row) for row in rows)


def _initial(symbol, start, end):
    return {
        "schema": SCHEMA,
        "symbol": symbol,
        "start_ms": start,
        "end_ms": end,
        "cursor_ms": start,
        "stage": "main",
        "closed_events": [],
        "boundary_ms": None,
        "boundary_group": [],
        "pending_prefix": [],
        "pending_boundary_ms": None,
        "pending_boundary": [],
        "pages": 0,
        "last_query": None,
        "last_response_count": None,
        "error_code": None,
    }


def _checkpoint(value, symbol, start, end):
    """恢复时复核协议、原事件及状态，不信任单独的cursor数字。"""
    state = _copy(value)
    required = set(_initial(symbol, start, end))
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("INVALID_CHECKPOINT_SHAPE")
    if (state["schema"], state["symbol"], state["start_ms"], state["end_ms"]) != (
        SCHEMA,
        symbol,
        start,
        end,
    ):
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH")
    cursor = _integer(state["cursor_ms"], "CURSOR")
    pages = _integer(state["pages"], "PAGES")
    if (
        not start <= cursor <= end
        or pages < 0
        or state["stage"] not in {"main", "boundary", "complete", "blocked"}
    ):
        raise ValueError("INVALID_CHECKPOINT_STATE")
    closed = _page(state["closed_events"], symbol, start, end - 1, cap=None)
    if closed and closed[-1]["fundingTime"] > cursor:
        raise ValueError("CHECKPOINT_CLOSED_EVENTS_AFTER_CURSOR")
    boundary = state["boundary_ms"]
    if boundary is not None:
        _integer(boundary, "BOUNDARY")
        group = _page(state["boundary_group"], symbol, boundary, boundary)
        if (
            not group
            or len(group) >= LIMIT
            or _group(group) != _group([row for row in closed if row["fundingTime"] == boundary])
        ):
            raise ValueError("INVALID_CHECKPOINT_BOUNDARY")
        if state["stage"] not in {"complete", "blocked"} and boundary != cursor:
            raise ValueError("CHECKPOINT_BOUNDARY_CURSOR_MISMATCH")
    elif state["boundary_group"] or (closed and state["stage"] != "complete"):
        raise ValueError("CHECKPOINT_MISSING_BOUNDARY")
    if state["stage"] == "boundary":
        target = _integer(state["pending_boundary_ms"], "PENDING_BOUNDARY")
        if not cursor <= target < end:
            raise ValueError("INVALID_CHECKPOINT_PENDING_BOUNDARY")
        prefix = _page(state["pending_prefix"], symbol, cursor, target - 1, cap=None)
        pending = _page(state["pending_boundary"], symbol, target, target)
        if not pending or len(prefix) + len(pending) + len(state["boundary_group"]) != LIMIT:
            raise ValueError("INVALID_CHECKPOINT_PENDING_PAGE")
        if boundary is not None and any(row["fundingTime"] <= boundary for row in prefix):
            raise ValueError("CHECKPOINT_PENDING_OVERLAPS_CLOSED")
    elif state["stage"] != "blocked" and (
        state["pending_prefix"]
        or state["pending_boundary"]
        or state["pending_boundary_ms"] is not None
    ):
        raise ValueError("UNEXPECTED_CHECKPOINT_PENDING_PAGE")
    if state["stage"] == "complete" and cursor != end:
        raise ValueError("INCOMPLETE_CHECKPOINT_END")
    if state["stage"] == "blocked" and not isinstance(state["error_code"], str):
        raise ValueError("INVALID_BLOCKED_CHECKPOINT")
    return state


def capture_pages(
    symbol, start_ms, end_ms, fetch, checkpoint=None, on_checkpoint=None, max_pages=64
):
    """查询[start_ms,end_ms)，重复边界闭合后再推进，不漏同钟特殊事件。

    fetch(a,b)的两端均含，外层固定limit=1000并保存原响应。pages为累计已
    验响应数；max_pages限制本次调用的fetch次数，超额保留最近检查点可恢复。
    API_RANGE_VERIFIED只描述接口协议覆盖，不证明服务器历史真值或首次发布时间。
    """
    start, end = _integer(start_ms, "START"), _integer(end_ms, "END")
    if start >= end or not isinstance(symbol, str) or not symbol:
        raise ValueError("INVALID_SYMBOL_OR_RANGE")
    if type(max_pages) is not int or max_pages <= 0:
        raise ValueError("INVALID_PAGE_BUDGET")
    if not callable(fetch) or (on_checkpoint is not None and not callable(on_checkpoint)):
        raise ValueError("INVALID_CALLBACK")
    state = (
        _initial(symbol, start, end)
        if checkpoint is None
        else _checkpoint(checkpoint, symbol, start, end)
    )
    calls = 0

    def emit():
        if on_checkpoint is not None:
            on_checkpoint(_copy(state))

    def fail(code):
        state.update(stage="blocked", error_code=code)
        emit()
        raise ValueError(code)

    def request(begin, finish):
        nonlocal calls
        if calls >= max_pages:
            raise ValueError("PAGE_BUDGET_EXCEEDED")
        calls += 1
        try:
            response = fetch(begin, finish)
        except Exception as error:
            raise ValueError("FETCH_FAILED") from error
        rows = _page(response, symbol, begin, finish)
        state["pages"] += 1
        state["last_query"] = {"start_inclusive_ms": begin, "end_inclusive_ms": finish}
        state["last_response_count"] = len(rows)
        return rows

    if state["stage"] == "blocked":
        raise ValueError(state["error_code"])
    while state["stage"] != "complete":
        if state["stage"] == "main":
            rows = request(state["cursor_ms"], end - 1)
            anchor = state["boundary_ms"]
            if anchor is not None:
                overlap = [row for row in rows if row["fundingTime"] == anchor]
                if _group(overlap) != _group(state["boundary_group"]):
                    fail("BOUNDARY_GROUP_CHANGED")
                new_rows = [row for row in rows if row["fundingTime"] > anchor]
            else:
                new_rows = rows
            if len(rows) < LIMIT:
                state["closed_events"].extend(new_rows)
                state.update(stage="complete", cursor_ms=end)
                emit()
                break
            target = rows[-1]["fundingTime"]
            if anchor is not None and target <= anchor:
                fail("NON_ADVANCING_SATURATED_BOUNDARY")
            state.update(
                stage="boundary",
                pending_boundary_ms=target,
                pending_prefix=[row for row in new_rows if row["fundingTime"] < target],
                pending_boundary=[row for row in new_rows if row["fundingTime"] == target],
            )
            emit()
        else:
            target = state["pending_boundary_ms"]
            rows = request(target, target)
            if len(rows) == LIMIT:
                fail("SATURATED_TIMESTAMP")
            if not rows or not _group(state["pending_boundary"]) <= _group(rows):
                fail("BOUNDARY_GROUP_CHANGED")
            state["closed_events"].extend(state["pending_prefix"] + rows)
            state.update(
                cursor_ms=target,
                boundary_ms=target,
                boundary_group=rows,
                pending_prefix=[],
                pending_boundary_ms=None,
                pending_boundary=[],
                stage="main",
            )
            if target == end - 1:
                state.update(stage="complete", cursor_ms=end)
            emit()
    events = _copy(state["closed_events"])
    return {
        "status": "API_RANGE_VERIFIED" if events else "API_EMPTY_RANGE",
        "events": events,
        "cursor_ms": state["cursor_ms"],
        "pages": state["pages"],
        "fetch_calls_this_run": calls,
        "checkpoint": _copy(state),
        "server_historical_truth_proven": False,
        "event_type_counts": dict(Counter(row["rateType"] for row in events)),
    }


def compare_month(archive_rows, rest_events, start_ms, end_ms):
    """精确比较右开区间普通费率多重集合，Special留给现金而不强配归档。"""
    start, end = _integer(start_ms, "START"), _integer(end_ms, "END")
    if start >= end or not isinstance(archive_rows, list) or not isinstance(rest_events, list):
        raise ValueError("INVALID_COMPARISON_RANGE_OR_ROWS")
    archived = Counter()
    for row in archive_rows:
        if (
            not isinstance(row, dict)
            or not {"calc_time", "last_funding_rate", "funding_interval_hours"} <= row.keys()
        ):
            raise ValueError("MISSING_ARCHIVE_FIELDS")
        raw_clock = row["calc_time"]
        stamp = (
            int(raw_clock)
            if isinstance(raw_clock, str) and raw_clock.isdigit()
            else _integer(raw_clock, "CALC_TIME")
        )
        rate = _decimal(row["last_funding_rate"], "ARCHIVE_RATE")
        if _decimal(row["funding_interval_hours"], "ARCHIVE_INTERVAL") <= 0:
            raise ValueError("ARCHIVE_INTERVAL_MUST_BE_POSITIVE")
        if start <= stamp < end:
            archived[stamp, rate] += 1
    regular, special = Counter(), 0
    symbols = {row.get("symbol") for row in rest_events if isinstance(row, dict)}
    if len(symbols) > 1:
        raise ValueError("COMPARISON_REQUIRES_SINGLE_SYMBOL")
    if rest_events:
        # 不截去后才验证，以免将乱序、重复或缺字段隐藏在窗口外。
        stamps = [row.get("fundingTime") for row in rest_events if isinstance(row, dict)]
        if len(stamps) != len(rest_events):
            raise ValueError("MISSING_FUNDING_FIELDS")
        low, high = min(_integer(stamp, "FUNDING_TIME") for stamp in stamps), max(stamps)
        validated = _page(rest_events, next(iter(symbols)), low, high, cap=None)
        for row in validated:
            if start <= row["fundingTime"] < end:
                if row["rateType"] == "Regular":
                    regular[row["fundingTime"], _decimal(row["fundingRate"], "RATE")] += 1
                else:
                    special += 1
    missing, extra = archived - regular, regular - archived
    status = (
        "UNKNOWN_EMPTY"
        if not archived or not regular
        else ("MATCHED" if archived == regular else "MISMATCH")
    )
    return {
        "status": status,
        "passed": status == "MATCHED",
        "regular_correspondence_proven": status == "MATCHED",
        "archive_events": sum(archived.values()),
        "rest_regular_events": sum(regular.values()),
        "rest_special_events": special,
        "special_events_retained_for_cash": True,
        "archive_only_multiset_count": sum(missing.values()),
        "rest_only_multiset_count": sum(extra.values()),
        "timestamp_rounding_applied": False,
        "archive_has_proven_cash_mark_or_type": False,
        "api_range_coverage_must_be_proven_separately": True,
    }
