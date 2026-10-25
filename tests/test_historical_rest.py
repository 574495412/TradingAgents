"""只用离线内存接口检验同钟边界、精确费率及断点恢复，不请求网络。"""

import json
from copy import deepcopy

import pytest

from tradingagents.dataflows.research.historical_rest import capture_pages, compare_month


def event(stamp, kind="Regular", rate="0.001", mark="100", **extra):
    return {
        "symbol": "BTCUSDT",
        "fundingTime": stamp,
        "fundingRate": rate,
        "markPrice": mark,
        "rateType": kind,
        **extra,
    }


class Server:
    """按官方两端含时间和上限返回；可检查核心实际发出的边界查询。"""

    def __init__(self, events):
        self.events, self.calls = deepcopy(events), []

    def __call__(self, first, last):
        self.calls.append((first, last))
        return deepcopy([row for row in self.events if first <= row["fundingTime"] <= last][:1000])


@pytest.mark.parametrize("size", [0, 1, 999, 1000, 1001, 2001])
def test_range_sizes_keep_every_event_exactly_once(size):
    rows = [event(i + 1, source_id=f"id{i}") for i in range(size)]
    server = Server(rows)
    result = capture_pages("BTCUSDT", 1, size + 2, server)
    assert result["status"] == ("API_RANGE_VERIFIED" if rows else "API_EMPTY_RANGE")
    assert result["events"] == rows and result["cursor_ms"] == size + 2
    assert result["pages"] == len(server.calls)
    assert result["server_historical_truth_proven"] is False
    if size >= 1000:
        assert server.calls[1] == (1000, 1000)
        assert server.calls[2][0] == 1000


def test_same_clock_special_after_full_main_page_survives_boundary_fence():
    rows = [event(i) for i in range(999)] + [event(999), event(999, "Special", "-0.002")]
    server = Server(rows)
    result = capture_pages("BTCUSDT", 0, 1001, server)
    assert result["events"] == rows
    assert result["event_type_counts"] == {"Regular": 1000, "Special": 1}
    assert server.calls == [(0, 1000), (999, 999), (999, 1000)]


@pytest.mark.parametrize("same_clock_count", [1000, 1001, 1500])
def test_even_exactly_1000_at_one_clock_is_unprovably_saturated(same_clock_count):
    rows = [event(5, source_id=str(i)) for i in range(same_clock_count)]
    server, checkpoints = Server(rows), []
    with pytest.raises(ValueError, match="SATURATED_TIMESTAMP"):
        capture_pages("BTCUSDT", 0, 20, server, on_checkpoint=checkpoints.append)
    assert server.calls == [(0, 19), (5, 5)]
    assert checkpoints[-1]["stage"] == "blocked" and not checkpoints[-1]["closed_events"]
    with pytest.raises(ValueError, match="SATURATED_TIMESTAMP"):
        capture_pages(
            "BTCUSDT",
            0,
            20,
            lambda a, b: pytest.fail("不得重新盲查饱和组"),
            checkpoint=checkpoints[-1],
        )


def test_999_same_clock_events_can_be_closed_without_losing_later_special():
    rows = [event(1)] + [event(5, source_id=str(i)) for i in range(999)] + [event(7, "Special")]
    result = capture_pages("BTCUSDT", 0, 10, Server(rows))
    assert result["events"] == rows and len(result["events"]) == 1001


def test_closed_boundary_ties_may_reorder_but_cannot_change_the_multiset():
    rows = [event(i) for i in range(999)] + [event(999), event(999, "Special")]
    server = Server(rows)

    def reordered(first, last):
        page = server(first, last)
        if first == 999:
            page.reverse()
        return page

    result = capture_pages("BTCUSDT", 0, 1001, reordered)
    assert len(result["events"]) == 1001
    assert result["event_type_counts"] == {"Regular": 1000, "Special": 1}


@pytest.mark.parametrize(
    "where",
    ["fence_changed", "fence_empty", "overlap_changed", "overlap_empty", "overlap_loses_special"],
)
def test_boundary_changes_or_empty_overlap_cannot_be_called_complete(where):
    rows = [event(i) for i in range(999)] + [event(999), event(999, "Special")]
    server = Server(rows)

    def changed(first, last):
        page = server(first, last)
        if len(server.calls) == 2 and where.startswith("fence"):
            if where.endswith("empty"):
                return []
            page[0]["fundingRate"] = "0.9"
        if len(server.calls) == 3 and where.startswith("overlap"):
            if where.endswith("empty"):
                return []
            if where.endswith("special"):
                return page[:1]
            page[0]["markPrice"] = "101"
        return page

    with pytest.raises(ValueError, match="BOUNDARY_GROUP_CHANGED"):
        capture_pages("BTCUSDT", 0, 1001, changed)


@pytest.mark.parametrize(
    "missing", ["markPrice", "rateType", "fundingRate", "fundingTime", "symbol"]
)
def test_missing_fields_never_default_to_regular_or_cash_zero(missing):
    row = event(1)
    row.pop(missing)
    with pytest.raises(ValueError, match="MISSING_FUNDING_FIELDS"):
        capture_pages("BTCUSDT", 0, 2, lambda a, b: [row])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("rateType", "Other", "UNKNOWN_RATE_TYPE"),
        ("markPrice", "0", "MARK_MUST_BE_POSITIVE"),
        ("markPrice", "-1", "MARK_MUST_BE_POSITIVE"),
        ("markPrice", "NaN", "MARK_NONFINITE"),
        ("fundingRate", "Infinity", "RATE_NONFINITE"),
        ("fundingTime", True, "FUNDING_TIME_MUST_BE_INTEGER"),
        ("symbol", "ETHUSDT", "SYMBOL_MISMATCH"),
    ],
)
def test_invalid_field_values_block_the_capture(field, value, reason):
    row = event(1)
    row[field] = value
    with pytest.raises(ValueError, match=reason):
        capture_pages("BTCUSDT", 0, 2, lambda a, b: [row])


def test_same_page_exact_or_decimal_equivalent_duplicates_are_unresolved():
    for rows in ([event(1), event(1)], [event(1, rate="0.0010"), event(1, rate="1E-3")]):
        with pytest.raises(ValueError, match="UNRESOLVED_MULTIPLICITY"):
            capture_pages("BTCUSDT", 0, 2, Server(rows))


def test_identical_required_values_with_different_original_event_ids_stay_distinct():
    rows = [event(1, source_id="a"), event(1, source_id="b")]
    result = capture_pages("BTCUSDT", 0, 2, Server(rows))
    assert result["events"] == rows


@pytest.mark.parametrize("budget", [1, 2])
def test_resume_after_full_page_or_completed_fence_keeps_prefix_and_group(budget):
    rows = [event(i) for i in range(999)] + [event(999), event(999, "Special")] + [event(1000)]
    checkpoints, server = [], Server(rows)
    with pytest.raises(ValueError, match="PAGE_BUDGET_EXCEEDED"):
        capture_pages(
            "BTCUSDT", 0, 1002, server, on_checkpoint=checkpoints.append, max_pages=budget
        )
    checkpoint = json.loads(json.dumps(checkpoints[-1]))
    result = capture_pages("BTCUSDT", 0, 1002, server, checkpoint=checkpoint)
    assert result["events"] == rows
    assert result["pages"] == 3 and result["fetch_calls_this_run"] == 3 - budget
    calls = len(server.calls)
    assert (
        capture_pages("BTCUSDT", 0, 1002, server, checkpoint=result["checkpoint"])[
            "fetch_calls_this_run"
        ]
        == 0
    )
    assert len(server.calls) == calls


def test_checkpoint_identity_and_group_tampering_fail_before_fetch():
    rows = [event(i) for i in range(1001)]
    checkpoints = []
    with pytest.raises(ValueError, match="PAGE_BUDGET_EXCEEDED"):
        capture_pages(
            "BTCUSDT", 0, 1002, Server(rows), on_checkpoint=checkpoints.append, max_pages=2
        )
    for field in ("symbol", "boundary_group"):
        checkpoint = deepcopy(checkpoints[-1])
        checkpoint[field] = "ETHUSDT" if field == "symbol" else []
        with pytest.raises(ValueError, match="CHECKPOINT"):
            capture_pages(
                "BTCUSDT",
                0,
                1002,
                lambda a, b: pytest.fail("无效断点不得请求"),
                checkpoint=checkpoint,
            )


def test_zero_negative_rates_extra_fields_and_one_millisecond_are_unchanged():
    rows = [event(1, rate="0", nested={"name": "原字段"}), event(60_001, "Special", "-0.000001")]
    result = capture_pages("BTCUSDT", 0, 60_002, Server(rows))
    assert result["events"] == rows and result["events"][1]["fundingTime"] == 60_001


def archive(stamp, rate="0.001"):
    return {"calc_time": str(stamp), "last_funding_rate": rate, "funding_interval_hours": "8"}


def test_month_comparison_is_precise_regular_only_and_preserves_special_cash_count():
    long_rate = "0.123456789012345678901234567890123456789"
    archived = [archive(1, "0"), archive(60_001, long_rate)]
    rows = [
        event(1, rate="-0.000"),
        event(60_001, rate=long_rate),
        event(60_001, "Special", "-0.002"),
    ]
    result = compare_month(archived, rows, 0, 100_000)
    assert (
        result["status"] == "MATCHED"
        and result["passed"]
        and result["regular_correspondence_proven"]
    )
    assert result["rest_special_events"] == 1 and result["special_events_retained_for_cash"]
    rows[1]["fundingRate"] = "0.123456789012345678901234567890123456788"
    assert compare_month(archived, rows, 0, 100_000)["status"] == "MISMATCH"


@pytest.mark.parametrize(
    "archived,rows", [([], []), ([archive(1)], []), ([], [event(1)]), ([], [event(1, "Special")])]
)
def test_empty_either_side_is_unknown_instead_of_a_false_zero_match(archived, rows):
    result = compare_month(archived, rows, 0, 10)
    assert result["status"] == "UNKNOWN_EMPTY" and not result["passed"]
    assert not result["regular_correspondence_proven"]


def test_archive_multiplicity_and_exact_milliseconds_are_not_set_deduplicated_or_rounded():
    assert compare_month([archive(1), archive(1)], [event(1)], 0, 10)["status"] == "MISMATCH"
    assert compare_month([archive(1)], [event(0)], 0, 10)["status"] == "MISMATCH"
    assert compare_month([archive(10)], [event(10)], 0, 10)["status"] == "UNKNOWN_EMPTY"


def test_fetch_failure_and_oversized_unsorted_or_out_of_range_pages_are_not_empty():
    def failed(a, b):
        raise TimeoutError("离线模拟传输失败")

    with pytest.raises(ValueError, match="FETCH_FAILED"):
        capture_pages("BTCUSDT", 0, 3, failed)
    for page in ([event(2), event(1)], [event(3)], [event(i) for i in range(1001)]):
        with pytest.raises(ValueError):
            capture_pages("BTCUSDT", 0, 3, lambda a, b, page=page: page)


def test_full_page_boundary_at_exact_range_end_needs_fence_but_no_later_query():
    rows = [event(i) for i in range(1000)]
    server = Server(rows)
    result = capture_pages("BTCUSDT", 0, 1000, server)
    assert result["events"] == rows
    assert server.calls == [(0, 999), (999, 999)]


def test_checkpoint_callback_cannot_mutate_internal_groups_or_events():
    rows = [event(i) for i in range(1001)]

    def malicious_callback(state):
        state["closed_events"].clear()
        state["cursor_ms"] = 999999

    result = capture_pages("BTCUSDT", 0, 1002, Server(rows), on_checkpoint=malicious_callback)
    assert result["events"] == rows and result["cursor_ms"] == 1002
