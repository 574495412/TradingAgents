"""时间标签未来泄漏、缺失窗口、零分母与原文粒度的反证。"""

import csv
import io
import math
import zipfile

import pytest

from tradingagents.dataflows.research import metrics_clock_audit as module


def archive(rows):
    text, target = io.StringIO(), io.BytesIO()
    csv.writer(text).writerows(rows)
    with zipfile.ZipFile(target, "w") as zipped:
        zipped.writestr("metrics.csv", text.getvalue())
    return target.getvalue()


def test_following_window_detects_future_volume_and_does_not_rewrite_preceding_window():
    volumes = {k * module.MINUTE: (10, 2 if k < 5 else 8) for k in range(10)}
    clock = 5 * module.MINUTE
    assert module.ratio(volumes, clock, -5) == 0.25
    assert module.ratio(volumes, clock, 0) == 4
    pairs = [{"archive_ratio": 4, "preceding_five_minutes": 0.25, "following_five_minutes": 4}]
    result = module.summarize(pairs)
    assert result["interpretations"]["following_five_minutes"]["close_rows"] == 1
    assert result["interpretations"]["preceding_five_minutes"]["close_rows"] == 0
    volumes[clock] = 10, 0
    assert module.ratio(volumes, clock, -5) == 0.25
    assert module.ratio(volumes, clock, 0) != 4


def test_missing_minutes_and_zero_sell_remain_unknown_and_share_comparison_population():
    volumes = {k * module.MINUTE: (10, 10) for k in range(10)}
    assert module.ratio(volumes, 5 * module.MINUTE, 0) is None
    volumes.pop(0)
    assert module.ratio(volumes, 5 * module.MINUTE, -5) is None
    pairs = [
        {"archive_ratio": 1, "preceding_five_minutes": 1, "following_five_minutes": None},
        {"archive_ratio": 1, "preceding_five_minutes": 2, "following_five_minutes": 1},
    ]
    result = module.summarize(pairs)
    assert result["scheduled_rows"] == 2 and result["common_rows"] == 1
    assert all(v["common_rows"] == 1 for v in result["interpretations"].values())
    assert result["interpretations"]["preceding_five_minutes"]["close_rows"] == 0


def test_sparse_unsorted_metrics_preserve_empty_fields_zero_interest_and_exact_decimal_conflicts():
    data = archive(
        [
            ["create_time", "symbol", "sum_open_interest", module.FLOW],
            ["2026-10-01 00:05:00", "AUSDT", "0", ""],
            ["2026-10-01 00:00:00", "AUSDT", "1000000000.000000001", "1"],
            ["2026-10-01 00:00:00", "AUSDT", "1000000000.000000002", "1"],
        ]
    )
    rows, quality = module.parse_metrics(data, "AUSDT", "2026-10-01")
    assert rows[0][1][module.FLOW] is None and rows[1][1] != rows[2][1]
    assert quality["missing_by_column"][module.FLOW] == quality["zero_open_interest_rows"] == 1
    assert quality["missing_points_on_five_minute_reference_grid"] == 286
    assert quality["nonincreasing_steps_in_original_order"] == 2
    assert quality["duplicate_timestamp_rows"] == 1


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1"])
def test_invalid_metrics_are_not_promoted_by_an_officially_matching_checksum(value):
    data = archive(
        [
            ["create_time", "symbol", "sum_open_interest", module.FLOW],
            ["2026-10-01 00:00:00", "AUSDT", value, "1"],
        ]
    )
    with pytest.raises(ValueError):
        module.parse_metrics(data, "AUSDT", "2026-10-01")


def test_non_utc_or_subsecond_labels_cannot_be_silently_reinterpreted():
    assert module.timestamp("2026-10-01T00:00:00Z") == module.timestamp("2026-10-01 00:00:00")
    for value in ("2026-10-01T00:00:00+08:00", "2026-10-01T00:00:00.001"):
        with pytest.raises(ValueError):
            module.timestamp(value)


def test_minute_quantity_must_conserve_and_a_missing_close_cannot_become_complete():
    row = [0, "1", "1", "1", "1", "10", 59999, "10", 1, "3", "3", "0"]
    assert module.minute_volumes({"symbol": "AUSDT", "rows": [row]}, "AUSDT") == {0: (10, 3)}
    for replacement in (math.nan, -1, 11):
        changed = row.copy()
        changed[9] = replacement
        with pytest.raises(ValueError):
            module.minute_volumes({"symbol": "AUSDT", "rows": [changed]}, "AUSDT")
    with pytest.raises(ValueError):
        module.minute_volumes({"symbol": "AUSDT", "rows": [row, row]}, "AUSDT")


def test_live_prefix_requires_explicit_opt_in_before_any_source_read(tmp_path):
    with pytest.raises(ValueError, match="采集尚未终态"):
        module.run(tmp_path / "source", tmp_path / "minutes", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_sealed_view_does_not_erase_the_original_unfinished_collection_gate(tmp_path):
    module.save(
        tmp_path / "protocol.json",
        {
            "schema": "cz-uniform-public-input-view.v1",
            "primary_collection_terminal": False,
            "leader_labels_read": False,
            "orders_enabled": False,
            "returns_evaluated": False,
        },
    )
    module.save(tmp_path / "result.json", {"records": []})
    module.save(tmp_path / "receipts.json", {})
    with pytest.raises(ValueError, match="采集尚未终态"):
        module.source_records(tmp_path, False)


def test_index_completed_clock_and_decimal_duplicates_remain_distinct():
    first = module.timestamp("2026-10-01")
    row = [
        first,
        "1000000000.000000001",
        "1000000001",
        "999999999",
        "1000000000.000000001",
        "0",
        first + 59999,
        "0",
        0,
        "0",
        "0",
        "0",
    ]
    changed = row.copy()
    changed[4] = "1000000000.000000002"
    parsed, quality = module.parse_index(archive([row, changed]), "2026-10-01")
    assert parsed[0][0] == parsed[1][0] == first + module.MINUTE
    assert parsed[0][1] != parsed[1][1]
    assert (
        quality["duplicate_timestamp_rows"] == 1
        and quality["missing_points_on_minute_reference_grid"] == 1439
    )


@pytest.mark.parametrize("bad", ["high_low", "end_clock", "nonfinite"])
def test_index_bytes_can_match_official_checksum_yet_still_fail_economic_fields(bad):
    first = module.timestamp("2026-10-01")
    row = [first, "100", "101", "99", "100", "0", first + 59999, "0", 0, "0", "0", "0"]
    if bad == "high_low":
        row[4] = "102"
    elif bad == "end_clock":
        row[6] += 1
    else:
        row[1] = "NaN"
    with pytest.raises(ValueError):
        module.parse_index(archive([row]), "2026-10-01")
