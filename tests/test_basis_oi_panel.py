"""来源人口、原值、失效封存和覆盖守恒的合成验证，不读取真实市场。"""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tradingagents.dataflows.research import basis_oi_panel as panel
from tradingagents.dataflows.research.uniform_inputs import dependency
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts


@pytest.fixture(scope="module")
def full_population():
    dependencies = [
        dependency(kind, f"S{i:04d}USDT", day)
        for i in range(748)
        for day in (panel.utc_day(t) for t in range(panel.START, panel.END, panel.DAY))
        for kind in panel.KINDS
    ]
    records = [r | {"status": "PUBLIC_404"} for r in dependencies]
    protocol = {
        "schema": "cz-unconditional-public-factor-inputs.v1",
        "dependencies": dependencies,
        "orders_enabled": False,
        "leader_labels_read": False,
        "returns_evaluated": False,
    }
    result = {
        "schema": "cz-unconditional-public-factor-inputs-result.v1",
        "planned_dependencies": 25432,
        "recorded_dependencies": 25432,
        "records": records,
        "statuses": {"PUBLIC_404": 25432},
        "orders_enabled": False,
        "returns_evaluated": False,
    }
    return protocol, result


def test_population_accepts_unknown_values_without_requiring_success(full_population):
    symbols, records = panel.validate_population(*full_population)
    assert len(symbols) == 748 and len(records) == 25432


def test_successful_subset_or_duplicate_identity_cannot_replace_population(full_population):
    protocol, result = copy.deepcopy(full_population)
    protocol["dependencies"] = [r for r in protocol["dependencies"] if r["symbol"] < "S0718USDT"]
    with pytest.raises(ValueError, match="总体"):
        panel.validate_population(protocol, result)
    protocol, result = copy.deepcopy(full_population)
    result["records"][1] = result["records"][0]
    with pytest.raises(ValueError, match="重复"):
        panel.validate_population(protocol, result)


def test_unknown_download_status_and_published_summary_cannot_be_silently_changed(full_population):
    protocol, result = copy.deepcopy(full_population)
    result["records"][0]["status"] = "REPLACED_BY_CACHE"
    with pytest.raises(ValueError, match="状态"):
        panel.validate_population(protocol, result)
    protocol, result = copy.deepcopy(full_population)
    result["statuses"]["PUBLIC_404"] -= 1
    with pytest.raises(ValueError, match="统计"):
        panel.validate_population(protocol, result)


def bound_fixture(tmp_path, record, payload=None, receipt_status=200):
    relative = record["file"]
    archive = tmp_path / relative
    archive.parent.mkdir(parents=True)
    if payload is not None:
        archive.write_bytes(payload)
        record["sha256"] = sha(archive)
        archive.with_suffix(".zip.CHECKSUM").write_text(record["sha256"] + " file.zip\n")
    save(
        Path(str(archive) + ".receipt.json"),
        {"url": record["url"], "status": receipt_status, "sha256": record.get("sha256")},
    )
    bound = object.__new__(panel.BoundSource)
    bound.root, bound.inputs = tmp_path.resolve(), {}
    bound.expected = {
        str(p.relative_to(tmp_path)): sha(p) for p in tmp_path.rglob("*") if p.is_file()
    }
    return bound


def test_archive_binds_receipt_checksum_and_raw_bytes(tmp_path):
    row = dependency("metrics", "甲USDT", "2026-10-01") | {"status": panel.VERIFIED}
    bound = bound_fixture(tmp_path, row, b"synthetic original")
    assert bound.archive(row) == b"synthetic original"
    (tmp_path / row["file"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="字节"):
        bound.archive(row)


def test_failed_source_is_not_rescued_from_a_present_archive(tmp_path):
    row = dependency("metrics", "AUSDT", "2026-10-01") | {"status": "FAILED_OR_UNKNOWN"}
    bound = bound_fixture(tmp_path, row, b"present but failure stays unknown")
    assert bound.archive(row) is None and not bound.inputs


def test_404_conflict_and_source_escape_are_rejected(tmp_path):
    row = dependency("metrics", "AUSDT", "2026-10-01") | {"status": "PUBLIC_404"}
    bound = bound_fixture(tmp_path, row, receipt_status=404)
    assert bound.archive(row) is None
    (tmp_path / row["file"]).write_bytes(b"conflicting success")
    with pytest.raises(ValueError, match="404"):
        bound.archive(row)
    for relative in ("../outside", "/absolute/file"):
        with pytest.raises(ValueError, match="路径"):
            panel.safe_source_path(tmp_path, relative)


def test_symlink_inside_root_cannot_replace_a_source(tmp_path):
    (tmp_path / "original").write_bytes(b"source")
    (tmp_path / "alias").symlink_to("original")
    with pytest.raises(ValueError, match="软链接"):
        panel.safe_source_path(tmp_path, "alias")


def test_explicit_arrow_schema_retains_empty_columns_decimal_text_and_lineage():
    rows = [
        {
            "symbol": "甲USDT",
            "clock_ms": 0,
            "index_completed_ms": None,
            "index_duplicate_count": None,
            "index_source_line_nos": [],
            "sum_open_interest": None,
        }
    ]
    schema = panel.panel_schema(rows)
    assert schema.field("sum_open_interest").type == pa.string()
    assert schema.field("index_completed_ms").type == pa.int64()
    changed = rows[0] | {
        "sum_open_interest": "0.000000000000000000001",
        "index_source_line_nos": [2, 3],
    }
    table = pa.Table.from_pylist([changed], schema=schema)
    assert table.to_pylist() == [changed]


def clock_rows():
    return [
        {
            "clock_ms": int(t),
            "index_1m_close": "100",
            "index_1m_close_value_status": "KNOWN_RAW_VALUE",
            "sum_open_interest": "0" if i % 2 else None,
            "sum_open_interest_value_status": "KNOWN_RAW_VALUE" if i % 2 else "EMPTY",
        }
        for i, t in enumerate(range(panel.START, panel.START + panel.DAY, 5 * panel.MINUTE))
    ]


def test_coverage_preserves_real_zero_and_unknown_by_all_calendar_axes():
    rows = clock_rows()
    coverage = panel.Coverage(
        [r["clock_ms"] for r in rows], ["index_1m_close", "sum_open_interest"]
    )
    coverage.add("甲USDT", rows)
    result = coverage.result()
    oi = result["fields"]["sum_open_interest"]
    assert oi["known_rows"] == oi["unknown_rows"] == 144
    assert oi["lag_ms"] == 600000
    assert result["by_day"]["2026-09-28"]["sum_open_interest"] == {
        "EMPTY": 144,
        "KNOWN_RAW_VALUE": 144,
    }
    assert result["by_utc_hour"]["23"]["sum_open_interest"] == {"EMPTY": 6, "KNOWN_RAW_VALUE": 6}
    assert len(result["by_clock"]) == 288
    assert sum(x["complete_fields_rows"] for x in result["by_clock"]) == 144


def test_bad_row_does_not_partially_mutate_coverage():
    rows = clock_rows()
    coverage = panel.Coverage(
        [r["clock_ms"] for r in rows], ["index_1m_close", "sum_open_interest"]
    )
    rows[100]["sum_open_interest"] = "1"
    with pytest.raises(ValueError, match="不一致"):
        coverage.add("AUSDT", rows)
    assert coverage.rows == 0 and not coverage.by_symbol
    assert not np.any(coverage.known_by_clock)


def test_compressed_calendar_or_duplicate_symbol_cannot_inflate_coverage():
    rows = clock_rows()
    coverage = panel.Coverage(
        [r["clock_ms"] for r in rows], ["index_1m_close", "sum_open_interest"]
    )
    with pytest.raises(ValueError, match="日历"):
        coverage.add("AUSDT", rows[:-1])
    coverage.add("AUSDT", rows)
    with pytest.raises(ValueError, match="日历"):
        coverage.add("AUSDT", rows)
    assert coverage.rows == 288


def test_all_raw_lines_including_outside_panel_clock_are_preserved(tmp_path):
    original = [
        {
            "source_line_no": 2,
            "raw_T_ms": panel.END - 300000,
            "fields": ["甲USDT", "1.2345678901234567890123456789"],
        },
        {"source_line_no": 3, "fields": ["unparseable time", ""]},
    ]
    context = {k: "synthetic" for k in panel.RAW_SCHEMA.names if k != "record_json"}
    target = tmp_path / "raw.parquet"
    with pq.ParquetWriter(target, panel.RAW_SCHEMA) as writer:
        panel.write_raw(writer, context, original)
    actual = pq.read_table(target).to_pylist()
    assert [json.loads(r["record_json"]) for r in actual] == original


def test_existing_output_is_untouched_and_partial_preparation_failure_is_sealed(
    tmp_path, monkeypatch
):
    target = tmp_path / "new"

    def fail_after_create(source, output):
        output = Path(output)
        output.mkdir(exist_ok=False)
        (output / "partial.txt").write_text("partial evidence")
        raise ValueError("合成准备阶段改动")

    monkeypatch.setattr(panel, "_run", fail_after_create)
    with pytest.raises(ValueError, match="合成"):
        panel.run(tmp_path / "source", target)
    verify_receipts(target)
    before = sha(target / "receipts.json")
    with pytest.raises(FileExistsError):
        panel.run(tmp_path / "source", target)
    assert sha(target / "receipts.json") == before
    assert read(target / "construction_failure.json")["returns_evaluated"] is False


def test_nested_output_guard_runs_before_reading_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "nested"
    with pytest.raises(ValueError, match="嵌套"):
        panel.run(source, target)
    assert not target.exists()
