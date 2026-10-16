"""缓存快照不得改写来源、混淆接收时钟或缩小全量人口。"""

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from tradingagents.dataflows.research import cache_snapshot as module
from tradingagents.dataflows.research.uniform_inputs import Fetcher
from tradingagents.research.quant.artifacts import read, save, sha


def make_archive(root, kind="metrics"):
    item = module.dependency(kind, "AUSDT", "2026-10-01")
    path = root / item["file"]
    path.parent.mkdir(parents=True)
    text = io.StringIO()
    if kind == "metrics":
        csv.writer(text).writerows(
            [["create_time", "symbol", "sum_open_interest"], ["2026-10-01 00:00:00", "AUSDT", "1"]]
        )
    else:
        clock = 1790812800000
        csv.writer(text).writerow(
            [clock, "100", "101", "99", "100", "0", clock + 59999, "0", 0, "0", "0", "0"]
        )
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("public.csv", text.getvalue())
    path.with_suffix(".zip.CHECKSUM").write_text(sha(path) + "  " + path.name)
    save(
        path.with_suffix(".zip.receipt.json"),
        {
            "status": 200,
            "url": item["url"],
            "sha256": sha(path),
            "captured_at": "2026-10-15T10:00:00+00:00",
        },
    )
    record = {
        "status": "VERIFIED",
        "symbol": "AUSDT",
        "day": item["day"],
        "sha256": sha(path),
        "path": str(Path("artifacts") / root.name / item["file"]),
    }
    return item, record, path


def test_snapshot_preserves_original_receipt_time_and_bytes_across_working_directories(
    tmp_path, monkeypatch
):
    root, output, cwd = (tmp_path / name for name in ("cache", "snapshot", "immutable"))
    output.mkdir()
    cwd.mkdir()
    item, record, original = make_archive(root)
    before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.chdir(cwd)
    identities = {}
    value = module.snapshot_archive(root, item, record, output, identities)
    assert value["status"] == "VERIFIED_BYTES_AND_PROFILE"
    assert value["historical_first_publication_time_proven"] is False
    assert value["original_archive"] == str(original)
    assert (
        read(output / (item["file"] + ".receipt.json"))["captured_at"]
        == "2026-10-15T10:00:00+00:00"
    )
    assert before == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["foreign_url", "wrong_checksum"])
def test_corrupted_source_is_not_copied_or_relabelled_as_success(tmp_path, damage):
    root, output = tmp_path / "cache", tmp_path / "snapshot"
    output.mkdir()
    item, record, original = make_archive(root)
    if damage == "wrong_checksum":
        original.with_suffix(".zip.CHECKSUM").write_text("0" * 64)
    else:
        receipt = original.with_suffix(".zip.receipt.json")
        value = read(receipt) | {"url": item["url"].replace("AUSDT", "BUSDT")}
        receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="旧缓存字节、来源或官方CHECKSUM不同"):
        module.snapshot_archive(root, item, record, output, {})
    assert not (output / item["file"]).exists()


def test_snapshot_pipeline_keeps_failed_candidate_and_full_universe_without_network(
    tmp_path, monkeypatch
):
    primary, metrics, index, output = (
        tmp_path / name for name in ("primary", "metrics_cache", "index_cache", "snapshot")
    )
    primary.mkdir()
    _, bad, original = make_archive(metrics)
    _, good, _ = make_archive(index, "indexPriceKlines")
    original.with_suffix(".zip.CHECKSUM").write_text("0" * 64)
    save(metrics / "metrics_manifest.json", {"records": [bad]})
    save(index / "manifest.json", {"records": [good]})
    symbols = ["AUSDT", *(f"B{i}USDT" for i in range(747))]
    days = module.pd.date_range("2026-09-28", "2026-10-14").strftime("%Y-%m-%d")
    items = [
        module.dependency(kind, symbol, day)
        for symbol in symbols
        for day in days
        for kind in ("metrics", "indexPriceKlines")
    ]
    registered = {
        str(p): sha(p) for p in (metrics / "metrics_manifest.json", index / "manifest.json")
    }
    save(
        primary / "protocol.json",
        {
            "dependencies": items,
            "leader_labels_read": False,
            "orders_enabled": False,
            "returns_evaluated": False,
            "reuse_manifest_sha256": registered,
        },
    )
    primary_before = sha(primary / "protocol.json")

    def forbidden_network(*args, **kwargs):
        raise AssertionError("缓存快照不能发起下载")

    monkeypatch.setattr(Fetcher, "fetch", forbidden_network)
    result = module.run(primary, metrics, index, output)
    assert result["planned_dependencies"] == 25432 and result["recorded_cache_archives"] == 2
    assert result["statuses"] == {"FAILED_OR_UNKNOWN": 1, "VERIFIED_BYTES_AND_PROFILE": 1}
    assert result["network_requests"] == 0 and result["whole_population_collected"] is False
    assert primary_before == sha(primary / "protocol.json")
    with pytest.raises(FileExistsError):
        module.run(primary, metrics, index, output)
    with pytest.raises(ValueError, match="快照不能写进"):
        module.run(primary, metrics, index, primary / "bad_snapshot")


def test_reduced_or_duplicate_population_cannot_be_called_complete():
    item = module.dependency("metrics", "AUSDT", "2026-10-01")
    with pytest.raises(ValueError):
        module.validate_dependencies([item, item])
