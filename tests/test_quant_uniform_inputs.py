"""完整依赖采集的边界、坏负载、限流和真实缺失反证。"""

import csv
import io
import threading
import zipfile
from urllib.error import HTTPError, URLError

import pytest

from tradingagents.dataflows.research import uniform_inputs as module


def archive(rows):
    text, buffer = io.StringIO(), io.BytesIO()
    csv.writer(text).writerows(rows)
    with zipfile.ZipFile(buffer, "w") as stream:
        stream.writestr("public.csv", text.getvalue())
    return buffer.getvalue()


def test_dependency_is_fixed_by_cartesian_inputs_and_unicode_symbols_are_encoded():
    item = module.dependency("metrics", "牛来USDT", "2026-10-01")
    assert item["url"].startswith(module.BASE) and "%E7%89%9B" in item["url"]
    assert module.dependency("indexPriceKlines", "AUSDT", "2026-10-01")["url"].endswith(
        "/AUSDT/1m/AUSDT-1m-2026-10-01.zip"
    )
    with pytest.raises(ValueError):
        module.dependency("metrics", "../A", "2026-10-01")


def test_metrics_keeps_boundary_duplicates_visible_without_claiming_publication_time():
    item = module.dependency("metrics", "AUSDT", "2026-10-01")
    data = archive(
        [
            ["create_time", "symbol", "sum_open_interest"],
            ["2026-10-01 00:00:00", "AUSDT", "1"],
            ["2026-10-01 00:00:00", "AUSDT", "2"],
        ]
    )
    result = module.profile(data, item)
    assert result["duplicate_timestamps"] == 1 and result["rows"] == 2
    assert result["historical_first_publication_time_proven"] is False


@pytest.mark.parametrize("failure", ["bad_end", "invalid_price", "wrong_day", "wrong_symbol"])
def test_invalid_archive_payload_does_not_turn_into_valid_feature_data(failure):
    if failure == "wrong_symbol":
        item = module.dependency("metrics", "AUSDT", "2026-10-01")
        data = archive(
            [["create_time", "symbol", "sum_open_interest"], ["2026-10-01 00:00:00", "BUSDT", "1"]]
        )
    else:
        item = module.dependency("indexPriceKlines", "AUSDT", "2026-10-01")
        start = int(module.pd.Timestamp("2026-10-01", tz="UTC").value // 1_000_000)
        row = [start, "100", "101", "99", "100", "1", start + 59999, "100", 1, "1", "100", "0"]
        if failure == "bad_end":
            row[6] += 1
        elif failure == "invalid_price":
            row[4] = "NaN"
        else:
            row[0] -= 60_000
            row[6] -= 60_000
        data = archive([row])
    with pytest.raises(ValueError):
        module.profile(data, item)


@pytest.mark.parametrize("status", [404, 429, 503, 0, 200])
def test_only_404_is_a_missing_archive_while_rate_limit_stops_new_requests(
    monkeypatch, tmp_path, status
):
    attempts, now = [], [0.0]
    fetcher = module.Fetcher(100)

    class Stop:
        stopped = False

        def wait(self, seconds):
            now[0] += seconds
            return self.stopped

        def set(self):
            self.stopped = True

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, maximum):
            return b"original-bytes"

    class Opener:
        def open(self, url, timeout):
            attempts.append(now[0])
            if status == 0:
                raise URLError("人工网络失败")
            if status != 200:
                raise HTTPError(url, status, "人工状态", {}, None)
            return Response()

    fetcher.stop, fetcher.opener = Stop(), Opener()
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    target = tmp_path / "public.zip"
    if status in {200, 404}:
        value = fetcher.fetch(module.BASE + "/public.zip", target)
        assert (value is None) == (status == 404)
        assert len(attempts) == 1
        assert module.read(target.with_suffix(".zip.receipt.json"))["status"] == status
        assert fetcher.fetch(module.BASE + "/public.zip", target) == value
        assert len(attempts) == 1
    else:
        with pytest.raises(ValueError):
            fetcher.fetch(module.BASE + "/public.zip", target)
        assert len(attempts) == (1 if status == 429 else 3)
        assert not target.with_suffix(".zip.receipt.json").exists()
        assert list(tmp_path.glob("*.failed-*.json"))
        assert fetcher.stop.stopped == (status == 429)


def test_redirect_cannot_reach_an_undeclared_host():
    from urllib.request import Request

    with pytest.raises(HTTPError):
        module.NoRedirects().redirect_request(
            Request(module.BASE + "/one"), None, 302, "redirect", {}, "https://example.com/private"
        )


def test_negative_open_interest_and_foreign_fetch_target_are_rejected(tmp_path):
    item = module.dependency("metrics", "AUSDT", "2026-10-01")
    data = archive(
        [["create_time", "symbol", "sum_open_interest"], ["2026-10-01 00:00:00", "AUSDT", "-1"]]
    )
    with pytest.raises(ValueError):
        module.profile(data, item)
    with pytest.raises(ValueError):
        module.Fetcher(100).fetch("https://example.com/private", tmp_path / "invalid.zip")


def test_complete_collection_creates_evidence_directory_before_result_writes(tmp_path):
    dependencies = [
        module.dependency(kind, symbol, "2026-10-01")
        for symbol in ("AUSDT", "BUSDT")
        for kind in ("metrics", "indexPriceKlines")
    ]
    stop = threading.Event()

    def worker(item):
        assert (tmp_path / "records").is_dir()
        return item | {
            "status": "PUBLIC_404" if item["kind"] == "metrics" else "VERIFIED_BYTES_AND_PROFILE"
        }

    result = module.collect_records(tmp_path, dependencies, worker, stop)
    assert len(result) == 4 and not stop.is_set()
    files = list((tmp_path / "records").glob("*.json"))
    assert len(files) == 4
    assert {module.read(path)["url"] for path in files} == {item["url"] for item in dependencies}


def test_main_collection_write_failure_stops_requests_and_does_not_wait_for_all_network_jobs(
    monkeypatch, tmp_path
):
    stop = threading.Event()
    dependencies = [module.dependency("metrics", f"A{i}USDT", "2026-10-01") for i in range(100)]
    completed = []

    def worker(item):
        if completed:
            assert stop.wait(1)
        completed.append(item["url"])
        return item | {"status": "PUBLIC_404"}

    def broken_write(path, value):
        raise OSError("人工证据落盘失败")

    monkeypatch.setattr(module, "save", broken_write)
    with pytest.raises(OSError):
        module.collect_records(tmp_path, dependencies, worker, stop)
    assert stop.is_set() and len(completed) < len(dependencies)


def test_old_relative_cache_uses_its_explicit_root_when_running_from_immutable_package(
    monkeypatch, tmp_path
):
    root = tmp_path / "original_repo" / "artifacts" / "public_v001"
    item = module.dependency("metrics", "AUSDT", "2026-10-01")
    expected = root / item["file"]
    expected.parent.mkdir(parents=True)
    expected.write_bytes("原始公开档案".encode())
    record = {
        "symbol": "AUSDT",
        "day": "2026-10-01",
        "path": "artifacts/public_v001/" + item["file"],
    }
    frozen = tmp_path / "immutable_runtime"
    frozen.mkdir()
    monkeypatch.chdir(frozen)
    assert module.resolve_reuse_archive(root, record, "metrics") == expected
    assert (
        module.resolve_reuse_archive(root, record | {"path": str(expected)}, "metrics") == expected
    )
    for path in (
        "../public_v001/" + item["file"],
        "/unrelated/" + item["file"],
        "artifacts/wrong/" + item["file"],
    ):
        with pytest.raises(ValueError):
            module.resolve_reuse_archive(root, record | {"path": path}, "metrics")
