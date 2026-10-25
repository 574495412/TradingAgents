"""历史采集离线校验；所有HTTP均替身，不触发批量下载或收益研究。"""

import csv
import gzip
import hashlib
import io
import urllib.parse
import zipfile
from pathlib import Path

import pytest

from tradingagents.dataflows.research import historical_capture as capture


def task(kind="1m", symbol="PUBLICUSDT", listed=True):
    return {
        "id": f"{kind}_{symbol}_2026-08",
        "kind": kind,
        "symbol": symbol,
        "month": "2026-08",
        "key": capture.archive_key(kind, symbol, "2026-08"),
        "listed": listed,
        "checksum_key_listed": listed,
        "listed_bytes": None,
        "initial_status": "PENDING" if listed else "NOT_LISTED",
    }


def zipped(item, rows, header=None):
    body = io.StringIO()
    writer = csv.writer(body)
    writer.writerow(
        header
        or (
            capture.COLUMNS
            if item["kind"] == "1m"
            else ["calc_time", "funding_interval_hours", "last_funding_rate"]
        )
    )
    writer.writerows(rows)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(Path(item["key"]).name[:-4] + ".csv", body.getvalue())
    return stream.getvalue()


def candle(clock, **changes):
    row = [
        str(clock),
        "100",
        "101",
        "99",
        "100",
        "2",
        str(clock + 59999),
        "200",
        "3",
        "1",
        "100",
        "0",
    ]
    for index, value in changes.items():
        row[int(index)] = str(value)
    return row


def fixture_plan(item):
    return {
        "archive_tasks": [item],
        "rest_symbols": [item["symbol"]],
        "start_ms": capture.START_MS,
        "end_exclusive_ms": capture.END_MS,
    }


def test_all_symbols_and_unlisted_months_remain_explicit():
    item = task()
    metadata = [
        {
            "kind": item["kind"],
            "symbol": item["symbol"],
            "month": item["month"],
            "key": item["key"],
            "bytes": 10,
            "checksum_key_listed": True,
            "last_modified": "2026-09-01T00:00:00Z",
        }
    ]
    symbols, tasks = capture.build_tasks(["PUBLICUSDT"], ["HISTORYUSDT"], metadata)
    assert symbols == ["HISTORYUSDT", "PUBLICUSDT"]
    assert len(tasks) == 8 and sum(t["listed"] for t in tasks) == 1
    assert sum(t["initial_status"] == "NOT_LISTED" for t in tasks) == 7
    assert all(t["historical_tradability"] == "UNKNOWN" for t in tasks)


def test_csv_retains_all_original_fields_and_explicit_month_gaps(tmp_path):
    item = task()
    first, end = capture.month_bounds(item["month"])
    raw = tmp_path / "sample.zip"
    raw.write_bytes(zipped(item, [candle(first + 2 * 60000), candle(first + 3 * 60000)]))
    normalized = tmp_path / "normalized.csv.gz"
    profile, values = capture.profile_archive(raw, item, normalized)
    assert values == [] and profile["rows"] == 2
    assert profile["missing_minutes"] == (end - first) // 60000 - 2
    assert profile["missing_minute_intervals_right_open"] == [
        [first, first + 2 * 60000],
        [first + 4 * 60000, end],
    ]
    assert profile["historical_tradability_proven"] is False
    with gzip.open(normalized, "rt") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [*capture.COLUMNS, "completed_boundary_ms", "symbol"]
    assert rows[1][-2:] == [str(first + 3 * 60000), item["symbol"]]


@pytest.mark.parametrize(
    "changes",
    [
        {"1": "0"},
        {"2": "99"},
        {"5": "-1"},
        {"7": "-1"},
        {"8": "1.5"},
        {"9": "3"},
        {"10": "300"},
        {"4": "NaN"},
        {"6": "1"},
    ],
)
def test_bad_clock_price_or_quantity_is_not_normalized(tmp_path, changes):
    item = task()
    raw = tmp_path / "bad.zip"
    raw.write_bytes(zipped(item, [candle(capture.START_MS, **changes)]))
    with pytest.raises((ValueError, ArithmeticError)):
        capture.profile_archive(raw, item)


def test_duplicate_minutes_and_unknown_header_are_not_filled(tmp_path):
    item = task()
    raw = tmp_path / "duplicate.zip"
    raw.write_bytes(zipped(item, [candle(capture.START_MS)] * 2))
    with pytest.raises(ValueError, match="DUPLICATE"):
        capture.profile_archive(raw, item)
    raw.write_bytes(zipped(item, [], header=["wrong"]))
    with pytest.raises(ValueError, match="12_FIELDS"):
        capture.profile_archive(raw, item)


def test_funding_archive_keeps_plus_one_ms_without_cash_defaults(tmp_path):
    item = task("fundingRate")
    raw = tmp_path / "funding.zip"
    raw.write_bytes(zipped(item, [[str(capture.START_MS + 1), "8", "-0.00010000"]]))
    profile, rows = capture.profile_archive(raw, item)
    assert profile["minimum_ms"] == capture.START_MS + 1
    assert profile["funding_mark_and_type_available"] is False
    assert rows == [
        {
            "calc_time": str(capture.START_MS + 1),
            "funding_interval_hours": "8",
            "last_funding_rate": "-0.00010000",
        }
    ]
    assert "rateType" not in rows[0] and "markPrice" not in rows[0]


def test_official_checksum_must_match_both_bytes_and_filename(tmp_path):
    item = task()
    raw, check = tmp_path / "zip", tmp_path / "check"
    raw.write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    check.write_text(f"{digest}  {Path(item['key']).name}")
    capture.verified_checksum(raw, check, item)
    check.write_text(f"{digest}  other.zip")
    with pytest.raises(ValueError, match="CHECKSUM"):
        capture.verified_checksum(raw, check, item)


def test_client_refuses_nonlisted_keys_or_extra_rest_query_fields(tmp_path):
    item = task()
    client = capture.Client(tmp_path, fixture_plan(item))
    with pytest.raises(ValueError, match="已冻结"):
        client.validate_url(capture.BASE + item["key"].replace("PUBLIC", "OTHER"))
    rest = capture.Client(tmp_path, fixture_plan(item), rest=True)
    url = (
        capture.FUNDING
        + "?"
        + urllib.parse.urlencode(
            {
                "symbol": item["symbol"],
                "startTime": capture.START_MS,
                "endTime": capture.END_MS - 1,
                "limit": 1000,
            }
        )
    )
    rest.validate_url(url)
    with pytest.raises(ValueError, match="查询字段"):
        rest.validate_url(url + "&apiKey=forbidden")


def test_rest_rate_spacing_includes_every_request(tmp_path):
    item, clock, calls = task(), [100.0], []

    def sleep(seconds):
        clock[0] += seconds

    def transport(url, limit):
        calls.append(clock[0])
        return 200, {}, b"[]"

    client = capture.Client(
        tmp_path,
        fixture_plan(item),
        rest=True,
        transport=transport,
        monotonic=lambda: clock[0],
        sleep=sleep,
    )
    url = (
        capture.FUNDING
        + "?"
        + urllib.parse.urlencode(
            {
                "symbol": item["symbol"],
                "startTime": capture.START_MS,
                "endTime": capture.END_MS - 1,
                "limit": 1000,
            }
        )
    )
    client.fetch(url, tmp_path / "requests", 1000)
    client.fetch(url, tmp_path / "requests", 1000)
    assert calls == [100.0, 101.5]


def test_rate_limit_is_persisted_and_new_client_cannot_immediately_retry(tmp_path):
    item, calls = task(), []

    def transport(url, limit):
        calls.append(url)
        return 429, {"Retry-After": "120"}, b'{"code":429}'

    url = (
        capture.FUNDING
        + "?"
        + urllib.parse.urlencode(
            {
                "symbol": item["symbol"],
                "startTime": capture.START_MS,
                "endTime": capture.END_MS - 1,
                "limit": 1000,
            }
        )
    )
    for index in range(2):
        client = capture.Client(tmp_path, fixture_plan(item), rest=True, transport=transport)
        with pytest.raises(RuntimeError, match="BACKOFF"):
            client.fetch(url, tmp_path / f"requests{index}", 1000)
    assert len(calls) == 1
    assert len(list((tmp_path / "backoffs/rest").glob("*.json"))) == 1


def test_unlisted_does_not_fetch_and_verified_archive_is_resumable(tmp_path):
    item, calls = task(), []
    raw = zipped(item, [candle(capture.START_MS)])
    item["listed_bytes"] = len(raw)
    checksum = f"{hashlib.sha256(raw).hexdigest()}  {Path(item['key']).name}".encode()

    def transport(url, limit):
        calls.append(url)
        return 200, {}, checksum if url.endswith(".CHECKSUM") else raw

    capture.save(tmp_path / "capture_plan.json", fixture_plan(item))
    client = capture.Client(tmp_path, fixture_plan(item), transport=transport)
    first = capture.capture_archive(tmp_path, item, client)
    assert first["status"] == "VERIFIED" and len(calls) == 2
    assert capture.capture_archive(tmp_path, item, client) == first
    assert len(calls) == 2
    missing = task(symbol="HISTORYUSDT", listed=False)
    assert capture.capture_archive(tmp_path, missing, client)["status"] == "NOT_LISTED"
    assert len(calls) == 2
    body = next(tmp_path.glob("archive_jobs/*/zip_requests/*.body"))
    body.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="字节变化"):
        capture.capture_archive(tmp_path, item, client)


def test_unicode_frozen_symbol_is_preserved_and_http_path_is_encoded(tmp_path):
    item = task(symbol="哈基米USDT")
    assert "哈基米USDT" in item["key"]
    url = capture.archive_url(item["key"])
    assert url.isascii() and urllib.parse.unquote(url).endswith(item["key"])
    capture.Client(tmp_path, fixture_plan(item)).validate_url(url)
    for name in ("../BTCUSDT", "BTC/USDT", "BTC?USDT", "BTC#USDT"):
        with pytest.raises(ValueError, match="符号"):
            capture.archive_key("1m", name, "2026-08")


def test_retry_after_case_and_http_date_survive_cross_process_backoff(tmp_path):
    from datetime import UTC, datetime
    from email.utils import format_datetime

    item = task()
    deadline = capture.time.time() + 3600
    client = capture.Client(
        tmp_path,
        fixture_plan(item),
        rest=True,
        transport=lambda url, limit: (
            418,
            {"ReTrY-AfTeR": format_datetime(datetime.fromtimestamp(deadline, UTC))},
            b"limited",
        ),
    )
    query = urllib.parse.urlencode(
        {
            "symbol": item["symbol"],
            "startTime": capture.START_MS,
            "endTime": capture.END_MS - 1,
            "limit": 1000,
        }
    )
    with pytest.raises(RuntimeError, match="BACKOFF"):
        client.fetch(capture.FUNDING + "?" + query, tmp_path / "requests", 1000)
    backoff = capture.read(next(tmp_path.glob("backoffs/rest/*.json")))
    assert backoff["not_before_ms"] >= int(deadline * 1000) - 1000


def rest_fixture(tmp_path, monkeypatch):
    item = task()
    plan = fixture_plan(item)
    capture.save(tmp_path / "capture_plan.json", plan)
    events = [
        {
            "symbol": item["symbol"],
            "fundingTime": capture.START_MS + i,
            "fundingRate": "-0.0001000000",
            "markPrice": "100",
            "rateType": "Regular",
        }
        for i in range(1001)
    ]
    calls = []

    def transport(url, maximum):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        first, last = int(query["startTime"][0]), int(query["endTime"][0])
        calls.append((first, last))
        rows = [row for row in events if first <= row["fundingTime"] <= last][:1000]
        return 200, {}, capture.json.dumps(rows).encode()

    client = capture.Client(
        tmp_path, plan, rest=True, transport=transport, sleep=lambda seconds: None
    )
    original = capture.capture_pages
    monkeypatch.setattr(
        capture, "capture_pages", lambda *args, **kwargs: original(*args, **kwargs, max_pages=1)
    )
    interrupted = capture.capture_rest(tmp_path, item["symbol"], plan, client)
    monkeypatch.setattr(capture, "capture_pages", original)
    assert interrupted["status"] == "FAILED_UNKNOWN"
    return item, plan, client, calls


def test_rest_wrapper_resumes_good_checkpoint_and_keeps_uncommitted_orphan(tmp_path, monkeypatch):
    item, plan, client, calls = rest_fixture(tmp_path, monkeypatch)
    folder = tmp_path / "rest_jobs" / item["symbol"]
    orphan = folder / "checkpoint_0001.json"
    orphan.write_text('{"partial":')
    result = capture.capture_rest(tmp_path, item["symbol"], plan, client)
    assert result["status"] == "API_RANGE_VERIFIED" and result["rows"] == 1001
    assert len(calls) == 3 and calls[1] == (capture.START_MS + 999,) * 2
    assert result["uncommitted_checkpoint_files"] == {
        str(orphan.relative_to(tmp_path)): capture.sha(orphan)
    }
    assert capture.capture_rest(tmp_path, item["symbol"], plan, client) == result
    assert len(calls) == 3


@pytest.mark.parametrize("corrupt", ["raw", "checkpoint"])
def test_rest_wrapper_rejects_changed_committed_dependency_before_new_http(
    tmp_path, monkeypatch, corrupt
):
    item, plan, client, calls = rest_fixture(tmp_path, monkeypatch)
    folder = tmp_path / "rest_jobs" / item["symbol"]
    target = (
        next(folder.glob("requests/*.body"))
        if corrupt == "raw"
        else next(folder.glob("checkpoint_*.json"))
    )
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="字节"):
        capture.capture_rest(tmp_path, item["symbol"], plan, client)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "orphan,expected",
    [
        ("000000.body", "000001.body"),
        ("000007.tmp", "000008.body"),
        (".000007.body.1.2.3.tmp", "000008.body"),
    ],
)
def test_orphan_response_body_cannot_collide_with_next_request(tmp_path, orphan, expected):
    item = task()
    folder = tmp_path / "requests"
    folder.mkdir()
    (folder / orphan).write_bytes(b"old orphan")
    client = capture.Client(
        tmp_path, fixture_plan(item), transport=lambda url, maximum: (200, {}, b"new")
    )
    body, _ = client.fetch(capture.archive_url(item["key"]), folder, 1000)
    assert body.name == expected and body.read_bytes() == b"new"
    assert (folder / orphan).read_bytes() == b"old orphan"


def test_archive_rate_limit_skips_rest_phase(tmp_path, monkeypatch):
    item = task()
    plan = fixture_plan(item)
    plan["symbols"] = [item["symbol"]]
    monkeypatch.setattr(capture, "verify_runtime", lambda output: plan)

    def archive(output, task, client):
        client.stopped.set()
        return {"status": "FAILED_UNKNOWN"}

    monkeypatch.setattr(capture, "capture_archive", archive)
    monkeypatch.setattr(
        capture, "capture_rest", lambda *args: pytest.fail("限流后不应发起REST阶段")
    )
    capture.capture(tmp_path, archives=True, rest=True, workers=1)


@pytest.mark.parametrize("change", ["data", "plan_identity"])
def test_seal_rejects_changed_completed_price_dependency(tmp_path, monkeypatch, change):
    item = task()
    raw = zipped(item, [candle(capture.START_MS)])
    item["listed_bytes"] = len(raw)
    checksum = f"{hashlib.sha256(raw).hexdigest()}  {Path(item['key']).name}".encode()
    plan = fixture_plan(item)
    plan.update(symbols=[item["symbol"]], rest_symbols=[])
    capture.save(tmp_path / "capture_plan.json", plan)
    client = capture.Client(
        tmp_path,
        plan,
        transport=lambda url, maximum: (200, {}, checksum if url.endswith(".CHECKSUM") else raw),
    )
    record = capture.capture_archive(tmp_path, item, client)
    assert record["status"] == "VERIFIED"
    monkeypatch.setattr(capture, "verify_runtime", lambda output: plan)
    if change == "data":
        next(tmp_path.glob("archive_jobs/*/normalized_*.csv.gz")).write_bytes(b"corrupt")
    else:
        record["plan_sha256"] = "0" * 64
        capture.save(tmp_path / "archive_jobs" / item["id"] / "attempt_0001.json", record)
    with pytest.raises(ValueError, match="字节变化|身份"):
        capture.seal(tmp_path)
    assert not (tmp_path / "receipts.json").exists()


def test_unlisted_funding_and_empty_rest_are_unproven_instead_of_cash_zero(tmp_path):
    item = task(kind="fundingRate", listed=False)
    plan = fixture_plan(item)
    plan["symbols"] = [item["symbol"]]
    capture.save(tmp_path / "capture_plan.json", plan)
    capture.capture_archive(tmp_path, item, capture.Client(tmp_path, plan))
    capture.save(tmp_path / "rest_jobs" / item["symbol"] / "events.json", [])
    capture.save(
        tmp_path / "rest_jobs" / item["symbol"] / "attempt_0000.json",
        {
            "status": "API_EMPTY_RANGE",
            "events": "rest_jobs/" + item["symbol"] + "/events.json",
            "files": {},
        },
    )
    rows = capture.compare_all(tmp_path, plan)
    assert len(rows) == 2
    assert all(
        row["status"] == "UNPROVEN_MISSING_SOURCE" and row["regular_correspondence_proven"] is False
        for row in rows
    )


def test_unicode_checksum_member_normalization_and_resume_are_one_chain(tmp_path):
    item, calls = task(symbol="哈基米USDT"), []
    raw = zipped(item, [candle(capture.START_MS), candle(capture.START_MS + 60000)])
    item["listed_bytes"] = len(raw)
    checksum = f"{hashlib.sha256(raw).hexdigest()}  {Path(item['key']).name}".encode()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member = archive.infolist()[0]
        assert member.flag_bits & 0x800 and member.filename == "哈基米USDT-1m-2026-08.csv"

    def transport(url, maximum):
        assert url.isascii()
        calls.append(url)
        return 200, {}, checksum if url.endswith(".CHECKSUM") else raw

    capture.save(tmp_path / "capture_plan.json", fixture_plan(item))
    client = capture.Client(tmp_path, fixture_plan(item), transport=transport)
    result = capture.capture_archive(tmp_path, item, client)
    assert result["status"] == "VERIFIED" and result["profile"]["rows"] == 2
    normalized = next(tmp_path.glob("archive_jobs/*/normalized_*.csv.gz"))
    with gzip.open(normalized, "rt", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [*capture.COLUMNS, "completed_boundary_ms", "symbol"]
    assert all(row[-1] == "哈基米USDT" for row in rows[1:])
    assert capture.capture_archive(tmp_path, item, client) == result
    assert len(calls) == 2
    capture.verify_files(tmp_path, result["files"])


def test_checksum_utf8_is_strict_and_unicode_basename_must_match_exactly(tmp_path):
    item, raw, check = task(symbol="币安人生USDT"), tmp_path / "raw", tmp_path / "checksum"
    raw.write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    check.write_bytes(f"{digest}  哈基米USDT-1m-2026-08.zip".encode())
    with pytest.raises(ValueError, match="CHECKSUM"):
        capture.verified_checksum(raw, check, item)
    check.write_bytes(digest.encode() + b"  \xff.zip")
    with pytest.raises(UnicodeDecodeError):
        capture.verified_checksum(raw, check, item)
