"""历史资金费身份依赖显式根目录和字节，不能依赖旧研究工作目录。"""

import pytest

from tradingagents.research.quant.artifacts import save, sha
from tradingagents.research.quant.backtesting.search import funding_source_path, load_funding


def test_migrated_funding_source_replays_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "tradingagents" / "public_v001"
    raw = root / "funding_raw/0-9999.json"
    raw.parent.mkdir(parents=True)
    events = [{"symbol": "PUBLIC", "fundingTime": 1234, "fundingRate": "0.001", "markPrice": "100"}]
    save(raw, events)
    leaf = {
        "path": "artifacts/prior/public_v001/funding_raw/0-9999.json",
        "start_ms": 0,
        "end_ms": 9999,
        "sha256": sha(raw),
    }
    save(
        root / "funding_manifest.json", {"api_time_range_coverage_verified": True, "leaves": [leaf]}
    )
    save(root / "funding_events.json", events)
    monkeypatch.chdir(tmp_path)
    value = load_funding(root)
    assert value[0]["rate"] == 0.001 and value[0]["time_ms"] == 1234
    assert funding_source_path(root, leaf) == raw


@pytest.mark.parametrize(
    "path",
    [
        "../public_v001/funding_raw/0-9999.json",
        "/foreign/funding_raw/0-9999.json",
        "public_v001/funding_raw/1-2.json",
    ],
)
def test_paths_cannot_silently_switch_root_or_declared_time_interval(tmp_path, path):
    with pytest.raises(ValueError):
        funding_source_path(tmp_path / "public_v001", {"path": path, "start_ms": 0, "end_ms": 9999})
