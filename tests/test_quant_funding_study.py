"""普通结算用于预测，股息调整仍留在真实现金账本，缺覆盖不能补成零。"""

import pytest

from tradingagents.research.quant.artifacts import save, sha
from tradingagents.research.quant.backtesting.search import load_funding
from tradingagents.research.quant.funding_study import regular_events


def fixture(root, kind="Special"):
    raw = [
        {
            "symbol": "PUBLIC",
            "fundingTime": 1000,
            "fundingRate": "0.0",
            "markPrice": "100",
            "rateType": "Regular",
        },
        {
            "symbol": "PUBLIC",
            "fundingTime": 2000,
            "fundingRate": "-0.005",
            "markPrice": "100",
            "rateType": kind,
        },
    ]
    path = root / "funding_raw/0-9999.json"
    path.parent.mkdir(parents=True)
    save(path, raw)
    capture = {"start": "1970-01-01T00:00:00Z", "end": "1970-01-01T00:00:10Z"}
    manifest = {
        "api_time_range_coverage_verified": True,
        "leaves": [
            {
                "start_ms": 0,
                "end_ms": 9999,
                "path": f"{root.name}/funding_raw/0-9999.json",
                "sha256": sha(path),
            }
        ],
    }
    save(root / "funding_events.json", raw)
    save(root / "funding_manifest.json", manifest)
    return capture, manifest


def test_special_cash_is_retained_but_cannot_become_a_regular_carry_predictor(tmp_path):
    capture, manifest = fixture(tmp_path)
    ordinary, counts = regular_events(tmp_path, capture, manifest)
    assert counts == {"Regular": 1, "Special": 1}
    assert len(ordinary) == 1 and ordinary[0]["rate"] == 0
    cash = load_funding(tmp_path)
    assert len(cash) == 2 and cash[1]["rate"] == -0.005


def test_unknown_rate_type_does_not_silently_become_ordinary_carry(tmp_path):
    capture, manifest = fixture(tmp_path, "Unknown")
    with pytest.raises(ValueError, match="事件类型"):
        regular_events(tmp_path, capture, manifest)


def test_request_time_coverage_is_checked_independently_of_the_manifest_flag(tmp_path):
    capture, manifest = fixture(tmp_path)
    capture["end"] = "1970-01-01T00:00:11Z"
    with pytest.raises(ValueError, match="完整覆盖"):
        regular_events(tmp_path, capture, manifest)
