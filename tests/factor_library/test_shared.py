"""Cross-asset calendar, as-of news and no-lookahead invariants."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from factors.common.panel import event_features
from factors.equity import calculate
from factors.macro.news import parse
from research_tools.factor_lab.compose import validate
from research_tools.factor_lab.io import read
from research_tools.factor_lab.metrics import prediction_metrics


def markets(days=800):
    clocks = pd.date_range("2021-01-01", periods=days, freq="B", tz="UTC")
    result = {}
    for j, s in enumerate(["A", "B", "C", "D"]):
        c = 100 + np.arange(days) * (0.03 + j * 0.02) + np.sin(np.arange(days) * (0.1 + j * 0.03))
        result[s] = pd.DataFrame(
            {
                "open": c,
                "high": c + 1,
                "low": c - 1,
                "close": c + 0.1,
                "volume": 1000 + np.arange(days),
                "quote_volume": c * 1000,
            },
            index=clocks,
        )
    return result


def test_equities_use_shared_factors_without_crypto_funding_or_calendar_fill():
    source = markets()
    out, raw, catalog = calculate(source, benchmark_symbol="A")
    assert len(catalog) == 69 and len(out["A"]) == 800
    assert out["A"].index.equals(source["A"].index)
    assert "rsi14" in out["A"] and "benchmark_mom_24" in out["A"]
    assert not any(k.startswith("funding_") for k in out["A"])
    assert out["A"]["policy_rate"].isna().all()
    assert not any(out["A"].index.dayofweek >= 5)
    assert all("equity" in c["asset_classes"] for c in catalog)


def test_complete_shared_feature_prefix_is_unchanged_by_future_rows():
    source = markets()
    a, _, _ = calculate(source, benchmark_symbol="A")
    short = {s: frame.iloc[:760] for s, frame in source.items()}
    b, _, _ = calculate(short, benchmark_symbol="A")
    for s in source:
        pd.testing.assert_frame_equal(a[s].iloc[:760], b[s], rtol=1e-10, atol=1e-10)


def test_independent_threads_do_not_share_mutable_factor_catalog():
    source = markets(750)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda _: calculate(source, benchmark_symbol="A")[2], range(2)))
    assert all(len(x) == 69 for x in rows)


def test_official_news_available_only_after_declared_latency():
    html = """<p class="article__time">March 20, 2024</p>For release at 2:00 p.m. EDT
    <div class="col-xs-12 col-sm-8 col-md-8"><p>The Committee maintains the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. Inflation remains elevated.</p>Voting for</div>"""
    event = parse(html)
    event["policy_rate_change"] = 0
    assert event["published_at"] == "2024-03-20T18:00:00+00:00"
    clocks = pd.DatetimeIndex(["2024-03-20T18:04Z", "2024-03-20T18:05Z"])
    f = event_features(clocks, [event])
    assert pd.isna(f.policy_rate.iloc[0]) and f.policy_rate.iloc[1] == 5.375


def test_ui_rejects_nonfinite_unknown_factor_and_excess_risk():
    plan = read("research_tools/factor_lab/plan.json")
    for payload in [
        {"preset": "custom", "weights": {"x": float("nan")}, "window": "2025Q1"},
        {"preset": "custom", "weights": {"alien": 1}, "window": "2025Q1"},
        {"preset": "full", "window": "2025Q1", "gross": 2},
    ]:
        with pytest.raises(ValueError):
            validate(payload, plan, ["x"])


def test_prediction_rank_ic_and_auc_match_known_ordering():
    clocks = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    index = pd.MultiIndex.from_product([clocks, ["A", "B", "C", "D"]], names=["time", "symbol"])
    label = pd.Series([-2, -1, 1, 2] * 3, index=index, dtype=float)
    m = prediction_metrics(label, label)
    assert m["mse"] == 0 and m["rank_ic"] == 1 and m["auc"] == 1 and m["direction_accuracy"] == 1
