"""Trading-state availability, future perturbations, missingness and event clocks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factors.common.market_states import classify_market, event_risk

PARAMETERS = json.loads(
    (Path(__file__).resolve().parents[2] / "research_tools/regime_program/plan.json").read_text()
)["state_parameters"]


def samples():
    rng = np.random.default_rng(23)
    index = pd.date_range("2023-01-01", periods=550, freq="D", tz="UTC")
    result = {}
    for symbol in ["BTCUSDT"] + [f"A{i}" for i in range(11)]:
        price = 100 * np.exp(rng.normal(0.0002, 0.02, len(index)).cumsum())
        result[symbol] = pd.DataFrame({"close": price, "quote_volume": 1e8}, index=index)
    return result


def test_appending_future_and_mutating_current_day_cannot_reclassify_past():
    frames = samples()
    full, eligible = classify_market(frames, PARAMETERS)
    partial, partial_eligible = classify_market(
        {s: f.iloc[:450] for s, f in frames.items()}, PARAMETERS
    )
    pd.testing.assert_frame_equal(full.iloc[:450], partial)
    pd.testing.assert_frame_equal(eligible.iloc[:450], partial_eligible)
    changed = {s: f.copy() for s, f in frames.items()}
    for f in changed.values():
        f.iloc[450:, 0] *= 1000
        f.iloc[450:, 1] = 0
    perturbed, _ = classify_market(changed, PARAMETERS)
    pd.testing.assert_frame_equal(full.iloc[:451], perturbed.iloc[:451])
    assert full.state.iloc[:100].eq("unknown").all()
    assert full.state.iloc[300:].ne("unknown").all()


def test_observed_equity_calendar_is_not_filled_with_fake_weekend_decisions():
    frames = {s: f.loc[f.index.dayofweek < 5] for s, f in samples().items()}
    result, _ = classify_market(frames, PARAMETERS)
    assert result.index.equals(frames["BTCUSDT"].index)
    assert not (result.index.dayofweek >= 5).any()


def test_liquidity_excludes_low_volume_names_and_does_not_invent_state_confidence():
    frames = samples()
    for s in sorted(frames)[:5]:
        frames[s].loc[:, "quote_volume"] = 0
    result, eligible = classify_market(frames, PARAMETERS)
    assert eligible.iloc[-1].sum() == 7
    assert result.state.iloc[-1] == "liquidity_stress"
    for f in frames.values():
        f.loc[:, "quote_volume"] = np.nan
    unknown, _ = classify_market(frames, PARAMETERS)
    assert unknown.state.iloc[-1] == "unknown"


def test_event_window_starts_at_available_clock_and_expires_without_anticipation():
    times = pd.date_range("2025-01-29 18:00", "2025-01-30 20:00", freq="5min", tz="UTC")
    events = [
        {
            "published_at": "2025-01-29T19:00:00+00:00",
            "available_at": "2025-01-29T19:05:00+00:00",
            "source_kind": "official_fomc_release",
        }
    ]
    risk = event_risk(times, events)
    assert not risk.loc[:"2025-01-29 19:00"].any()
    assert risk.loc["2025-01-29 19:05":"2025-01-30 19:00"].all()
    assert not risk.loc["2025-01-30 19:05":].any()
    events[0]["available_at"] = "2025-01-29T18:00:00+00:00"
    with pytest.raises(ValueError, match="causal"):
        event_risk(times, events)


def test_ambiguous_or_duplicate_decision_calendars_are_rejected():
    frames = samples()
    frames["BTCUSDT"].index = frames["BTCUSDT"].index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        classify_market(frames, PARAMETERS)
