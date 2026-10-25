"""核对H03精确时钟、零／空及含主效应的对照，合成数值非盈利证据。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.open_interest_flow import (
    FLOW,
    GROWTH,
    NAME,
    OI,
    components,
)
from tradingagents.research.quant.factors.position_tilt import FEATURES, MINUTE, TARGET
from tradingagents.research.quant.open_interest_flow_study import (
    BASELINE,
    COMMON,
    MODEL_SPECS,
    score,
)
from tradingagents.research.quant.position_tilt_study import ASSETS, CUTOFF, DAY, START, models


def source():
    hourly = pd.DataFrame({"clock_ms": np.array([120, 180, 240], dtype=np.int64) * MINUTE})
    hourly["requested_metrics_raw_T_ms"] = hourly.clock_ms - 10 * MINUTE
    hourly["raw_T_ms"] = hourly.requested_metrics_raw_T_ms
    hourly[OI] = [100.0, 200.0, 100.0]
    hourly[OI + "_value_status"] = "KNOWN_RAW_VALUE"
    index = pd.Index(np.arange(300, dtype=np.int64) * MINUTE, name="clock_ms")
    minute = pd.DataFrame({"quote_volume": 2.0, "taker_buy_quote_volume": 1.5}, index=index)
    for name in list(minute):
        minute[name + "_value_status"] = "KNOWN_RAW_VALUE"
    return hourly, minute


def test_exact_lagged_window_and_known_nonexpansion_zero():
    h, m = source()
    # u=170分钟，110分钟为左开边界，111分钟才属于窗口。
    m.loc[110 * MINUTE, "taker_buy_quote_volume"] = 0.0
    result = components(h, m)
    assert pd.isna(result.loc[0, GROWTH])
    assert result.loc[1, GROWTH] == pytest.approx(np.log(2))
    assert result.loc[1, FLOW] == pytest.approx(0.5)
    assert result.loc[1, NAME] == pytest.approx(np.log(2) / 2)
    assert result.loc[2, GROWTH] == 0.0
    assert result.loc[2, NAME] == 0.0
    m.loc[111 * MINUTE, "taker_buy_quote_volume"] = 0.0
    assert components(h, m).loc[1, FLOW] == pytest.approx(0.475)


def test_final_ten_minutes_and_future_oi_do_not_change_past_signal():
    h, m = source()
    before = components(h, m)
    m.loc[171 * MINUTE : 180 * MINUTE, "taker_buy_quote_volume"] = 0.0
    h.loc[2, OI] = 9999999.0
    after = components(h, m)
    pd.testing.assert_series_equal(before.loc[1], after.loc[1])


@pytest.mark.parametrize(
    "damage",
    ["missing_oi", "zero_oi", "negative_oi", "missing_quote", "zero_volume", "invalid_taker"],
)
def test_unknown_is_not_known_zero(damage):
    h, m = source()
    if damage == "missing_oi":
        h.loc[1, OI + "_value_status"] = "EMPTY"
    elif damage == "zero_oi":
        h.loc[1, OI] = 0.0
    elif damage == "negative_oi":
        h.loc[1, OI] = -1.0
    elif damage == "missing_quote":
        m.loc[150 * MINUTE, "quote_volume_value_status"] = "MISSING_CLOCK"
    elif damage == "zero_volume":
        m.loc[111 * MINUTE : 170 * MINUTE, ["quote_volume", "taker_buy_quote_volume"]] = 0.0
    else:
        m.loc[150 * MINUTE, "taker_buy_quote_volume"] = 3.0
    assert pd.isna(components(h, m).loc[1, NAME])


def test_missing_flow_is_unknown_even_when_oi_contracts():
    h, m = source()
    m.loc[220 * MINUTE, "quote_volume_value_status"] = "MISSING_CLOCK"
    result = components(h, m)
    assert result.loc[2, GROWTH] == 0.0
    assert pd.isna(result.loc[2, NAME])


@pytest.mark.parametrize("damage", ["hour_gap", "minute_gap", "minute_duplicate", "wrong_raw_time"])
def test_calendar_damage_rejected(damage):
    h, m = source()
    if damage == "hour_gap":
        h = h.drop(1)
    elif damage == "minute_gap":
        m = m.drop(150 * MINUTE)
    elif damage == "minute_duplicate":
        m = pd.concat([m, m.iloc[-1:]])
    else:
        h.loc[1, "raw_T_ms"] += MINUTE
    with pytest.raises(ValueError, match="日历"):
        components(h, m)


def test_primary_baseline_includes_both_main_effects_and_one_lambda():
    rng = np.random.default_rng(20261017)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame[FLOW], frame[GROWTH] = rng.uniform(-1, 1, 100), rng.uniform(0, 1, 100)
    frame[NAME] = frame[FLOW] * frame[GROWTH]
    frame[TARGET] = frame[NAME] * 0.01
    frame["clock_ms"] = np.arange(100, dtype=np.int64) * 3600000 + START
    frame["symbol"] = "SYNTHETIC"
    fits = models(
        frame, baseline_features=BASELINE, factor_name=NAME, extra_model_specs=MODEL_SPECS
    )
    assert len(fits) == 6
    assert fits["baseline_common"]["features"] == list(BASELINE)
    assert NAME not in fits["baseline_common"]["features"]
    assert fits["augmented_common"]["features"] == list(COMMON)
    assert fits["controls_common"]["features"] == list(FEATURES)
    assert fits["augmented_common"] == fits["augmented_own"]
    assert all(len(f["fits"]) == 1 and f["fits"][0]["penalty"] == 1 for f in fits.values())
    assert fits["augmented_common"]["fits"][0]["coefficients"][NAME] > 0


def test_model_name_override_rejected():
    with pytest.raises(ValueError, match="覆盖主对照"):
        models(pd.DataFrame(), extra_model_specs=(("baseline_common", FEATURES, FEATURES),))


def test_h03_full_score_uses_native_concentration_gate_and_reports_all_models():
    rng = np.random.default_rng(20261017)
    rows = []
    for asset in ASSETS:
        for clock in [START + i * 3600000 for i in range(24)] + [
            CUTOFF + i * DAY for i in range(10)
        ]:
            for symbol in range(20):
                row = dict(zip(FEATURES, rng.normal(size=len(FEATURES)), strict=True))
                row[FLOW], row[GROWTH] = rng.uniform(-1, 1), rng.uniform(0, 1)
                row[NAME] = row[FLOW] * row[GROWTH]
                row[TARGET] = row[NAME] * 0.02 + rng.normal(scale=0.001)
                row.update(
                    symbol=f"S{symbol:02}",
                    clock_ms=clock,
                    asset_class=asset,
                    label_path_complete=True,
                )
                rows.append(row)
    predictions, fits, summary, tests, diagnostics = score(pd.DataFrame(rows))
    assert len(predictions) == 400
    assert len(tests) == 2
    for row in tests:
        checks = row["mechanism_checks"]
        assert "positive_without_largest_contributor" in checks
        assert "positive_quintile_spread" not in checks
        assert row["bh_family_size"] == 2
    for asset in ASSETS:
        assert len(fits[asset]) == len(summary[asset]["all_model_errors"]) == 6
        assert "controls_common" in diagnostics[asset]
        assert fits[asset]["augmented_common"] == fits[asset]["augmented_own"]
