"""量化评分必须隔离未来标签、保留常量未知并拒绝事后改组。"""

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.regression import (
    FEATURES,
    bh_adjust,
    correlation,
    cross_section,
    diagnostic_spread,
    evaluate,
    hac_mean,
)
from tradingagents.research.quant.regression_run import (
    MINUTE,
    features_from_minutes,
    label_phase,
    ms,
    validated_rows,
)


def test_future_minutes_cannot_change_current_factors_but_do_change_the_separate_label():
    clock = np.arange(1, 201, dtype=np.int64) * MINUTE
    raw = pd.DataFrame(
        {
            "open": np.exp(np.arange(200) / 1000),
            "close": np.exp(np.arange(200) / 1000),
            "quote_volume": 100.0,
            "taker_buy_quote": 55.0,
        },
        index=clock,
    )
    original = features_from_minutes(raw)
    changed = raw.copy()
    changed.loc[clock[110] :, ["open", "close"]] *= 2
    later = features_from_minutes(changed)
    assert np.allclose(
        original.loc[clock[100], list(FEATURES)], later.loc[clock[100], list(FEATURES)]
    )
    assert original.loc[clock[100], "target_20"] != later.loc[clock[100], "target_20"]
    expected = raw.loc[clock[122], "open"] / raw.loc[clock[102], "open"] - 1
    assert math.isclose(original.loc[clock[100], "target_20"], expected)
    raw.loc[clock[80], "close"] = np.nan
    assert features_from_minutes(raw).loc[clock[100], list(FEATURES)].isna().all()


def test_training_target_cannot_cross_the_predeclared_validation_boundary():
    boundary = ms("2026-10-06T00:00:00Z")
    assert label_phase(boundary - 60 * MINUTE, 60) == "excluded_boundary"
    assert label_phase(boundary - 60 * MINUTE, 20) == "training"
    assert label_phase(boundary, 20) == "retrospective_validation"


def artificial_cross_section():
    generator = np.random.default_rng(1729)
    raw = pd.DataFrame(generator.normal(size=(40, len(FEATURES))), columns=FEATURES)
    raw["symbol"] = [f"合约{i:03}" for i in range(40)]
    raw["target"] = 0.01 * raw["return_3m"]
    return raw


def test_regression_recovers_predictive_factor_and_reports_rank_deficiency_without_dropping_columns():
    frame = artificial_cross_section()
    value, _, _ = cross_section(frame, "target")
    assert value["factor_statistics"]["return_3m"]["rank_ic"] > 0.999
    assert value["multivariate"]["return_3m"] > 0
    frame["flow_12m"] = frame["flow_3m"]
    value, _, _ = cross_section(frame, "target")
    assert (
        value["multivariate"] is None
        and value["multivariate_failure"] == "RANK_DEFICIENT_FIXED_DESIGN"
    )


def test_selected_missing_future_label_keeps_the_entire_diagnostic_spread_unknown():
    raw = artificial_cross_section()
    _, known, ranks = cross_section(raw, "target")
    first = diagnostic_spread(known, ranks["return_3m"], "target", 1)
    selected = known["return_3m"].idxmax()
    raw.loc[selected, "target"] = np.nan
    _, new_known, new_ranks = cross_section(raw, "target")
    assert ranks.equals(new_ranks)
    assert (
        diagnostic_spread(new_known, new_ranks["return_3m"], "target", 1)["reason"]
        == "SELECTED_FUTURE_TARGET_MISSING"
    )
    assert first["gross"] is not None


def test_hac_mean_matches_direct_bartlett_sum_and_missing_days_do_not_compress_time():
    x = np.array([0.1, 0.2, -0.4, 0.3, -0.2, 0.1, 0.4])
    result = hac_mean(x)
    errors = x - x.mean()
    expected = (
        errors @ errors
        + sum(2 * (1 - lag / 4) * (errors[lag:] @ errors[:-lag]) for lag in (1, 2, 3))
    ) / len(x) ** 2
    assert math.isclose(result["standard_error"] ** 2, expected)
    gap = hac_mean([0.1, np.nan, 0.2, -0.4, 0.3, -0.2, 0.1, 0.4])
    assert gap["days"] == result["days"]
    assert gap["standard_error"] != result["standard_error"]
    assert hac_mean([0.1])["t"] is None


def test_multiple_testing_and_constant_values_keep_their_real_meaning():
    assert bh_adjust([0.01, 0.04, 0.03, None]) == [0.03, 0.04, 0.04, None]
    assert correlation([1, 1, 1], [1, 2, 3], rank=True) is None
    assert correlation([1, 2, 3], [1, 2, 3], rank=True) == 1


def test_validation_outcomes_cannot_refit_training_coefficients_or_factor_directions():
    frames = []
    for day, phase in (("2026-10-01", "training"), ("2026-10-02", "validation")):
        for hour in range(4):
            frame = artificial_cross_section()
            frame["clock_ms"] = ms(day + "T00:00:00Z") + hour * 60 * MINUTE
            frame["asset_class"] = "人工资产"
            for horizon in (20, 60):
                frame[f"target_{horizon}"] = frame["target"]
                frame[f"phase_{horizon}"] = phase
            frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    protocol = {
        "phases": {
            "training": {"start": "2026-10-01", "end": "2026-10-02"},
            "validation": {"start": "2026-10-02", "end": "2026-10-03"},
        }
    }
    original = evaluate(samples, protocol)
    changed = samples.copy()
    changed.loc[changed["phase_20"] == "validation", ["target_20", "target_60"]] *= -10
    later = evaluate(changed, protocol)
    for key, group in original["groups"].items():
        assert (
            group["training_mean_coefficients"]
            == later["groups"][key]["training_mean_coefficients"]
        )
        for factor, statistics in group["phases"]["training"]["factors"].items():
            assert (
                statistics["training_direction"]
                == later["groups"][key]["phases"]["validation"]["factors"][factor][
                    "training_direction"
                ]
            )


@pytest.mark.parametrize(
    "failure", ["duplicate", "future_boundary", "negative_volume", "inconsistent_buy_volume"]
)
def test_raw_minute_integrity_cannot_be_silently_repaired(failure):
    row = [0, "100", "101", "99", "100", "1", 59999, "100", 1, "0.5", "50", "0"]
    rows = [row]
    if failure == "duplicate":
        rows.append(row.copy())
    elif failure == "future_boundary":
        row[6] += 1
    elif failure == "negative_volume":
        row[5] = "-1"
    else:
        row[10] = "101"
    with pytest.raises(ValueError):
        validated_rows({"symbol": "A", "rows": rows}, "A", set())
