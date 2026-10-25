"""因子诊断须保留同值、日历缺口和标签缺失，不能制造交易收益。"""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factor_diagnostics import (
    HOUR,
    VERSION,
    assign_memberships,
    evaluate,
)

START = 1_791_244_800_000


def panel(clocks=(START, START + HOUR), *, offset=0):
    rows = []
    for clock in clocks:
        for member in range(20):
            rows.append(
                {
                    "asset_class": "CRYPTO",
                    "symbol": f"S{member:02}",
                    "clock_ms": clock + offset,
                    "score": float(member),
                    "control": 1.0,
                    "target": member / 10_000,
                }
            )
    return pd.DataFrame(rows)


def run(frame, calendar=(START, START + HOUR), **kwargs):
    return evaluate(
        frame,
        score_column="score",
        target_column="target",
        calendar_ms=calendar,
        scope="retrospective_validation",
        **kwargs,
    )


def test_distinct_scores_form_five_layers_with_raw_and_demeaned_returns_separate():
    frame = panel()
    result = run(frame)
    row = result["sections"][0]
    assert row["quantile_counts"] == {str(q): 4 for q in range(1, 6)}
    assert row["quantiles"]["1"]["raw_mean_return"] == pytest.approx(0.00015)
    assert row["quantiles"]["5"]["raw_mean_return"] == pytest.approx(0.00175)
    assert row["quantiles"]["1"]["cross_section_demeaned_mean_return"] == pytest.approx(-0.0008)
    assert row["spreads"]["raw_mean_return"]["q5_minus_q1"] == pytest.approx(0.0016)
    assert row["spreads"]["raw_mean_return"]["half_gross_long_short_diagnostic"] == pytest.approx(
        0.0008
    )
    assert result["schema"] == VERSION
    assert result["returns_are_diagnostic_labels_not_account_equity"]
    assert not result["costs_or_funding_deducted"]
    assert not result["economic_gate_changed"] and not result["production_eligible"]
    assert result["summary"]["CRYPTO"]["quantile_returns"]["5"]["raw_mean_return"][
        "mean"
    ] == pytest.approx(0.00175)
    assert result["summary"]["CRYPTO"]["persistence"][str(HOUR)]["rank_autocorrelation"][
        "mean"
    ] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_future_targets_and_input_row_order_cannot_change_tied_memberships():
    frame = panel()
    frame.loc[frame.score.between(3, 14), "score"] = 7.0
    before, _ = assign_memberships(frame, score_column="score", calendar_ms=(START, START + HOUR))
    frame["target"] = np.nan
    after, _ = assign_memberships(
        frame.drop(columns="target").sample(frac=1, random_state=8),
        score_column="score",
        calendar_ms=(START, START + HOUR),
    )
    pd.testing.assert_frame_equal(before, after)
    assert before.loc[before.score == 7, "quantile"].nunique() == 1


def test_all_zero_scores_stay_in_middle_bucket_without_symbol_selected_tail_profit():
    frame = panel()
    frame["score"] = 0.0
    result = run(frame)
    for row in result["sections"]:
        assert row["status"] == "CONSTANT_SCORE"
        assert row["quantile_counts"] == {"1": 0, "2": 0, "3": 20, "4": 0, "5": 0}
        assert row["rank_ic"] is row["pearson_ic"] is None
        assert row["spreads"]["raw_mean_return"]["q5_minus_q1"] is None
        assert row["quantiles"]["5"]["status"] == "EMPTY_QUANTILE"
    prior = result["sections"][1]["persistence"][str(HOUR)]
    assert prior["rank_autocorrelation"] is None
    assert prior["status"] == "CONSTANT_INTERSECTION_RANKS"
    assert prior["quantile_turnover"]["3"]["entry_fraction"] == 0


def test_missing_middle_label_does_not_reform_quantiles_or_invent_universe_demeaning():
    frame = panel()
    frame.loc[(frame.clock_ms == START) & (frame.symbol == "S09"), "target"] = np.nan
    row = run(frame)["sections"][0]
    assert row["common_feature_members"] == 20 and row["label_known_members"] == 19
    assert row["quantile_counts"] == {str(q): 4 for q in range(1, 6)}
    assert row["quantiles"]["3"]["status"] == "SELECTED_TARGET_MISSING"
    assert row["universe_raw_mean"] is None
    assert row["spreads"]["raw_mean_return"]["q5_minus_q1"] is not None
    assert row["spreads"]["cross_section_demeaned_mean_return"]["q5_minus_q1"] is None
    assert row["rank_ic"] is None


def test_missing_tail_label_makes_the_complete_tail_return_unknown():
    frame = panel()
    frame.loc[frame.symbol == "S19", "target"] = np.nan
    row = run(frame)["sections"][0]
    assert row["quantiles"]["5"]["members"] == 4
    assert row["quantiles"]["5"]["known_labels"] == 3
    assert row["quantiles"]["5"]["raw_mean_return"] is None
    assert row["spreads"]["raw_mean_return"]["q5_minus_q1"] is None


def test_explicit_common_features_do_not_shrink_on_partial_unknown_model_score():
    frame = panel()
    frame.loc[0, "score"] = np.nan
    row = run(frame, common_features=("control",))["sections"][0]
    assert row["status"] == "INCOMPLETE_SCORE_ON_COMMON_POPULATION"
    assert row["common_feature_members"] == 20 and row["finite_score_members"] == 19
    assert sum(row["quantile_counts"].values()) == 0
    assert row["rank_ic"] is None


@pytest.mark.parametrize("offset", [0, 5 * 60_000])
def test_exact_lag_turnover_and_rank_autocorrelation_support_both_hour_offsets(offset):
    clocks = (START, START + HOUR, START + 2 * HOUR)
    frame = panel(clocks, offset=offset)
    frame.loc[frame.clock_ms != START + offset, "score"] *= -1
    result = run(frame, tuple(clock + offset for clock in clocks))
    first = result["sections"][1]["persistence"][str(HOUR)]
    second = result["sections"][2]["persistence"][str(HOUR)]
    assert first["rank_autocorrelation"] == pytest.approx(-1)
    assert first["quantile_turnover"]["5"]["entry_fraction"] == 1
    assert second["rank_autocorrelation"] == pytest.approx(1)
    assert second["quantile_turnover"]["5"]["entry_fraction"] == 0


def test_calendar_gap_is_not_compressed_and_longer_exact_lag_can_still_be_known():
    clocks = (START, START + HOUR, START + 2 * HOUR)
    frame = panel((START, START + 2 * HOUR))
    result = run(frame, clocks, lag_ms=(HOUR, 2 * HOUR))
    assert result["sections"][1]["status"] == "MISSING_SECTION"
    short = result["sections"][2]["persistence"][str(HOUR)]
    longer = result["sections"][2]["persistence"][str(2 * HOUR)]
    assert short["status"] == "UNAVAILABLE_MEMBERSHIP"
    assert short["quantile_turnover"]["5"]["entry_fraction"] is None
    assert longer["rank_autocorrelation"] == pytest.approx(1)
    assert longer["quantile_turnover"]["5"]["entry_fraction"] == 0


def test_icir_period_is_explicit_and_daily_aggregation_is_not_hourly_annualization():
    frame = panel()
    frame.loc[frame.clock_ms != START, "target"] *= -1
    info = run(frame)["summary"]["CRYPTO"]["information"]["rank_ic"]
    assert info["period"]["valid_periods"] == 2
    assert info["period"]["icir_unannualized"] == pytest.approx(0)
    assert info["period"]["positive_fraction"] == 0.5
    assert info["period"]["period_ms"] == HOUR
    assert info["daily"]["valid_periods"] == 1
    assert info["daily"]["icir_unannualized"] is None
    assert not info["period"]["annualization_applied"]


def test_direction_is_fixed_and_validation_cannot_choose_a_profitable_sign():
    frame = panel()
    positive, negative = run(frame), run(frame, direction=-1)
    assert positive["sections"][0]["rank_ic"] == pytest.approx(1)
    assert negative["sections"][0]["rank_ic"] == pytest.approx(-1)
    assert negative["sections"][0]["spreads"]["raw_mean_return"]["q5_minus_q1"] < 0
    assert negative["protocol"]["direction"] == -1


def test_population_churn_is_separate_from_reranking_and_future_targets():
    frame = panel()
    mask = (frame.clock_ms != START) & (frame.symbol == "S19")
    frame.loc[mask, "symbol"] = "NEW"
    value = run(frame)["sections"][1]["persistence"][str(HOUR)]
    assert value["quantile_turnover"]["5"]["entry_fraction"] == 0.25
    assert value["quantile_turnover"]["5"]["entries_from_new_population"] == 1
    assert value["quantile_turnover"]["5"]["entries_from_reranking"] == 0
    frame["target"] *= -1000
    assert run(frame)["sections"][1]["persistence"] == {str(HOUR): value}


def test_declared_missing_asset_and_unknown_periods_remain_visible():
    result = run(panel(), asset_classes=("CRYPTO", "MISSING"))
    missing = [row for row in result["sections"] if row["asset_class"] == "MISSING"]
    assert len(missing) == 2 and all(row["status"] == "MISSING_SECTION" for row in missing)
    assert result["summary"]["MISSING"]["information"]["rank_ic"]["period"]["valid_periods"] == 0


def test_entirely_empty_input_retains_declared_groups_and_calendar():
    frame = pd.DataFrame(columns=["asset_class", "clock_ms", "symbol", "score", "target"])
    result = run(frame, asset_classes=("CRYPTO", "TRADIFI"))
    assert len(result["sections"]) == 4
    assert all(row["status"] == "MISSING_SECTION" for row in result["sections"])
    json.dumps(result, allow_nan=False)


def test_unknown_and_infinite_inputs_remain_unknown_without_nonfinite_json():
    frame = panel()
    frame["target"] = np.inf
    result = run(frame)
    assert all(row["rank_ic"] is None for row in result["sections"])
    assert all(row["quantiles"]["5"]["raw_mean_return"] is None for row in result["sections"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "float_clock",
        "float_calendar",
        "gap_calendar",
        "zero_frequency",
        "bad_lag",
        "future_feature",
        "boolean_score",
        "unknown_asset",
        "no_scope",
    ],
)
def test_invalid_or_ambiguous_contracts_fail_explicitly(failure):
    frame, calendar, kwargs = panel(), (START, START + HOUR), {}
    if failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif failure == "float_clock":
        frame["clock_ms"] = frame.clock_ms.astype(float)
    elif failure == "float_calendar":
        calendar = tuple(map(float, calendar))
    elif failure == "gap_calendar":
        calendar = (START, START + 2 * HOUR)
    elif failure == "zero_frequency":
        kwargs["frequency_ms"] = 0
    elif failure == "bad_lag":
        kwargs["lag_ms"] = (HOUR + 1,)
    elif failure == "future_feature":
        kwargs["common_features"] = ("target",)
    elif failure == "boolean_score":
        frame["score"] = True
    elif failure == "unknown_asset":
        kwargs["asset_classes"] = ("OTHER",)
    else:
        with pytest.raises(ValueError, match="阶段范围"):
            evaluate(
                frame, score_column="score", target_column="target", calendar_ms=calendar, scope=""
            )
        return
    with pytest.raises(ValueError):
        run(frame, calendar, **deepcopy(kwargs))
