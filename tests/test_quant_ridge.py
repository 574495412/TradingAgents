"""用独立增广最小二乘和时间扰动验证岭回归，而非比较自生成系数。"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.ridge import DAY_MS, PENALTIES, fit_path, predict, select_penalty


def samples():
    """共线和常量同时存在，部分标签未知但特征仍有效。"""
    rng = np.random.default_rng(824)
    x = rng.normal(size=80)
    frame = pd.DataFrame({"clock_ms": np.arange(80), "x": x, "duplicate": 2 * x, "flat": 7.0})
    frame["target_y"] = 3.1 + 0.7 * x + rng.normal(size=80) * 0.02
    frame.loc[[2, 8, 12], "target_y"] = np.nan
    return frame


def fitted(frame, **kwargs):
    """使用明确的成熟时钟，避免测试本身偷用未来行。"""
    return fit_path(
        frame,
        features=("x", "duplicate", "flat"),
        target="target_y",
        refit_ms=100,
        training_start_ms=0,
        label_delay_ms=2,
        **kwargs,
    )


@pytest.mark.parametrize("penalty", PENALTIES)
def test_coefficients_match_independent_augmented_least_squares(penalty):
    frame = samples()
    path = fitted(frame, penalties=(penalty,))
    fit = path["fits"][0]
    assert fit["status"] == "READY"
    columns = path["features"]
    complete_y = np.isfinite(frame.target_y)
    x = frame.loc[complete_y, columns].to_numpy()
    x = (x - [path["means"][f] for f in columns]) / [path["scales"][f] for f in columns]
    y = frame.loc[complete_y, "target_y"].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    # 参照直接增广带截距的设计；惩罚块截距列为零，走SVD而非实现中的正规方程。
    extra = np.column_stack([np.zeros(len(columns)), np.sqrt(penalty) * np.eye(len(columns))])
    augmented = np.vstack([design / np.sqrt(len(x)), extra])
    target = np.r_[y / np.sqrt(len(y)), np.zeros(len(columns))]
    expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
    actual = np.r_[fit["intercept"], [fit["coefficients"][f] for f in columns]]
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-9)
    assert fit["coefficients"]["flat"] == 0
    assert path["scales"]["flat"] == 1


def test_average_loss_penalty_is_invariant_to_repeating_every_observation():
    frame = samples()
    path = fitted(frame)
    repeated = fitted(pd.concat([frame, frame], ignore_index=True))
    for left, right in zip(path["fits"], repeated["fits"], strict=True):
        np.testing.assert_allclose(
            list(left["coefficients"].values()), list(right["coefficients"].values()), atol=2e-10
        )
        assert left["intercept"] == pytest.approx(right["intercept"], abs=2e-12)


def test_label_translation_only_changes_unpenalized_intercept():
    frame = samples()
    original = fitted(frame, penalties=(100.0,))
    shifted = frame.copy()
    shifted.target_y += 1234
    updated = fitted(shifted, penalties=(100.0,))
    assert updated["fits"][0]["intercept"] - original["fits"][0]["intercept"] == pytest.approx(1234)
    np.testing.assert_allclose(
        list(updated["fits"][0]["coefficients"].values()),
        list(original["fits"][0]["coefficients"].values()),
        atol=2e-13,
    )


def test_scaler_population_precedes_missing_target_filter_and_is_strictly_mature():
    frame = samples()
    frame.loc[2, "x"] = 1000
    path = fitted(frame)
    assert path["means"]["x"] == pytest.approx(frame.x.mean())
    assert path["common_feature_rows"] == 80
    assert path["known_target_rows"] == 77
    extra = pd.DataFrame(
        {
            "clock_ms": [98, 99, 110],
            "x": [1e20] * 3,
            "duplicate": [1e30] * 3,
            "flat": [99] * 3,
            "target_y": [1e25] * 3,
        }
    )
    with_future = fitted(pd.concat([frame, extra], ignore_index=True))
    assert with_future == path
    assert path["maximum_scaler_label_end_ms"] == 81


def test_shared_population_is_preserved_for_a_smaller_model():
    frame = samples()
    frame["extra"] = 1.0
    frame.loc[0, "extra"] = np.nan
    path = fit_path(
        frame,
        features=("x",),
        common_features=("x", "extra"),
        target="target_y",
        refit_ms=100,
        training_start_ms=0,
        label_delay_ms=2,
    )
    assert path["common_feature_rows"] == 79
    assert path["means"]["x"] == pytest.approx(frame.x.iloc[1:].mean())
    values = predict(frame, path, PENALTIES[0])
    assert np.isnan(values.iloc[0])
    assert values.iloc[1:].notna().all()
    changed = frame.copy()
    changed.target_y = np.nan
    pd.testing.assert_series_equal(values, predict(changed, path, PENALTIES[0]))


def test_training_constant_stays_zero_coefficient_when_future_feature_varies():
    frame = samples()
    path = fitted(frame)
    before = predict(frame, path, PENALTIES[0])
    future = frame.copy()
    future.flat = np.linspace(-1e100, 1e100, len(future))
    pd.testing.assert_series_equal(before, predict(future, path, PENALTIES[0]))


def test_training_failure_is_explicit_and_serializable():
    frame = samples().iloc[:2]
    path = fitted(frame)
    assert {fit["status"] for fit in path["fits"]} == {"INSUFFICIENT_TRAINING_ROWS"}
    assert predict(frame, path, PENALTIES[0]).isna().all()
    json.dumps(path, allow_nan=False)


@pytest.mark.parametrize(
    "penalties", [(), (0,), (-1,), (np.inf,), (np.nan,), (True,), ("1",), (1, 1)]
)
def test_invalid_penalty_cannot_create_a_different_estimator(penalties):
    with pytest.raises(ValueError):
        fitted(samples(), penalties=penalties)


def calendar_frame():
    """十二天同钟样本，外层截止之后仍保留未来行供扰动检验。"""
    rows = []
    rng = np.random.default_rng(91)
    for day in range(12):
        for hour in range(24):
            value = rng.normal()
            rows.append(
                {
                    "clock_ms": day * DAY_MS + hour * 3_600_000 + 300_000,
                    "x": value,
                    "flat": 3.0,
                    "target_y": 0.02 * value + rng.normal() * 0.005,
                }
            )
    return pd.DataFrame(rows)


def selection(frame, **kwargs):
    """只在第三至第七天的内层日历选择；第九天才作外层重训。"""
    return select_penalty(
        frame,
        features=("x", "flat"),
        target="target_y",
        training_start_ms=0,
        outer_refit_ms=9 * DAY_MS,
        label_delay_ms=21 * 60_000,
        folds=((3 * DAY_MS, 5 * DAY_MS), (5 * DAY_MS, 7 * DAY_MS)),
        **kwargs,
    )


def test_outer_future_prices_and_labels_do_not_change_inner_choice_or_final_fit():
    frame = calendar_frame()
    initial = selection(frame)
    assert initial["status"] == "READY"
    changed = frame.copy()
    future = changed.clock_ms + 21 * 60_000 >= 9 * DAY_MS
    changed.loc[future, ["x", "flat", "target_y"]] = [1e100, -1e100, 1e200]
    assert selection(changed) == initial
    assert initial["final_path"]["maximum_fitted_label_end_ms"] < 9 * DAY_MS
    assert all(
        fold["training_path"]["maximum_fitted_label_end_ms"] < fold["start_ms"]
        for fold in initial["folds"]
    )
    json.dumps(initial, allow_nan=False)


def test_exact_tie_selects_larger_penalty_without_dropping_constant_columns():
    frame = calendar_frame()
    frame[["x", "flat", "target_y"]] = [2.0, 3.0, 7.0]
    result = selection(frame)
    assert result["selected_penalty"] == 100.0
    assert all(row["mean_daily_mse"] == 0 for row in result["grid"])
    assert result["final_path"]["constant_features"] == ["x", "flat"]


def test_missing_inner_day_is_not_compressed_out_of_selection_calendar():
    frame = calendar_frame()
    frame = frame[frame.clock_ms // DAY_MS != 4]
    result = selection(frame)
    assert result["status"] == "NO_COMPLETE_INNER_VALIDATION"
    assert result["selected_penalty"] is None
    assert all(row["mean_daily_mse"] is None for row in result["grid"])
    json.dumps(result, allow_nan=False)


def test_validation_days_have_equal_weight_regardless_of_member_count():
    rows = []
    for day, count, label in ((0, 30, 0), (1, 30, 0), (2, 30, 0), (3, 10, 1), (4, 100, 3)):
        rows.extend(
            {"clock_ms": day * DAY_MS + (i + 1) * 60_000, "x": 2.0, "target_y": label}
            for i in range(count)
        )
    result = select_penalty(
        pd.DataFrame(rows),
        features=("x",),
        target="target_y",
        training_start_ms=0,
        outer_refit_ms=6 * DAY_MS,
        label_delay_ms=0,
        folds=((3 * DAY_MS, 5 * DAY_MS),),
    )
    assert all(row["mean_daily_mse"] == 5 for row in result["grid"])


def test_overlapping_or_outer_future_inner_fold_is_rejected():
    frame = calendar_frame()
    for folds in (
        ((3 * DAY_MS, 5 * DAY_MS), (4 * DAY_MS, 6 * DAY_MS)),
        ((3 * DAY_MS, 10 * DAY_MS),),
        ((DAY_MS + 1, 5 * DAY_MS),),
    ):
        with pytest.raises(ValueError):
            select_penalty(
                frame,
                features=("x",),
                target="target_y",
                training_start_ms=0,
                outer_refit_ms=9 * DAY_MS,
                label_delay_ms=0,
                folds=folds,
            )


def test_predict_rejects_mutated_frozen_constant_coefficient():
    frame = samples()
    path = copy.deepcopy(fitted(frame))
    path["fits"][0]["coefficients"]["flat"] = 1.0
    with pytest.raises(ValueError):
        predict(frame, path, PENALTIES[0])


def test_label_or_phase_cannot_be_smuggled_into_feature_whitelist():
    frame = samples()
    frame["phase_training"] = 1.0
    with pytest.raises(ValueError):
        fit_path(
            frame,
            features=("x", "phase_training"),
            target="target_y",
            refit_ms=100,
            training_start_ms=0,
            label_delay_ms=0,
        )


def test_overflowed_prediction_cannot_win_by_dropping_a_common_validation_member():
    x = np.tile([-1.0, 1.0], 10)
    train = pd.DataFrame({"clock_ms": np.arange(20) * 60_000, "x": x, "target_y": 2e9 * x})
    validation = pd.DataFrame(
        {
            "clock_ms": [DAY_MS + 60_000, DAY_MS + 120_000],
            "x": [1.0, 1e300],
            "target_y": [2e9 / (1 + 1e-6), (2e9 / 101) * 1e300],
        }
    )
    frame = pd.concat([train, validation], ignore_index=True)
    result = select_penalty(
        frame,
        features=("x",),
        target="target_y",
        training_start_ms=0,
        outer_refit_ms=3 * DAY_MS,
        label_delay_ms=0,
        folds=((DAY_MS, 2 * DAY_MS),),
    )
    assert result["selected_penalty"] == 100.0
    for row in result["folds"][0]["validation"][:-1]:
        assert row["status"] == "INCOMPLETE_INNER_PREDICTIONS"
        assert row["known_feature_rows"] == 2
        assert row["unknown_prediction_rows"] == 1
    assert all(row["mean_daily_mse"] is None for row in result["grid"][:-1])
    # 内层选择有效不等于外层可拟合；极值使最终scaler失败也必须如实保留。
    assert result["status"] == "FINAL_FIT_NOT_READY"
    json.dumps(result, allow_nan=False)
