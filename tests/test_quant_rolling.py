"""历史扩展训练须按标签成熟时间隔离，并保持五模型共同拟合人口。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.funding import NAMES
from tradingagents.research.quant.funding_study import MODELS
from tradingagents.research.quant.linear_family import evaluate, fit_section
from tradingagents.research.quant.regression import FEATURES
from tradingagents.research.quant.regression_run import MINUTE, PHASES, ms
from tradingagents.research.quant.response_study import phase_at
from tradingagents.research.quant.rolling import END_MS, REFITS, coefficients_at, fit_expanding

ASSETS = ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL")


def panel(clocks=None):
    """用固定随机截面构建原阶段标签，边界标签保留未知。"""
    if clocks is None:
        clocks = [
            "2026-10-01T01:00:00Z",
            "2026-10-05T19:00:00Z",
            "2026-10-05T20:00:00Z",
            "2026-10-06T00:00:00Z",
            "2026-10-07T19:00:00Z",
            "2026-10-07T20:00:00Z",
            "2026-10-08T00:00:00Z",
            "2026-10-09T19:00:00Z",
            "2026-10-10T00:00:00Z",
            "2026-10-11T19:00:00Z",
            "2026-10-11T20:00:00Z",
            "2026-10-12T00:00:00Z",
            "2026-10-13T00:00:00Z",
        ]
    generator = np.random.default_rng(1729)
    features = FEATURES + NAMES
    parts = []
    for asset in ASSETS:
        for clock in clocks:
            frame = pd.DataFrame(generator.normal(size=(40, len(features))), columns=features)
            frame["symbol"] = [f"{asset}{index:03}" for index in range(40)]
            frame["asset_class"], frame["clock_ms"] = asset, ms(clock)
            for horizon in (20, 60, 240):
                phase = phase_at(ms(clock), horizon)
                frame[f"phase_{horizon}"] = phase
                frame[f"target_{horizon}"] = (
                    np.nan
                    if phase == "excluded_boundary"
                    else (frame[FEATURES[0]] + 2 * frame[NAMES[0]]) * 0.01
                    + generator.normal(scale=0.001, size=40)
                )
            parts.append(frame)
    return pd.concat(parts, ignore_index=True), {
        "features": features,
        "models": MODELS,
        "horizons_minutes": [20, 60, 240],
        "phases": deepcopy(PHASES),
        "contrasts": {},
        "scored_factors": [],
    }


def snapshot(result, refit=0, horizon=240, asset=ASSETS[0]):
    """取一个固定资产期限的重训快照。"""
    return result["refits"][str(REFITS[refit])]["groups"][f"{asset}:{horizon}m"]


def test_fixed_four_blocks_and_first_refit_match_original_frozen_training():
    data, protocol = panel()
    result, original = fit_expanding(data, protocol), evaluate(data, protocol)
    assert result["refit_times_ms"] == list(REFITS)
    assert [value["end_ms"] for value in result["refits"].values()] == [*REFITS[1:], END_MS]
    assert len(result["refits"]) == 4
    assert result["independent_oos"] is result["production_eligible"] is False
    for group_name, old in original["groups"].items():
        new = result["refits"][str(REFITS[0])]["groups"][group_name]
        assert new["frozen_training_coefficients"] == old["frozen_training_coefficients"]
        assert new["training_common_fit_hours"] == old["training_common_fit_hours"]


def test_label_exit_equal_to_refit_is_excluded_until_a_later_refit():
    data, protocol = panel(["2026-10-07T23:38:00Z", "2026-10-07T23:39:00Z", "2026-10-07T23:40:00Z"])
    result = fit_expanding(data, protocol)
    current = snapshot(result, refit=1, horizon=20)
    later = snapshot(result, refit=2, horizon=20)
    assert current["common_training_clocks_ms"] == [ms("2026-10-07T23:38:00Z")]
    assert current["maximum_training_label_end_ms"] == REFITS[1] - MINUTE
    assert current["diagnostics"]["purged_unmatured_hours"] == 2
    assert later["training_common_fit_hours"] == 3


def test_future_labels_and_features_cannot_change_earlier_refits():
    data, protocol = panel()
    original = fit_expanding(data, protocol)
    changed = data.copy()
    for horizon in (20, 60, 240):
        future = changed.clock_ms + (horizon + 1) * MINUTE >= REFITS[1]
        changed.loc[future, f"target_{horizon}"] *= -100
    changed.loc[changed.clock_ms >= REFITS[1], list(FEATURES + NAMES)] *= -3
    altered = fit_expanding(changed, protocol)
    for refit in REFITS[:2]:
        assert original["refits"][str(refit)] == altered["refits"][str(refit)]
    assert (
        snapshot(original, refit=2)["frozen_training_coefficients"]
        != snapshot(altered, refit=2)["frozen_training_coefficients"]
    )


def test_one_rank_deficient_model_excludes_the_hour_from_every_model_mean():
    data, protocol = panel(["2026-10-01T01:00:00Z", "2026-10-01T02:00:00Z"])
    defective = data.clock_ms == ms("2026-10-01T02:00:00Z")
    data.loc[defective, NAMES[1]] = data.loc[defective, NAMES[0]]
    result = fit_expanding(data, protocol)
    group = snapshot(result)
    assert group["training_common_fit_hours"] == 1
    assert group["diagnostics"]["partial_model_fit_hours"] == 1
    assert group["diagnostics"]["individually_fitted_hours"]["legacy"] == 2
    assert group["diagnostics"]["failure_counts_by_model"]["expanded"] == {
        "RANK_DEFICIENT_FIXED_DESIGN": 1
    }
    frame = data[(data.asset_class == ASSETS[0]) & ~defective]
    _, _, expected, _ = fit_section(frame, "target_240", FEATURES + NAMES, MODELS)
    assert group["frozen_training_coefficients"] == expected


def test_original_boundary_nan_is_never_recovered_and_input_is_not_modified():
    data, protocol = panel()
    before = data.copy(deep=True)
    result = fit_expanding(data, protocol)
    missing_clock = ms("2026-10-05T20:00:00Z")
    for refit in range(4):
        group = snapshot(result, refit=refit)
        assert missing_clock not in group["common_training_clocks_ms"]
        assert group["diagnostics"]["original_boundary_hours"] == 1
        assert group["maximum_training_label_end_ms"] < REFITS[refit]
    pd.testing.assert_frame_equal(data, before)


def test_all_models_use_union_feature_population_before_filtering_known_targets():
    data, protocol = panel(["2026-10-01T01:00:00Z"])
    asset = data.asset_class == ASSETS[0]
    indices = data.index[asset]
    data.loc[indices[0], NAMES[1]] = np.nan
    data.loc[indices[1], "target_240"] = np.nan
    result = fit_expanding(data, protocol)
    record = next(
        row
        for row in result["hourly_diagnostics"]
        if row["asset_class"] == ASSETS[0] and row["horizon_minutes"] == 240
    )
    assert record["known_feature_members"] == 39
    assert record["known_target_members"] == 38
    _, _, expected, _ = fit_section(data[asset], "target_240", FEATURES + NAMES, MODELS)
    assert snapshot(result)["frozen_training_coefficients"] == expected


def test_empty_common_training_is_explicit_and_does_not_borrow_future_coefficients():
    data, protocol = panel(["2026-10-06T00:00:00Z"])
    result = fit_expanding(data, protocol)
    first, later = snapshot(result), snapshot(result, refit=1)
    assert first["status"] == "NO_COMMON_FIT_HOURS"
    assert first["model_status"] == dict.fromkeys(MODELS, "NO_COMMON_FIT_HOURS")
    assert first["frozen_training_coefficients"] == dict.fromkeys(MODELS)
    assert first["maximum_training_label_end_ms"] is None
    assert later["status"] == "READY"
    assert coefficients_at(result, ASSETS[0], 240, "expanded", REFITS[0]) is None


@pytest.mark.parametrize("field", ["features", "models", "horizons_minutes", "phases"])
def test_changing_the_original_model_family_or_calendar_is_rejected(field):
    data, protocol = panel(["2026-10-01T01:00:00Z"])
    if field == "features":
        protocol[field] += ("target_20",)
    elif field == "models":
        protocol[field] = {name: columns for name, columns in MODELS.items() if name != "controls"}
    elif field == "horizons_minutes":
        protocol[field] = [20, 60]
    else:
        protocol[field]["training"]["end"] = "2026-10-08T00:00:00Z"
    with pytest.raises(ValueError, match="原特征、五模型"):
        fit_expanding(data, protocol)


@pytest.mark.parametrize("tampering", ["phase", "filled_boundary", "duplicate"])
def test_fabricated_phases_boundary_labels_and_duplicate_rows_are_rejected(tampering):
    data, protocol = panel(["2026-10-05T20:00:00Z"])
    if tampering == "phase":
        data["phase_240"] = "training"
    elif tampering == "filled_boundary":
        data["target_240"] = 0.0
    else:
        data = pd.concat([data, data.iloc[:1]])
    with pytest.raises(ValueError):
        fit_expanding(data, protocol)


def test_coefficient_lookup_changes_only_at_block_boundaries_and_returns_a_copy():
    data, protocol = panel()
    result = fit_expanding(data, protocol)
    arguments = (result, ASSETS[0], 240, "expanded")
    before = coefficients_at(*arguments, REFITS[1] - 1)
    after = coefficients_at(*arguments, REFITS[1])
    assert before == snapshot(result)["frozen_training_coefficients"]["expanded"]
    assert after == snapshot(result, refit=1)["frozen_training_coefficients"]["expanded"]
    assert before != after
    after[FEATURES[0]] = 12345
    assert coefficients_at(*arguments, REFITS[1])[FEATURES[0]] != 12345
    for clock in (REFITS[0] - 1, END_MS, True):
        with pytest.raises(ValueError, match="固定历史滚动"):
            coefficients_at(*arguments, clock)
    with pytest.raises(ValueError, match="已声明"):
        coefficients_at(result, ASSETS[0], 240, "winner", REFITS[0])
