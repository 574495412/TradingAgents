"""固定模型族须能识别人工预测因子，且不能用验证结果重估或扩大人口。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.response import NAMES
from tradingagents.research.quant.linear_family import evaluate, fit_section
from tradingagents.research.quant.regression import FACTORS, FEATURES
from tradingagents.research.quant.response_study import CONTRASTS, MODELS, phase_at


def panel():
    generator = np.random.default_rng(1729)
    features = FEATURES + NAMES
    parts = []
    phases = {
        "training": {"start": "2026-10-01T00:00:00Z", "end": "2026-10-02T00:00:00Z"},
        "validation": {"start": "2026-10-02T00:00:00Z", "end": "2026-10-03T00:00:00Z"},
        "viewed_later": {"start": "2026-10-03T00:00:00Z", "end": "2026-10-04T00:00:00Z"},
    }
    for asset in ("人工甲", "人工乙"):
        for phase, bounds in phases.items():
            for hour in range(3):
                frame = pd.DataFrame(generator.normal(size=(40, len(features))), columns=features)
                frame["symbol"] = [f"{asset}{i:03}" for i in range(40)]
                frame["asset_class"] = asset
                frame["clock_ms"] = (
                    int(pd.Timestamp(bounds["start"]).value // 1_000_000) + hour * 3_600_000
                )
                for h in (20, 60, 240):
                    frame[f"target_{h}"] = frame[NAMES[0]] * 0.01
                    frame[f"phase_{h}"] = phase
                parts.append(frame)
    return pd.concat(parts, ignore_index=True), {
        "features": features,
        "scored_factors": FACTORS + NAMES,
        "models": MODELS,
        "contrasts": CONTRASTS,
        "horizons_minutes": [20, 60, 240],
        "phases": phases,
    }


def test_nine_models_share_the_same_known_population_even_after_a_factor_is_dropped():
    data, protocol = panel()
    frame = data.iloc[:40].copy()
    frame.loc[0, NAMES[2]] = np.nan
    known, ranked, fits, failures = fit_section(frame, "target_20", protocol["features"], MODELS)
    assert len(known) == len(ranked) == 39 and len(fits) == 9 and failures == {}
    assert frame.loc[0, "symbol"] not in set(known.symbol)


def test_artificial_predictive_factor_has_incremental_value_but_cannot_grant_production():
    data, protocol = panel()
    result = evaluate(data, protocol)
    assert result["declared_tests"] == 270
    assert result["production_eligible"] is result["economic_account_evaluated"] is False
    for group in result["groups"].values():
        validation = group["phases"]["validation"]
        assert validation["comparisons"][f"add_{NAMES[0]}_vs_legacy"]["mean"] > 0


def test_validation_returns_cannot_refit_any_training_coefficients():
    data, protocol = panel()
    initial = evaluate(data, protocol)
    changed = data.copy()
    changed.loc[changed.phase_20 != "training", ["target_20", "target_60", "target_240"]] *= -30
    later = evaluate(changed, protocol)
    for key, group in initial["groups"].items():
        assert (
            group["frozen_training_coefficients"]
            == later["groups"][key]["frozen_training_coefficients"]
        )
        assert group["phases"]["training"] == later["groups"][key]["phases"]["training"]


def test_missing_future_label_does_not_change_the_feature_ranks():
    data, protocol = panel()
    frame = data.iloc[:40].copy()
    _, before, _, _ = fit_section(frame, "target_20", protocol["features"], MODELS)
    frame.loc[0, "target_20"] = np.nan
    _, after, _, _ = fit_section(frame, "target_20", protocol["features"], MODELS)
    pd.testing.assert_frame_equal(before, after)


def test_future_columns_and_fabricated_phase_boundaries_are_rejected():
    data, protocol = panel()
    modified = deepcopy(protocol)
    modified["features"] += ("target_20",)
    with pytest.raises(ValueError, match="未来标签"):
        evaluate(data, modified)
    data.loc[data.phase_20 == "training", "phase_240"] = "excluded_boundary"
    with pytest.raises(ValueError, match="固定日历"):
        evaluate(data, protocol)


def test_four_hour_labels_do_not_cross_the_predeclared_phase_boundary():
    boundary = int(pd.Timestamp("2026-10-06T00:00:00Z").value // 1_000_000)
    assert phase_at(boundary - 4 * 3_600_000, 240) == "excluded_boundary"
    assert phase_at(boundary - 5 * 3_600_000, 240) == "training"


def test_unknown_p_values_remain_unknown_but_can_count_in_the_declared_family():
    data, protocol = panel()
    protocol["bh_count_unknown_in_declared_family"] = True
    result = evaluate(data, protocol)
    assert result["declared_tests"] == result["bh_adjustment_family_size"] == 270
    assert result["valid_p_value_tests"] == 0
    for group in result["groups"].values():
        for phase in group["phases"].values():
            for comparison in phase["comparisons"].values():
                assert comparison["p_asymptotic_two_sided"] is None
                assert comparison["bh_q_all_declared_tests"] is None
