"""增量检验必须保留相同人口、冻结训练且能拒绝有害冗余信号。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.incremental import (
    CONTRASTS,
    MODELS,
    evaluate,
    fit_all,
    predictions,
)
from tradingagents.research.quant.regression import FEATURES


def panel():
    generator = np.random.default_rng(20261016)
    parts = []
    for phase, day in (("training", "2026-10-01"), ("validation", "2026-10-02")):
        for hour in range(4):
            frame = pd.DataFrame(generator.normal(size=(40, len(FEATURES))), columns=FEATURES)
            frame["symbol"] = [f"A{i:03}" for i in range(40)]
            frame["clock_ms"] = (
                int(pd.Timestamp(day, tz="UTC").value // 1_000_000) + hour * 3_600_000
            )
            frame["asset_class"] = "人工类别"
            for horizon in (20, 60):
                frame[f"target_{horizon}"] = frame["return_3m"] * 0.01
                frame[f"phase_{horizon}"] = phase
            parts.append(frame)
    protocol = {
        "phases": {
            "training": {"start": "2026-10-01", "end": "2026-10-02"},
            "validation": {"start": "2026-10-02", "end": "2026-10-03"},
        }
    }
    return pd.concat(parts, ignore_index=True), protocol


def test_all_models_use_the_original_complete_feature_population_even_after_dropping_a_factor():
    samples, _ = panel()
    frame = samples.iloc[:40].copy()
    frame.loc[0, "flow_12m"] = np.nan
    known, ranks, fits, failures = fit_all(frame, "target_20")
    assert len(known) == 39 and len(fits) == 7 and failures == {}
    assert "A000" not in set(known["symbol"])
    for name, beta in fits.items():
        assert set(beta) == set(MODELS[name])
        result = predictions(known, ranks, "target_20", beta)
        assert result["members"] == 39


def test_predictive_factor_has_positive_control_and_drop_contrasts_but_does_not_grant_profitability():
    samples, protocol = panel()
    result = evaluate(samples, protocol)
    for group in result["groups"].values():
        validation = group["phases"]["validation"]
        assert validation["comparisons"]["controls_only"]["r_squared_improvement_same_rows"] > 0
        assert validation["comparisons"]["without_return_3m"]["r_squared_improvement_same_rows"] > 0
        assert set(validation["comparisons"]) == set(CONTRASTS)
    assert (
        result["stable_profitability_proven"] is False
        and result["economic_account_evaluated"] is False
    )


def test_future_labels_do_not_refit_any_of_the_seven_training_models():
    samples, protocol = panel()
    initial = evaluate(samples, protocol)
    changed = samples.copy()
    changed.loc[changed["phase_20"] == "validation", ["target_20", "target_60"]] *= -20
    later = evaluate(changed, protocol)
    for name, group in initial["groups"].items():
        assert (
            group["frozen_training_coefficients"]
            == later["groups"][name]["frozen_training_coefficients"]
        )
        assert group["phases"]["training"] == later["groups"][name]["phases"]["training"]


def test_future_target_missing_does_not_rerank_or_refit_population():
    samples, _ = panel()
    frame = samples.iloc[:40].copy()
    known, ranks, _, _ = fit_all(frame, "target_20")
    frame.loc[frame["return_3m"].idxmax(), "target_20"] = np.nan
    altered, future_ranks, _, _ = fit_all(frame, "target_20")
    assert list(known["symbol"]) == list(altered["symbol"]) and ranks.equals(future_ranks)
    score = predictions(altered, future_ranks, "target_20", {"return_3m": 0.01})
    assert score["spread"]["reason"] == "SELECTED_FUTURE_TARGET_MISSING"


def test_changed_parent_coefficients_are_rejected_instead_of_reinterpreted():
    samples, protocol = panel()
    parent = {
        "groups": {"人工类别:20m": {"training_mean_coefficients": dict.fromkeys(FEATURES, 999)}}
    }
    with pytest.raises(ValueError, match="首轮回归"):
        evaluate(samples, protocol, parent)
