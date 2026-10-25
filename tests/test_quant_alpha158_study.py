"""固定研究协议的身份、因果人口、调参调用和四比较，仅使用合成表。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant import alpha158_study as study
from tradingagents.research.quant.artifacts import save, sha, verify_receipts
from tradingagents.research.quant.ridge import fit_path

ASSETS = ("CRYPTO_PERPETUAL", "TRADFI_PERPETUAL")


def tables(clocks=None, members=25):
    """特征是独立随机数，不从行情构造，也不访问任何历史研究结果。"""
    if clocks is None:
        clocks = [study.REFITS[0] + 5 * study.MINUTE]
    rows = [
        {"symbol": f"{asset}_{i}", "asset_class": asset, "clock_ms": clock}
        for clock in clocks
        for asset in ASSETS
        for i in range(members)
    ]
    ids = pd.DataFrame(rows)
    rng = np.random.default_rng(519)
    features = pd.concat(
        [
            ids,
            pd.DataFrame(rng.normal(size=(len(ids), len(study.FEATURES))), columns=study.FEATURES),
        ],
        axis=1,
    )
    labels = features[[*study.IDENTITY, *study.BASELINE_FEATURES]].copy()
    labels["target_20"] = 0.001 * features.return_3m + 0.002 * features.KMID
    labels["phase_20"] = [study.phase_at(int(clock)) for clock in labels.clock_ms]
    labels["holding_path_observed"] = True
    return features, labels


def manual_record(full_ready=True):
    """已冻结的合成线性模型：基线预测0，扩展预测KMID，便于独立算误差。"""
    selections = {}
    for model, columns in study.MODELS.items():
        coefficients = dict.fromkeys(columns, 0.0)
        if "KMID" in columns:
            coefficients["KMID"] = 1.0
        path = {
            "features": list(columns),
            "common_features": list(study.FEATURES),
            "means": dict.fromkeys(columns, 0.0),
            "scales": dict.fromkeys(columns, 1.0),
            "constant_features": [],
            "fits": [
                {"penalty": 1.0, "status": "READY", "intercept": 0.0, "coefficients": coefficients}
            ],
        }
        ready = full_ready or model == "baseline_ridge_9"
        selections[model] = {
            "status": "READY" if ready else "NO_COMPLETE_INNER_VALIDATION",
            "selected_penalty": 1.0 if ready else None,
            "final_path": path if ready else None,
        }
    return {"models": selections}


def coefficient_fixture(unknown_second_asset=False):
    return {
        "refits": {
            str(refit): {
                "groups": {
                    f"{asset}:20m": manual_record(not unknown_second_asset or asset != ASSETS[1])
                    for asset in ASSETS
                }
            }
            for refit in study.REFITS
        }
    }


def test_join_is_exact_and_preserves_full_identity_when_rows_are_shuffled():
    features, labels = tables()
    labels = labels.sample(frac=1, random_state=1)
    original = features.copy(deep=True)
    result = study.join_labels(features, labels)
    expected = labels.set_index(list(study.IDENTITY)).target_20
    actual = result.set_index(list(study.IDENTITY)).target_20
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())
    assert len(result) == len(features) and result.common_member_gate.all()
    pd.testing.assert_frame_equal(features, original)


def test_missing_target_or_holding_path_cannot_change_common_population():
    features, labels = tables(members=25)
    features.loc[0, study.ALPHA_FEATURES[-1]] = np.nan
    before = study.join_labels(features, labels)
    changed = labels.copy()
    changed.target_20 = np.nan
    changed.holding_path_observed = False
    after = study.join_labels(features, changed)
    columns = [
        "baseline_feature_complete",
        "common_feature_complete",
        "common_member_gate",
        "common_feature_members_at_clock",
    ]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert not before.common_feature_complete.iloc[0]
    assert before.baseline_feature_complete.iloc[0]
    assert before.common_feature_members_at_clock.iloc[0] == 24


def test_20_member_gate_uses_all_167_for_both_models_without_deleting_rows():
    features, labels = tables(members=20)
    features.loc[0, "KMID"] = np.nan
    result = study.join_labels(features, labels)
    group = result[result.asset_class == ASSETS[0]]
    assert len(group) == 20 and group.common_feature_complete.sum() == 19
    assert not group.common_member_gate.any()
    for model in study.MODELS:
        assert study.predict(group, manual_record(), model).isna().all()
    assert result[result.asset_class == ASSETS[1]].common_member_gate.all()


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "changed_identity",
        "changed_old_feature",
        "extra_target",
        "bad_phase",
    ],
)
def test_join_rejects_ambiguous_or_changed_source(mutation):
    features, labels = tables()
    if mutation == "duplicate":
        labels = pd.concat([labels, labels.iloc[:1]], ignore_index=True)
    elif mutation == "missing":
        labels = labels.iloc[1:]
    elif mutation == "changed_identity":
        labels.loc[0, "symbol"] = "OTHER"
    elif mutation == "changed_old_feature":
        features.loc[0, "return_3m"] += 1
    elif mutation == "extra_target":
        features["target_20"] = 1
    else:
        labels.loc[0, "phase_20"] = "training"
    with pytest.raises(ValueError):
        study.join_labels(features, labels)


def test_every_refit_uses_same_folds_grid_and_causal_feature_gate(monkeypatch):
    clocks = [study.TRAINING_START_MS + i * study.DAY_MS + 5 * study.MINUTE for i in range(16)]
    features, labels = tables(clocks=clocks)
    features.loc[0, "KMID"] = np.nan
    labels.loc[1, "target_20"] = np.nan
    frame = study.join_labels(features, labels)
    calls = []

    def spy(received, **kwargs):
        assert received.common_member_gate.all()
        assert 1 in received.index and np.isnan(received.loc[1, "target_20"])
        calls.append(kwargs)
        return {
            "status": "NO_COMPLETE_INNER_VALIDATION",
            "selected_penalty": None,
            "final_path": None,
        }

    # 第二资产的索引不含1；只在第一资产断言缺标签行仍传入。
    def per_asset_spy(received, **kwargs):
        assert received.common_member_gate.all()
        if received.asset_class.iloc[0] == ASSETS[0]:
            return spy(received, **kwargs)
        calls.append(kwargs)
        return {
            "status": "NO_COMPLETE_INNER_VALIDATION",
            "selected_penalty": None,
            "final_path": None,
        }

    monkeypatch.setattr(study, "select_penalty", per_asset_spy)
    records = study.fit_coefficients(frame)
    assert len(calls) == 16 and set(records["refits"]) == set(map(str, study.REFITS))
    for call in calls:
        refit = call["outer_refit_ms"]
        assert call["folds"] == (
            (refit - 4 * study.DAY_MS, refit - 2 * study.DAY_MS),
            (refit - 2 * study.DAY_MS, refit),
        )
        assert call["common_features"] == study.FEATURES
        assert call["penalties"] == study.PENALTIES and call["label_delay_ms"] == 21 * study.MINUTE
        assert call["features"] in study.MODELS.values()


def test_actual_ridge_records_predict_without_reading_future_target():
    features, labels = tables(clocks=[study.TRAINING_START_MS + 5 * study.MINUTE])
    joined = study.join_labels(features, labels)
    frame = joined[joined.asset_class == ASSETS[0]]
    selections = {}
    for model, columns in study.MODELS.items():
        path = fit_path(
            frame,
            features=columns,
            common_features=study.FEATURES,
            target="target_20",
            refit_ms=study.REFITS[0],
            training_start_ms=study.TRAINING_START_MS,
            label_delay_ms=study.LABEL_DELAY_MS,
            penalties=(1.0,),
        )
        selections[model] = {"status": "READY", "selected_penalty": 1.0, "final_path": path}
    changed = frame.copy()
    changed.target_20 = 1e200
    for model in study.MODELS:
        pd.testing.assert_series_equal(
            study.predict(frame, {"models": selections}, model),
            study.predict(changed, {"models": selections}, model),
        )
        assert study.predict(frame, {"models": selections}, model).notna().all()


def test_fixed_and_rolling_select_only_reached_refits_and_keep_training_label():
    marker = {"refits": {str(r): {"groups": {"asset:20m": {"r": r}}} for r in study.REFITS}}
    for clock, expected in (
        (study.REFITS[0] - 1, study.REFITS[0]),
        (study.REFITS[1] - 1, study.REFITS[0]),
        (study.REFITS[1], study.REFITS[1]),
        (study.END_MS, study.REFITS[-1]),
    ):
        assert study.record_at(marker, "asset", clock, "rolling")["r"] == expected
        assert study.record_at(marker, "asset", clock, "fixed")["r"] == study.REFITS[0]
    with pytest.raises(ValueError):
        study.record_at(marker, "asset", study.REFITS[0], "BEST")


def validation_samples():
    clocks = [
        study.REFITS[0] + day * study.DAY_MS + hour * study.HOUR_MS + 5 * study.MINUTE
        for day in range(8)
        for hour in (0, 1)
    ]
    features, labels = tables(clocks=clocks, members=40)
    day = (features.clock_ms - study.REFITS[0]) // study.DAY_MS
    hour = (features.clock_ms % study.DAY_MS) // study.HOUR_MS
    score = 1 + 0.4 * day + 2 * hour
    features["KMID"] = score.astype(float)
    labels["target_20"] = score.astype(float)
    # 第一个小时只有20共同成员，第二小时40：外层仍按小时而非行数加权。
    mask = (hour == 0) & features.symbol.str.rsplit("_", n=1).str[-1].astype(int).ge(20)
    features.loc[mask, study.ALPHA_FEATURES[-1]] = np.nan
    return study.join_labels(features, labels)


def test_four_comparisons_use_same_rows_equal_hour_days_and_unknown_p_counts_in_bh():
    samples = validation_samples()
    result, predictions = study.evaluate(samples, coefficient_fixture(unknown_second_asset=True))
    assert result["declared_tests"] == result["bh_adjustment_family_size"] == 4
    assert result["valid_p_value_tests"] == 2
    assert len(predictions) == 2 * len(samples)
    assert not predictions.duplicated([*study.IDENTITY, "coefficient_mode"]).any()
    for mode in study.METHODS:
        a = result["groups"][f"{ASSETS[0]}:20m"][mode]["phases"]["retrospective_validation"][
            "paired_mse_improvement"
        ]
        expected = {
            f"2026-10-{6 + d:02}": ((1 + 0.4 * d) ** 2 + (3 + 0.4 * d) ** 2) / 2 for d in range(8)
        }
        assert a["daily_values"] == pytest.approx(expected)
        assert a["mean"] == pytest.approx(np.mean(list(expected.values())))
        assert a["bh_q_4"] == pytest.approx(min(1.0, 2 * a["test_p"]))
        b = result["groups"][f"{ASSETS[1]}:20m"][mode]["phases"]["retrospective_validation"][
            "paired_mse_improvement"
        ]
        assert b["test_p"] is None and b["bh_q_4"] is None and not b["positive_and_q_at_most_005"]
    json.dumps(result, allow_nan=False)


def test_one_missing_prediction_invalidates_entire_common_section_even_without_its_label():
    features, labels = tables()
    frame = study.join_labels(features, labels)
    frame = frame[frame.asset_class == ASSETS[0]].copy()
    frame.loc[frame.index[0], "target_20"] = np.nan
    scores = pd.Series(0.0, index=frame.index)
    scores.iloc[0] = np.nan
    metric, status = study._metric(frame, scores, np.ones(len(frame), dtype=bool))
    assert metric is None and status == "INCOMPLETE_COMMON_PREDICTIONS"


def test_predicted_population_stays_fixed_when_validation_targets_change():
    samples = validation_samples()
    _, first = study.evaluate(samples, coefficient_fixture())
    changed = samples.copy()
    changed.target_20 = np.nan
    _, second = study.evaluate(changed, coefficient_fixture())
    columns = [
        *study.IDENTITY,
        "coefficient_mode",
        "common_member_gate",
        *[f"prediction_{m}" for m in study.MODELS],
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_auxiliary_diagnostic_contract_is_fixed_and_has_no_factor_selection(monkeypatch):
    samples = validation_samples()
    _, predictions = study.evaluate(samples, coefficient_fixture())
    calls = []

    def spy(frame, **kwargs):
        calls.append((frame, kwargs))
        return {"checked": True}

    monkeypatch.setattr(study, "diagnose", spy)
    output = study.prediction_diagnostics(samples, predictions)
    assert len(output) == len(calls) == 12
    for frame, args in calls:
        assert args["direction"] == 1 and args["lag_ms"] == (study.HOUR_MS,)
        assert args["common_features"] == ("common167_available",)
        assert args["score_column"] in {f"prediction_{name}" for name in study.MODELS}
        assert not set(study.ALPHA_FEATURES) & set(frame.columns)


def sealed_sources(tmp_path):
    feature_root, label_root = tmp_path / "features", tmp_path / "labels"
    feature_root.mkdir()
    label_root.mkdir()
    universe = {f"symbol_{i}": {"asset_class": ASSETS[i % 2]} for i in range(748)}
    source = {
        "schema": "ta-native-settlement-unwind-study.v1",
        "universe": universe,
        "phases": study.PHASES,
        "orders_enabled": False,
        "leader_labels_read": False,
    }
    save(label_root / "protocol.json", source)
    pd.DataFrame(
        columns=[*study.IDENTITY, *study.BASELINE_FEATURES, *study.LABEL_COLUMNS]
    ).to_parquet(label_root / "hourly_samples.parquet")
    save(label_root / "receipts.json", {p.name: sha(p) for p in label_root.iterdir()})
    feature = {
        "schema": "ta-alpha158-feature-coverage.v1",
        "alpha158_profile": study.PROFILE,
        "alpha158_source_sha256": sha(Path(study.__file__).parent / "factors/alpha158.py"),
        "source_root": str(label_root),
        "source_samples_path": str(label_root / "hourly_samples.parquet"),
        "source_receipts_sha256": sha(label_root / "receipts.json"),
        "input_identities": {
            str(label_root / "hourly_samples.parquet"): sha(label_root / "hourly_samples.parquet")
        },
        "features": list(study.FEATURES),
        "universe": universe,
    }
    save(feature_root / "protocol.json", feature)
    pd.DataFrame(columns=[*study.IDENTITY, *study.FEATURES]).to_parquet(
        feature_root / "features.parquet"
    )
    save(feature_root / "receipts.json", {p.name: sha(p) for p in feature_root.iterdir()})
    return feature_root, label_root, feature


def test_source_contract_is_exact_and_checks_frozen_feature_and_label_identities(tmp_path):
    feature_root, label_root, _ = sealed_sources(tmp_path)
    assert study._source_contract(feature_root, label_root)[1]["universe"]
    (label_root / "hourly_samples.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        study._source_contract(feature_root, label_root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha158_profile", "unit_adapted"),
        ("alpha158_source_sha256", "bad"),
        ("source_receipts_sha256", "bad"),
    ],
)
def test_a_different_frozen_feature_semantic_or_source_is_not_accepted(tmp_path, field, value):
    feature_root, label_root, feature = sealed_sources(tmp_path)
    feature[field] = value
    (feature_root / "protocol.json").unlink()
    save(feature_root / "protocol.json", feature)
    (feature_root / "receipts.json").unlink()
    save(feature_root / "receipts.json", {p.name: sha(p) for p in feature_root.iterdir()})
    with pytest.raises(ValueError):
        study._source_contract(feature_root, label_root)


def test_cli_engineering_failure_is_preserved_with_source_and_receipts(tmp_path, monkeypatch):
    feature_root, label_root, feature = sealed_sources(tmp_path)
    features, labels = tables()
    features.to_parquet(feature_root / "features.parquet", index=False)
    labels.to_parquet(label_root / "hourly_samples.parquet", index=False)
    monkeypatch.setattr(study, "_source_contract", lambda *_: (feature, {"universe": {}}))

    def fail(*_):
        raise ValueError("合成身份网格失败")

    monkeypatch.setattr(study, "_source_grid", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="合成身份网格失败"):
        study.run(feature_root, label_root, output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["stage"] == "load_join" and failure["numeric_conclusions_allowed"] is False
    assert not (output / "coefficients.json").exists()
    verify_receipts(output)
