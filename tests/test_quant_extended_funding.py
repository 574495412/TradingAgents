"""历史续检的数值、完整人口、标签成熟与实际列投影验证。"""

import copy

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant import extended_funding_study as study
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts


@pytest.fixture
def panel(monkeypatch):
    day = 24 * study.HOUR
    monkeypatch.setattr(study, "START", 0)
    monkeypatch.setattr(study, "END", 7 * day)
    monkeypatch.setattr(study, "REFITS", (2 * day, 4 * day, 6 * day))
    generator = np.random.default_rng(20261017)
    universe, frames, targets = {}, [], []
    for asset_number, asset in enumerate((*study.ASSETS, "UNKNOWN")):
        for member in range(22 if asset != "UNKNOWN" else 1):
            symbol = f"S{asset_number}_{member:02}"
            universe[symbol] = {"asset_class": asset}
            clocks = np.arange(study.START, study.END, study.HOUR)
            values = generator.normal(size=(len(clocks), len(study.FEATURES)))
            frame = pd.DataFrame(values, columns=study.FEATURES)
            frame["symbol"], frame["asset_class"], frame["clock_ms"] = symbol, asset, clocks
            frame["feature_eligible"] = asset != "UNKNOWN"
            labels = frame[list(study.IDENTITY)].copy()
            labels["target_240"] = (
                0.003 * values[:, 0]
                + 0.005 * values[:, -1]
                + generator.normal(0, 0.0002, len(clocks))
            )
            frames.append(frame[list(study.FEATURE_COLUMNS)])
            targets.append(labels)
    monkeypatch.setattr(study, "EXPECTED_SYMBOLS", len(universe))
    return pd.concat(frames, ignore_index=True), pd.concat(targets, ignore_index=True), universe


def test_full_population_predictions_and_declared_family_are_retained(panel):
    features, labels, universe = panel
    predictions, result, snapshots, hourly, _ = study.analyze(features, labels, universe)
    assert len(predictions) == 2 * len(features)
    assert set(predictions.symbol) == set(universe)
    assert not any(name.startswith(("target", "phase")) for name in predictions)
    assert (
        predictions[predictions.asset_class.eq("UNKNOWN")]
        .filter(like="prediction_")
        .isna()
        .all()
        .all()
    )
    assert (
        predictions[predictions.clock_ms < study.REFITS[0]]
        .filter(like="prediction_")
        .isna()
        .all()
        .all()
    )
    assert result["declared_tests"] == result["bh_adjustment_family_size"] == 24
    assert len(hourly) == 2 * 2 * ((study.END - study.REFITS[0]) // study.HOUR)
    for cutoff, record in snapshots.items():
        for group in record["groups"].values():
            assert group["status"] == "READY"
            assert group["maximum_training_label_end_ms"] < int(cutoff)


def test_snapshot_uses_strict_label_maturity_and_all_five_models(panel):
    cutoff = study.REFITS[0]

    def record(clock, value):
        return {
            "asset_class": study.ASSETS[0],
            "clock_ms": clock,
            "fits": {name: dict.fromkeys(columns, value) for name, columns in study.MODELS.items()},
            "failures": {},
        }

    earlier = record(cutoff - study.LABEL_DELAY - 1, 2.0)
    at_boundary = record(cutoff - study.LABEL_DELAY, 1000.0)
    incomplete = record(cutoff - study.LABEL_DELAY - 2, 1000.0)
    incomplete["fits"].pop("controls")
    incomplete["failures"]["controls"] = "SYNTHETIC_FAILURE"
    snapshot = study._snapshots([earlier, at_boundary, incomplete])[str(cutoff)]["groups"][
        study.ASSETS[0]
    ]
    assert snapshot["training_common_fit_hours"] == 1
    assert snapshot["maximum_training_label_end_ms"] == cutoff - 1
    assert all(
        value == 2.0
        for beta in snapshot["frozen_training_coefficients"].values()
        for value in beta.values()
    )


def test_first_hour_fit_matches_independent_augmented_qr(panel):
    features, labels, universe = panel
    joined = study._validate(features, labels, universe)
    first = joined[
        (joined.asset_class == study.ASSETS[0]) & (joined.clock_ms == study.START)
    ].sort_values("symbol")
    fitted = study._fit_hours(joined)[0]
    ranks = np.empty((len(first), len(study.FEATURES)))
    for column, name in enumerate(study.FEATURES):
        order = np.argsort(first[name].to_numpy(), kind="stable")
        ranks[order, column] = np.arange(len(first)) / (len(first) - 1) - 0.5
    for name, columns in study.MODELS.items():
        matrix = np.column_stack(
            [np.ones(len(first)), ranks[:, [study.FEATURES.index(c) for c in columns]]]
        )
        q, r = np.linalg.qr(matrix)
        expected = np.linalg.solve(r, q.T @ first.target_240.to_numpy())[1:]
        np.testing.assert_allclose(list(fitted["fits"][name].values()), expected, atol=1e-13)


def test_later_target_perturbation_cannot_change_any_registered_prediction(panel):
    features, labels, universe = panel
    before = study.analyze(features, labels, universe)
    changed = labels.copy()
    changed.loc[changed.clock_ms + study.LABEL_DELAY >= study.REFITS[-1], "target_240"] = 1e8
    after = study.analyze(features, changed, universe)
    pd.testing.assert_frame_equal(before[0], after[0])
    assert before[2] == after[2]


def test_missing_future_target_does_not_change_current_members_or_rank_predictions(panel):
    features, labels, universe = panel
    before = study.analyze(features, labels, universe)
    changed = labels.copy()
    row = changed[
        (changed.clock_ms == study.REFITS[-1]) & (changed.asset_class == study.ASSETS[0])
    ].index[0]
    changed.loc[row, "target_240"] = np.nan
    after = study.analyze(features, changed, universe)
    pd.testing.assert_frame_equal(before[0], after[0])
    assert before[2] == after[2]
    assert all(r["common_members"] == 22 for r in after[3])


def test_future_feature_change_does_not_modify_earlier_predictions(panel):
    features, labels, universe = panel
    before = study.analyze(features, labels, universe)[0]
    changed = features.copy()
    at = study.REFITS[-1] + study.HOUR
    changed.loc[changed.clock_ms == at, study.FEATURES[0]] = np.nan
    after = study.analyze(changed, labels, universe)[0]
    pd.testing.assert_frame_equal(before[before.clock_ms < at], after[after.clock_ms < at])
    assert not after[after.clock_ms == at].common_member_gate.any()


def test_rank_deficiency_does_not_fallback_to_partial_models_or_zero_coefficients(panel):
    features, labels, universe = panel
    features = features.copy()
    features[list(study.FEATURES)] = 1.0
    predictions, result, snapshots, _, fits = study.analyze(features, labels, universe)
    assert predictions.filter(like="prediction_").isna().all().all()
    assert all(r["failures"] for r in fits)
    assert all(
        group["status"] == "NO_COMMON_FIT_HOURS"
        for record in snapshots.values()
        for group in record["groups"].values()
    )
    assert result["declared_tests"] == 24
    assert result["valid_p_value_tests"] == 0


def test_daily_mse_is_hour_equal_then_calendar_day_equal(panel):
    features, labels, universe = panel
    _, result, _, hourly, _ = study.analyze(features, labels, universe)
    asset = study.ASSETS[0]
    part = [
        row for row in hourly if row["asset_class"] == asset and row["coefficient_mode"] == "fixed"
    ]
    days = sorted({row["day"] for row in part})
    daily = {
        day: float(np.mean([r["models"]["legacy"]["mse"] for r in part if r["day"] == day]))
        for day in days
    }
    actual = result["groups"][f"fixed:{asset}"]["model_metrics"]["legacy"]["mse"]
    assert actual["daily_values"] == daily
    assert actual["mean"] == pytest.approx(np.mean(list(daily.values())))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "float_clock",
        "wrong_class",
        "future_feature",
        "infinity",
        "unknown_eligibility",
        "label_identity",
    ],
)
def test_input_contract_rejects_identity_and_population_mutations(panel, fault):
    features, labels, universe = panel
    features, labels, universe = features.copy(), labels.copy(), copy.deepcopy(universe)
    if fault == "missing":
        features = features.iloc[1:]
    elif fault == "duplicate":
        features.iloc[1] = features.iloc[0]
    elif fault == "float_clock":
        features["clock_ms"] = features.clock_ms.astype(float)
    elif fault == "wrong_class":
        universe[features.symbol.iloc[0]]["asset_class"] = "GUESSED_COIN"
    elif fault == "future_feature":
        features["target_240"] = 100.0
    elif fault == "infinity":
        features.loc[0, study.FEATURES[0]] = np.inf
    elif fault == "unknown_eligibility":
        features["feature_eligible"] = features.feature_eligible.astype(object)
        features.loc[0, "feature_eligible"] = None
    else:
        labels.loc[0, "asset_class"] = "UNKNOWN"
    with pytest.raises(ValueError):
        study.analyze(features, labels, universe)


def test_metric_rejects_any_missing_common_prediction_or_overflow():
    known = pd.DataFrame({"target_240": np.linspace(-0.01, 0.01, 22)})
    prediction = np.ones(22)
    prediction[0] = np.nan
    assert study._metric(known, prediction) is None
    assert study._metric(known, np.full(22, 1e308)) is None


def test_runner_physically_projects_whitelists_and_seals_complete_outputs(
    tmp_path, panel, monkeypatch
):
    features, labels, universe = panel
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    features.assign(future_leak=1e8).to_parquet(source / "features.parquet", index=False)
    labels.assign(holding_path_observed_240=False).to_parquet(
        source / "labels.parquet", index=False
    )
    save(
        source / "protocol.json",
        {
            "schema": "ta-historical-funding-panel.v1",
            "start_ms": study.START,
            "end_ms": study.END,
            "features": study.FEATURES,
            "universe": universe,
            "study_plan_sha256": sha(study.PLAN),
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    save(source / "result.json", {"status": "COMPLETED_WITH_UNKNOWN"})
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})
    original = pd.read_parquet
    reads = []

    def reader(path, **kwargs):
        reads.append((path.name, kwargs["columns"]))
        return original(path, **kwargs)

    monkeypatch.setattr(study.pd, "read_parquet", reader)
    result = study.run(source, output)
    assert reads == [
        ("features.parquet", list(study.FEATURE_COLUMNS)),
        ("labels.parquet", list(study.LABEL_COLUMNS)),
    ]
    verify_receipts(output)
    assert result["declared_tests"] == 24
    assert not read(output / "protocol.json")["production_eligible"]
    assert result["economic_accounts_complete"] is False


@pytest.mark.parametrize("stage", ["upstream", "calculation"])
def test_runner_preserves_failure_without_valid_result(tmp_path, panel, monkeypatch, stage):
    features, labels, universe = panel
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    features.to_parquet(source / "features.parquet", index=False)
    labels.to_parquet(source / "labels.parquet", index=False)
    save(
        source / "protocol.json",
        {
            "schema": "ta-historical-funding-panel.v1",
            "start_ms": study.START,
            "end_ms": study.END,
            "features": study.FEATURES,
            "universe": universe,
            "study_plan_sha256": sha(study.PLAN),
            "orders_enabled": False,
            "leader_labels_read": False,
            "production_eligible": False,
        },
    )
    save(
        source / "result.json",
        {"status": "FAILED_INTEGRITY" if stage == "upstream" else "COMPLETED_WITH_UNKNOWN"},
    )
    save(source / "receipts.json", {p.name: sha(p) for p in source.iterdir()})

    def failure(*args):
        raise ArithmeticError("合成失败")

    monkeypatch.setattr(study, "analyze", failure)
    if stage == "upstream":
        with pytest.raises(ValueError, match="未成功完成"):
            study.run(source, output)
        assert not output.exists()
        return
    with pytest.raises(ArithmeticError):
        study.run(source, output)
    verify_receipts(output)
    assert read(output / "failure.json")["numeric_conclusions_allowed"] is False
    assert not (output / "result.json").exists()
