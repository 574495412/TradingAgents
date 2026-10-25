"""十二分数仅作固定诊断；全部输入为合成，不读取真实历史结果。"""

import json
import weakref

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant import extended_funding_diagnostics as diagnostics
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts


@pytest.fixture
def inputs(monkeypatch):
    study = diagnostics.study
    monkeypatch.setattr(study, "START", 0)
    monkeypatch.setattr(study, "END", 27 * study.HOUR)
    monkeypatch.setattr(diagnostics, "START", 0)
    monkeypatch.setattr(diagnostics, "END", 27 * study.HOUR)
    frames, labels, universe = [], [], {}
    for asset_index, asset in enumerate((*diagnostics.ASSETS, "UNKNOWN")):
        for member in range(22 if asset != "UNKNOWN" else 1):
            symbol = f"S{asset_index}_{member:02}"
            universe[symbol] = {"asset_class": asset}
            for clock in range(0, study.END, study.HOUR):
                values = {name: float(member + 1) for name in diagnostics.FEATURES}
                values[diagnostics.RAW_NAMES[0]] = float(member // 4)
                values[diagnostics.RAW_NAMES[1]] = 0.0
                if member == 20:
                    values[diagnostics.FEATURES[0]] = np.nan
                identity = {"symbol": symbol, "asset_class": asset, "clock_ms": clock}
                frames.append(
                    {**identity, **values, "feature_eligible": member != 21 and asset != "UNKNOWN"}
                )
                labels.append(
                    {**identity, "target_240": member / 10000 + clock / study.HOUR / 100000}
                )
    features, targets = pd.DataFrame(frames), pd.DataFrame(labels)
    monkeypatch.setattr(study, "EXPECTED_SYMBOLS", len(universe))
    common = (
        features.feature_eligible
        & np.isfinite(features[list(diagnostics.FEATURES)]).all(axis=1)
        & features.asset_class.isin(diagnostics.ASSETS)
    )
    population = common & (
        common.groupby([features.asset_class, features.clock_ms]).transform("sum") >= 20
    )
    pieces = []
    for mode in diagnostics.MODES:
        predicted = features[list(diagnostics.IDENTITY)].copy()
        predicted["coefficient_mode"], predicted["common_member_gate"] = mode, population
        members = features.symbol.str.rsplit("_", n=1).str[-1].astype(float)
        for number, model in enumerate(diagnostics.MODEL_NAMES):
            sign = np.where((mode == "rolling") & (features.clock_ms >= 24 * study.HOUR), -1, 1)
            predicted[f"prediction_{model}"] = np.where(
                population, sign * members * (number + 1) / 10000, np.nan
            )
        pieces.append(predicted)
    return (
        features[list(study.FEATURE_COLUMNS)],
        targets[list(study.LABEL_COLUMNS)],
        pd.concat(pieces, ignore_index=True),
        universe,
    )


def scoped(inputs):
    features, labels, predictions, universe = inputs
    base, predictions = diagnostics.validate_inputs(features, labels, predictions, universe)
    base, identities = diagnostics.scope_identities(base)
    return base, predictions, identities


def test_exact_twelve_scores_and_unknown_identity_are_preserved(inputs):
    assert len(diagnostics.SCORES) == 12
    assert len([s for s in diagnostics.SCORES if s["kind"] == "raw_regular_funding_factor"]) == 2
    assert all(score["direction"] == 1 for score in diagnostics.SCORES)
    base, predictions, identities = scoped(inputs)
    assert len(identities) == len(inputs[0])
    assert (
        identities.loc[identities.asset_class == "UNKNOWN", "diagnostic_asset_status"]
        .eq("EXCLUDED_UNKNOWN_ASSET_CLASS")
        .all()
    )
    assert set(base.asset_class) == set(diagnostics.ASSETS)
    assert int(identities.common_member_gate.sum()) == 2 * 20 * 27
    assert len(predictions) == 2 * len(identities)


@pytest.mark.parametrize("change", ["missing", "duplicate", "gate", "clock_float", "outside_score"])
def test_prediction_identity_and_common_population_cannot_be_redefined(inputs, change):
    features, labels, predictions, universe = inputs
    predictions = predictions.copy()
    if change == "missing":
        predictions = predictions.iloc[1:]
    elif change == "duplicate":
        predictions = pd.concat([predictions.iloc[1:], predictions.iloc[[1]]])
    elif change == "gate":
        predictions.loc[0, "common_member_gate"] = False
    elif change == "clock_float":
        predictions["clock_ms"] = predictions.clock_ms.astype(float)
    else:
        index = predictions.index[~predictions.common_member_gate][0]
        predictions.loc[index, "prediction_controls"] = 1
    with pytest.raises(ValueError):
        diagnostics.validate_inputs(features, labels, predictions, universe)


def test_raw_ties_constant_scores_future_labels_and_missing_labels_do_not_select_members(inputs):
    base, predictions, _ = scoped(inputs)
    first = diagnostics.evaluate_score(base, predictions, diagnostics.SCORES[0])
    zero = diagnostics.evaluate_score(base, predictions, diagnostics.SCORES[1])
    assert all(row["maximum_tie_size"] == 4 for row in first["sections"])
    assert all(row["status"] == "CONSTANT_SCORE" for row in zero["sections"])
    assert all(
        row["quantile_counts"] == {"1": 0, "2": 0, "3": 20, "4": 0, "5": 0}
        for row in zero["sections"]
    )
    changed = base.copy()
    changed["target_240"] = np.nan
    changed_result = diagnostics.evaluate_score(changed, predictions, diagnostics.SCORES[0])
    assert [row["quantile_counts"] for row in first["sections"]] == [
        row["quantile_counts"] for row in changed_result["sections"]
    ]
    assert all(row["label_known_members"] == 0 for row in changed_result["sections"])
    assert all(row["rank_ic"] is None for row in changed_result["sections"])
    assert "p_asymptotic_two_sided" not in json.dumps(first)
    assert not first["new_p_values_emitted"] and not first["economic_account_evaluated"]


def test_fixed_rolling_are_separate_and_exact_24_hour_lag_is_not_adjacent(inputs):
    base, predictions, _ = scoped(inputs)
    fixed_score = next(
        s for s in diagnostics.SCORES if s.get("mode") == "fixed" and s.get("model") == "legacy"
    )
    rolling_score = next(
        s for s in diagnostics.SCORES if s.get("mode") == "rolling" and s.get("model") == "legacy"
    )
    fixed = diagnostics.evaluate_score(base, predictions, fixed_score)
    rolling = diagnostics.evaluate_score(base, predictions, rolling_score)
    clock = 24 * diagnostics.HOUR
    fixed_row = next(
        r
        for r in fixed["sections"]
        if r["asset_class"] == diagnostics.ASSETS[0] and r["clock_ms"] == clock
    )
    rolling_row = next(
        r
        for r in rolling["sections"]
        if r["asset_class"] == diagnostics.ASSETS[0] and r["clock_ms"] == clock
    )
    assert fixed_row["persistence"][str(24 * diagnostics.HOUR)][
        "rank_autocorrelation"
    ] == pytest.approx(1)
    assert rolling_row["persistence"][str(24 * diagnostics.HOUR)][
        "rank_autocorrelation"
    ] == pytest.approx(-1)
    assert (
        rolling_row["persistence"][str(diagnostics.HOUR)]["quantile_turnover"]["5"][
            "entry_fraction"
        ]
        == 1
    )
    summary = rolling["summary"][diagnostics.ASSETS[0]]["information"]["rank_ic"]
    assert summary["period"]["period_ms"] == diagnostics.HOUR
    assert not summary["period"]["annualization_applied"]
    assert summary["daily"]["period_ms"] == 24 * diagnostics.HOUR
    assert len(summary["daily"]["daily_values"]) == 2


def test_missing_score_invalidates_complete_clock_and_missing_clock_remains(inputs):
    base, predictions, _ = scoped(inputs)
    score = next(
        s for s in diagnostics.SCORES if s.get("mode") == "fixed" and s.get("model") == "controls"
    )
    predictions = predictions.copy()
    chosen = predictions.index[
        (predictions.coefficient_mode == "fixed")
        & (predictions.clock_ms == 0)
        & predictions.common_member_gate
    ][0]
    predictions.loc[chosen, "prediction_controls"] = np.nan
    base = base[base.clock_ms != diagnostics.HOUR]
    result = diagnostics.evaluate_score(base, predictions, score)
    first = result["sections"][0]
    assert first["status"] == "INCOMPLETE_SCORE_ON_COMMON_POPULATION"
    assert first["common_feature_members"] == 20
    missing = next(
        row
        for row in result["sections"]
        if row["asset_class"] == diagnostics.ASSETS[0] and row["clock_ms"] == diagnostics.HOUR
    )
    assert missing["status"] == "MISSING_SECTION"
    later = next(
        row
        for row in result["sections"]
        if row["asset_class"] == diagnostics.ASSETS[0] and row["clock_ms"] == 2 * diagnostics.HOUR
    )
    assert later["persistence"][str(diagnostics.HOUR)]["status"] == "UNAVAILABLE_MEMBERSHIP"
    assert len(result["sections"]) == 2 * 27


def seal(path):
    save(
        path / "receipts.json",
        {str(p.relative_to(path)): sha(p) for p in path.rglob("*") if p.is_file()},
    )


def sources(tmp_path, inputs):
    features, labels, predictions, universe = inputs
    panel, source = tmp_path / "panel", tmp_path / "study"
    panel.mkdir()
    source.mkdir()
    parent = {
        "schema": "ta-historical-funding-panel.v1",
        "start_ms": diagnostics.study.START,
        "end_ms": diagnostics.END,
        "features": list(diagnostics.FEATURES),
        "universe": universe,
        "orders_enabled": False,
        "leader_labels_read": False,
        "production_eligible": False,
        "study_plan_sha256": sha(diagnostics.study.PLAN),
    }
    save(panel / "protocol.json", parent)
    save(panel / "result.json", {"status": "COMPLETED_WITH_UNKNOWN"})
    features.assign(forbidden_target="不能读取").to_parquet(panel / "features.parquet", index=False)
    labels.assign(forbidden_phase="不能读取").to_parquet(panel / "labels.parquet", index=False)
    seal(panel)
    protocol = {
        **parent,
        "schema": "ta-extended-funding-study.v1",
        "source": str(panel),
        "models": {name: list(columns) for name, columns in diagnostics.study.MODELS.items()},
        "refit_times_ms": list(diagnostics.study.REFITS),
        "source_sha256": {str(diagnostics.study.PLAN.resolve()): sha(diagnostics.study.PLAN)},
        "horizon_minutes": 240,
        "label_delay_ms": 241 * diagnostics.study.MINUTE,
        "input_identities": {
            str(panel / name): sha(panel / name)
            for name in ("receipts.json", "protocol.json", "features.parquet", "labels.parquet")
        },
    }
    save(source / "protocol.json", protocol)
    save(
        source / "result.json",
        {
            "schema": "ta-extended-funding-regression-result.v1",
            "declared_tests": 24,
            "protocol_sha256": sha(source / "protocol.json"),
        },
    )
    predictions.assign(forbidden_target="不能读取").to_parquet(
        source / "predictions.parquet", index=False
    )
    seal(source)
    return panel, source


def test_runner_projects_whitelists_keeps_all_scores_and_releases_prior_result(
    tmp_path, inputs, monkeypatch
):
    panel, source = sources(tmp_path, inputs)
    original_read, original_evaluate = pd.read_parquet, diagnostics.evaluate_score
    loaded, previous = [], []

    def projected(path, **kwargs):
        loaded.append((str(path), kwargs["columns"]))
        return original_read(path, **kwargs)

    class Tracked(dict):
        pass

    def tracked(*args):
        if previous:
            assert previous[-1]() is None
        result = Tracked(original_evaluate(*args))
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(pd, "read_parquet", projected)
    monkeypatch.setattr(diagnostics, "evaluate_score", tracked)
    output = tmp_path / "result"
    result = diagnostics.run(panel, source, output)
    assert result["scores"] == 12 and result["status"] == "COMPLETED_WITH_UNKNOWN"
    assert [columns for _, columns in loaded] == [
        list(diagnostics.study.FEATURE_COLUMNS),
        list(diagnostics.study.LABEL_COLUMNS),
        list(diagnostics.PREDICTION_COLUMNS),
    ]
    assert len(list((output / "scores").glob("*.json"))) == 12
    assert len(read(output / "summary.json")) == 12
    identities = original_read(output / "scope_identities.parquet")
    assert len(identities) == len(inputs[0]) and identities.asset_class.eq("UNKNOWN").any()
    verify_receipts(output)


def test_source_failure_and_later_source_mutation_cannot_be_successful(
    tmp_path, inputs, monkeypatch
):
    panel, source = sources(tmp_path, inputs)
    original = diagnostics.evaluate_score
    changed = False

    def mutate(*args):
        nonlocal changed
        result = original(*args)
        if not changed:
            (panel / "result.json").write_text('{"status":"FAILED_INTEGRITY"}')
            changed = True
        return result

    monkeypatch.setattr(diagnostics, "evaluate_score", mutate)
    output = tmp_path / "failed"
    result = diagnostics.run(panel, source, output)
    assert result["status"] == "FAILED_INTEGRITY"
    assert not read(output / "failure.json")["numeric_conclusions_allowed"]
    assert not (output / "summary.json").exists()
    assert len(list((output / "scores").glob("*.json"))) == 12
    verify_receipts(output)


def test_failed_upstream_is_refused_before_any_new_result(tmp_path, inputs):
    panel, source = sources(tmp_path, inputs)
    save(source / "failure.json", {"numeric_conclusions_allowed": False})
    (source / "receipts.json").unlink()
    seal(source)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="失败来源"):
        diagnostics.run(panel, source, output)
    assert not output.exists()
