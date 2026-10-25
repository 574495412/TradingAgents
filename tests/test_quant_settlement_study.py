"""原生结算假说的精确标签、共同标准化和时间隔离验证。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.regression_run import MINUTE, ms
from tradingagents.research.quant.settlement_study import (
    FEATURES,
    H01_NAME,
    LABEL_DELAY_MS,
    MODELS,
    REFITS,
    attribution,
    evaluate,
    fit_coefficients,
    fit_record,
    phase_at,
    predict,
    price_features,
    record_at,
)


def samples():
    rng = np.random.default_rng(704)
    rows = []
    clocks = [
        ms(f"2026-10-{day:02}T{hour:02}:05:00Z") for day in range(1, 15) for hour in (0, 1, 8, 16)
    ]
    for asset in ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL"):
        for clock in clocks:
            for member in range(24):
                values = rng.normal(size=len(FEATURES))
                if pd.Timestamp(clock, unit="ms", tz="UTC").hour not in (0, 8, 16):
                    values[-1] = 0
                row = dict(zip(FEATURES, values, strict=True))
                row.update(
                    symbol=f"{asset}_{member}",
                    asset_class=asset,
                    clock_ms=clock,
                    target_20=0.0002 + float(values @ np.arange(1, 11)) / 10000,
                    phase_20=phase_at(clock),
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_label_uses_twentieth_minute_close_not_following_open():
    clocks = np.arange(150) * MINUTE
    frame = pd.DataFrame(
        {
            "open": 100 + np.arange(150) * 2,
            "close": 101 + np.arange(150) * 3,
            "quote_volume": np.ones(150),
            "taker_buy_quote": np.full(150, 0.5),
        },
        index=clocks,
    )
    result = price_features(frame)
    row = 70
    expected = frame.close.iloc[row + 21] / frame.open.iloc[row + 2] - 1
    assert result.target_20.iloc[row] == expected
    assert expected != frame.open.iloc[row + 22] / frame.open.iloc[row + 2] - 1
    assert result.holding_path_observed.iloc[row]
    frame.loc[clocks[row + 10], "close"] = np.nan
    missing = price_features(frame)
    assert missing.target_20.iloc[row] == expected
    assert not missing.holding_path_observed.iloc[row]


def test_pooled_fit_retains_zero_hours_and_recovers_full_intercept_prediction():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'")
    record = fit_record(frame, REFITS[0])
    assert all(v == "READY" for v in record["status_by_model"].values())
    assert record["h01_known_zero_rows"] > 0
    assert record["h01_positive_direction_by_model"]["expanded_h01"]
    hour = frame[frame.clock_ms == frame.clock_ms.iloc[0]]
    scores = predict(hour, record, "expanded_h01")
    np.testing.assert_allclose(scores, hour.target_20, atol=1e-16)
    assert abs(record["coefficients"]["expanded_h01"]["intercept"]) > 0


def test_standardization_uses_mature_known_features_before_target_missingness():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'").copy()
    frame.loc[frame.index[0], "target_20"] = np.nan
    record = fit_record(frame, REFITS[0])
    before = frame[(frame.clock_ms + LABEL_DELAY_MS < REFITS[0])]
    assert record["common_feature_rows"] == len(before)
    assert record["known_target_rows"] == len(before) - 1
    assert record["means"][FEATURES[0]] == before[FEATURES[0]].mean()
    assert record["scales"][FEATURES[0]] == before[FEATURES[0]].std(ddof=0)


def test_future_and_exact_maturity_boundary_never_change_frozen_fit():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'").copy()
    expected = fit_record(frame, REFITS[0])
    future = frame.clock_ms + LABEL_DELAY_MS >= REFITS[0]
    frame.loc[future, [*FEATURES, "target_20"]] = 1e40
    edge = frame.iloc[[0]].copy()
    edge["clock_ms"] = REFITS[0] - LABEL_DELAY_MS
    edge[list(FEATURES)] = 1e60
    assert fit_record(pd.concat([frame, edge]), REFITS[0]) == expected


def test_zero_variance_h01_fails_augmented_models_without_deleting_zero_hours():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'").copy()
    frame[H01_NAME] = 0
    record = fit_record(frame, REFITS[0])
    assert record["status_by_model"]["legacy"] == "READY"
    assert record["status_by_model"]["add_h01"] == "ZERO_OR_UNKNOWN_TRAINING_VARIANCE"
    assert record["coefficients"]["expanded_h01"] is None
    assert record["h01_known_zero_rows"] == record["common_feature_rows"]


def test_negative_mechanism_coefficient_is_reported_without_sign_flip():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'").copy()
    frame["target_20"] = -frame[H01_NAME]
    record = fit_record(frame, REFITS[0])
    assert record["status_by_model"]["expanded_h01"] == "READY"
    assert record["coefficients"]["expanded_h01"][H01_NAME] < 0
    assert record["h01_positive_direction_by_model"]["expanded_h01"] is False
    hour = frame[frame.clock_ms == frame.clock_ms.iloc[0]]
    np.testing.assert_allclose(predict(hour, record, "expanded_h01"), hour.target_20, atol=1e-14)


def test_predict_ignores_future_columns_and_keeps_full_common_population():
    frame = samples().query("asset_class == 'CRYPTO_PERPETUAL'")
    record = fit_record(frame, REFITS[0])
    hour = frame[frame.clock_ms == frame.clock_ms.iloc[0]].copy()
    before = predict(hour, record, "legacy")
    hour["target_20"], hour["phase_20"] = -1e99, "untrusted"
    assert before.equals(predict(hour, record, "legacy"))
    hour.loc[hour.index[0], H01_NAME] = np.nan
    assert np.isnan(predict(hour, record, "legacy").iloc[0])
    with pytest.raises(ValueError, match="固定设计"):
        predict(hour, record, "legacy", (*FEATURES, "target_20"))


def test_all_models_modes_blocks_and_twelve_tests_are_retained():
    frame = samples()
    fitted = fit_coefficients(frame)
    assert list(fitted["models"]) == list(MODELS)
    assert len(fitted["refits"]) == 4
    assert (
        record_at(fitted, "CRYPTO_PERPETUAL", REFITS[2], "fixed")
        == fitted["refits"][str(REFITS[0])]["groups"]["CRYPTO_PERPETUAL:20m"]
    )
    assert (
        record_at(fitted, "CRYPTO_PERPETUAL", REFITS[2], "rolling")
        == fitted["refits"][str(REFITS[2])]["groups"]["CRYPTO_PERPETUAL:20m"]
    )
    result, hourly, predictions = evaluate(frame, fitted)
    assert result["declared_tests"] == result["bh_adjustment_family_size"] == 12
    assert result["previous_declared_funding_tests"] == 216
    assert set(result["groups"]) == {"CRYPTO_PERPETUAL:20m", "TRADIFI_PERPETUAL:20m"}
    assert len(predictions) == len(frame) * 2
    assert set(predictions.coefficient_mode) == {"fixed", "rolling"}
    assert len(hourly) == frame.groupby(["asset_class", "clock_ms"]).ngroups * 2
    for modes in result["groups"].values():
        for mode in ("fixed", "rolling"):
            assert len(modes[mode]["two_day_blocks"]) == 4
            assert set(modes[mode]["phases"]["retrospective_validation"]["models"]) == set(MODELS)
            assert "bh_q_12" in modes[mode]["phases"]["retrospective_validation"]["h01_rank_ic"]


def test_unknown_tests_still_occupy_all_twelve_bh_slots():
    frame = samples()
    frame.loc[frame.phase_20 == "retrospective_validation", "target_20"] = np.nan
    result, _, _ = evaluate(frame, fit_coefficients(frame))
    assert result["declared_tests"] == result["bh_adjustment_family_size"] == 12
    assert result["valid_p_value_tests"] == 0
    for modes in result["groups"].values():
        for mode in modes.values():
            assert mode["phases"]["retrospective_validation"]["h01_rank_ic"]["bh_q_12"] is None


def test_full_attribution_uses_centered_component_and_preserves_inactive_and_unknown():
    clock = REFITS[0] + 5 * MINUTE
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "asset_class": "CRYPTO_PERPETUAL",
            "clock_ms": clock,
            "target_20": [0, 0.1, 0.2, np.nan],
            H01_NAME: [0.001, 0, 0, np.nan],
            "settlement_latest_event_ms": [
                clock - 5 * MINUTE,
                clock - 5 * MINUTE,
                clock - 125 * MINUTE,
                np.nan,
            ],
            "settlement_age_minutes": [5, 5, 125, np.nan],
        }
    )
    record = {
        "means": {H01_NAME: 0.00025},
        "scales": {H01_NAME: 0.0005},
        "coefficients": {
            name: {"intercept": 0, **({H01_NAME: 0.3} if H01_NAME in columns else {})}
            for name, columns in MODELS.items()
        },
    }
    coefficients = {
        "refits": {str(refit): {"groups": {"CRYPTO_PERPETUAL:20m": record}} for refit in REFITS}
    }
    predictions = frame.copy()
    predictions["prediction_legacy"] = [0.1, 0.2, 0.3, np.nan]
    predictions["prediction_expanded"] = [0.15, 0.25, 0.35, np.nan]
    predictions["prediction_add_h01"] = [0.55, 0.05, 0.15, np.nan]
    predictions["prediction_expanded_h01"] = [0.6, 0.1, 0.2, np.nan]
    both = pd.concat(
        [predictions.assign(coefficient_mode=mode) for mode in ("fixed", "rolling")],
        ignore_index=True,
    )
    result = attribution(frame, coefficients, both)
    assert result["sample_rows_including_both_modes"] == 8
    value = next(row for row in result["by_date"] if row["coefficient_mode"] == "fixed")
    assert value["sample_rows"] == 4
    assert value["h01_unknown_rows"] == value["h01_nonzero_rows"] == 1
    assert value["h01_known_zero_rows"] == value["active_event_rows"] == 2
    assert value["expired_or_no_event_zero_rows"] == 1
    assert value["models"]["legacy"]["h01_contribution"] == {"rows": 3, "sum": 0, "mean": 0}
    assert value["models"]["expanded_h01"]["h01_contribution"]["mean"] == pytest.approx(0.05)
    assert value["models"]["legacy"]["squared_error"]["mean"] == pytest.approx(0.01)
    assert value["paired_mse_improvement"]["add_h01_vs_legacy"]["mean"] == pytest.approx(-0.0925)
    symbol = next(
        row
        for row in result["by_symbol"]
        if row["symbol"] == "C" and row["coefficient_mode"] == "fixed"
    )
    assert symbol["models"]["add_h01"]["h01_contribution"]["mean"] == pytest.approx(-0.15)
    events = [row for row in result["by_settlement_event"] if row["coefficient_mode"] == "fixed"]
    assert {row["settlement_event"] for row in events} == {
        f"A:{clock - 5 * MINUTE}",
        f"B:{clock - 5 * MINUTE}",
        "INACTIVE_NO_ACTIVE_EVENT",
        "UNKNOWN_H01",
    }
    assert sum(row["sample_rows"] for row in events) == len(frame)
    assert len(result["by_utc_hour"]) == 2


def test_attribution_rejects_outcome_selected_subset_or_duplicate_prediction_rows():
    clock = REFITS[0] + 5 * MINUTE
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "asset_class": "CRYPTO_PERPETUAL",
            "clock_ms": clock,
            H01_NAME: np.nan,
            "settlement_latest_event_ms": np.nan,
            "settlement_age_minutes": np.nan,
        }
    )
    prediction = frame.assign(
        coefficient_mode="fixed",
        target_20=np.nan,
        **{f"prediction_{name}": np.nan for name in MODELS},
    )
    with pytest.raises(ValueError, match="完整覆盖"):
        attribution(frame, {}, prediction)
    with pytest.raises(ValueError, match="身份重复"):
        attribution(frame, {}, pd.concat([prediction, prediction]))


def test_standard_main_passes_only_explicit_sources(monkeypatch):
    import tradingagents.research.quant.settlement_study as study

    called = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "settlement",
            "--source",
            "source",
            "--funding-root",
            "funding",
            "--native-review",
            "native",
            "--output",
            "output",
        ],
    )
    monkeypatch.setattr(study, "run", lambda *args: called.append(args))
    study.main()
    assert called == [("source", "funding", "native", "output")]
