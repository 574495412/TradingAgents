"""仅合成分钟与事件的双表验证，不读取真实新标签或执行账户。"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.research import historical_panel as panel
from tradingagents.research.quant.artifacts import read, save, sha, verify_receipts
from tradingagents.research.quant.regression import FEATURES as OLD_NAMES
from tradingagents.research.quant.regression_run import features_from_minutes


def minutes(n=700, start=0):
    clock = np.arange(n, dtype=np.int64) * panel.MINUTE + start
    index = np.arange(n, dtype=float)
    opening = 100 + 0.01 * index + 0.1 * np.sin(index / 3)
    return pd.DataFrame(
        {
            "open": opening,
            "high": opening + 1,
            "low": opening - 1,
            "close": opening + 0.1 * np.cos(index / 5),
            "quote_volume": 20 + index % 7,
            "taker_buy_quote": (20 + index % 7) * (0.2 + 0.01 * (index % 11)),
        },
        index=clock,
    )


def contract(**changes):
    row = {
        "asset_class": "CRYPTO_PERPETUAL",
        "snapshot_symbol_found": True,
        "qualification_interval_known": True,
        "onboard_ms": 0,
        "delivery_ms": 0,
        "original_748_member": True,
    }
    row.update(changes)
    return row


def test_all_seven_match_old_formula_without_computing_targets():
    frame = minutes()
    assert panel.LEGACY_NAMES == OLD_NAMES
    actual = panel.price_features(frame)
    expected = features_from_minutes(frame)[list(OLD_NAMES)]
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert not any("target" in name for name in actual)
    changed = frame.copy()
    changed.loc[changed.index[400] :, ["open", "close", "quote_volume", "taker_buy_quote"]] *= 3
    pd.testing.assert_frame_equal(actual.iloc[:400], panel.price_features(changed).iloc[:400])


def test_gap_zero_volume_and_warmup_preserve_unknown():
    frame = minutes()
    frame.loc[100 * panel.MINUTE, :] = np.nan
    frame.loc[300 * panel.MINUTE : 320 * panel.MINUTE, ["quote_volume", "taker_buy_quote"]] = 0
    actual = panel.price_features(frame)
    assert actual.iloc[:60].isna().all().all()
    assert actual.iloc[100:161].isna().all().all()
    assert actual.iloc[161].notna().all()
    assert np.isnan(actual.iloc[311]["flow_12m"])
    assert actual.iloc[311]["log_quote_volume_60m"] >= 0
    with pytest.raises(ValueError, match="压缩"):
        panel.price_features(frame.drop(index=panel.MINUTE))
    with pytest.raises(ValueError, match="唯一"):
        panel.price_features(pd.concat([frame.iloc[:1], frame]))


@pytest.mark.parametrize("horizon", panel.HORIZONS)
def test_open_to_open_labels_endpoint_clock_and_missing_interior(horizon):
    frame = minutes()
    clock = frame.index[100]
    labels = panel.future_labels(frame, contract())
    expected = frame.iloc[100 + horizon + 2]["open"] / frame.iloc[102]["open"] - 1
    assert labels.loc[clock, f"target_{horizon}"] == expected
    assert labels.loc[clock, f"label_end_{horizon}_ms"] == clock + (horizon + 1) * panel.MINUTE
    assert labels.loc[clock, f"holding_path_observed_{horizon}"]
    frame.iloc[110] = np.nan
    gapped = panel.future_labels(frame, contract())
    assert gapped.loc[clock, f"target_{horizon}"] == expected
    assert gapped.loc[clock, f"endpoint_observed_{horizon}"]
    assert not gapped.loc[clock, f"holding_path_observed_{horizon}"]
    assert labels.iloc[-1][f"label_status_{horizon}"] == "UNKNOWN_OPEN_ENDPOINT"


def test_first_decision_labels_have_future_endpoints_and_complete_status():
    frame = minutes(start=panel.START_MS + panel.MINUTE)
    labeled = frame.reindex(np.concatenate(([panel.START_MS], frame.index.to_numpy())))
    labels = panel.future_labels(labeled, contract())
    for horizon in panel.HORIZONS:
        row = labels.loc[panel.START_MS]
        expected = (
            frame.loc[panel.START_MS + (horizon + 2) * panel.MINUTE, "open"]
            / frame.loc[panel.START_MS + 2 * panel.MINUTE, "open"]
            - 1
        )
        assert row[f"target_{horizon}"] == expected
        assert row[f"label_end_{horizon}_ms"] == panel.START_MS + (horizon + 1) * panel.MINUTE
        assert row[f"endpoint_observed_{horizon}"]
        assert row[f"endpoint_eligible_{horizon}"]
        assert row[f"holding_path_observed_{horizon}"]
        assert row[f"label_status_{horizon}"] == "KNOWN"
    assert panel.price_features(frame).reindex([panel.START_MS]).isna().all().all()


def test_future_endpoint_qualification_does_not_remove_current_feature_member():
    frame = minutes()
    clock = frame.index[100]
    known = contract(delivery_ms=clock + 100 * panel.MINUTE)
    current, _ = panel.qualification([clock], known)
    assert current[0] and panel.price_features(frame).loc[clock].notna().all()
    labels = panel.future_labels(frame, known)
    assert labels.loc[clock, "endpoint_observed_240"]
    assert not labels.loc[clock, "endpoint_eligible_240"]
    assert np.isnan(labels.loc[clock, "target_240"])
    assert labels.loc[clock, "label_status_240"] == "OUTSIDE_STATIC_INTERVAL"
    assert np.isfinite(labels.loc[clock, "target_20"])
    assert labels.loc[clock, "holding_path_observed_240"]
    unknown = panel.future_labels(frame, contract(snapshot_symbol_found=False))
    assert unknown.loc[clock, "endpoint_observed_20"]
    assert unknown.loc[clock, "label_status_20"] == "UNKNOWN_QUALIFICATION"
    assert np.isnan(unknown.loc[clock, "target_20"])


def test_static_interval_strict_boundaries_and_unknown_type():
    known = contract(onboard_ms=10, delivery_ms=20)
    flags, states = panel.qualification([9, 10, 11, 19, 20], known)
    assert flags.tolist() == [False, False, True, True, False]
    assert states[0] == "OUTSIDE_STATIC_INTERVAL"
    flags, states = panel.qualification([11], contract(asset_class="UNKNOWN"))
    assert not flags[0] and states[0] == "UNKNOWN_TYPE"


def coverage(symbol="A", ready=(True, True)):
    return [
        {
            "symbol": symbol,
            "month": month,
            "start_ms": panel.month_bounds(month)[0],
            "end_exclusive_ms": panel.month_bounds(month)[1],
            "month_ready_under_source_assumptions": ready[i],
            "observed_regular_events": 0,
        }
        for i, month in enumerate(panel.MONTHS)
    ]


def event(clock, rate="0.0001", kind="Regular", **quality):
    return {
        "symbol": "A",
        "time_ms": int(clock),
        "rate": rate,
        "type": kind,
        "quality": {
            "explicit_type_verified": True,
            "identity_conflict": False,
            "trusted_typed_fields": True,
            **quality,
        },
    }


def counted(events, ready=(True, True)):
    rows = coverage(ready=ready)
    for row in rows:
        row["observed_regular_events"] = sum(
            e["type"] == "Regular" and row["start_ms"] <= e["time_ms"] < row["end_exclusive_ms"]
            for e in events
        )
    return rows


def test_funding_lag_zero_actual_event_special_and_unknown_month():
    split = panel.month_bounds("2026-09")[0]
    day = 24 * panel.HOUR
    events = [event(split - day + 1, "0"), event(split + 1, "0"), event(split + 1, "-9", "Special")]
    clocks = [split + 300000, split + 300001]
    result = panel.regular_factors(clocks, "A", events, counted(events))
    assert np.isnan(result.iloc[0][panel.FUNDING_NAMES[1]])
    assert result.iloc[1].eq(0).all()
    unknown = panel.regular_factors(clocks, "A", events, counted(events, (False, True)))
    assert unknown.isna().all().all()
    empty = panel.regular_factors(clocks, "A", [], counted([]))
    assert empty.isna().all().all()


@pytest.mark.parametrize("change", [{"identity_conflict": True}, {"trusted_typed_fields": False}])
def test_trusted_month_cannot_hide_conflicts_or_duplicate_regular(change):
    stamp = panel.START_MS + 1
    events = [event(stamp, **change)]
    with pytest.raises(ValueError, match="冲突"):
        panel.regular_factors([stamp + 3 * 24 * panel.HOUR], "A", events, counted(events))
    duplicate = [event(stamp), event(stamp)]
    with pytest.raises(ValueError, match="重复"):
        panel.regular_factors([stamp + 3 * 24 * panel.HOUR], "A", duplicate, counted(duplicate))


def test_metadata_preserves_unknown_and_never_infers_names():
    rows = [
        {
            "symbol": "STOCKUSDT",
            "snapshot_symbol_found": False,
            "snapshot": None,
            "original_748_member": False,
        },
        {
            "symbol": "A",
            "snapshot_symbol_found": True,
            "snapshot": {
                "symbol": "A",
                "contractType": "PERPETUAL",
                "onboardDate": 1,
                "deliveryDate": 0,
            },
            "original_748_member": True,
        },
    ]
    result = panel.universe_from_metadata(["A", "STOCKUSDT"], ["A"], rows)
    assert result["STOCKUSDT"]["asset_class"] == "UNKNOWN"
    assert result["A"]["asset_class"] == "CRYPTO_PERPETUAL"
    with pytest.raises(ValueError, match="双射"):
        panel.universe_from_metadata(["A", "STOCKUSDT"], ["A"], rows[:1])


def test_normalized_source_has_exact_fields_gaps_and_no_silent_dedup(tmp_path):
    stamp = panel.START_MS
    path = tmp_path / "minutes.csv.gz"

    def write(times):
        values = [
            [t, 100, 101, 99, 100.1, 2, t + 59999, 200, 3, 1, 100, 0, t + 60000, "A"] for t in times
        ]
        pd.DataFrame(
            values, columns=[*panel.RAW_COLUMNS, "completed_boundary_ms", "symbol"]
        ).to_csv(path, index=False, compression="gzip")

    write([stamp, stamp + 2 * panel.MINUTE])
    source = {"path": path, "month": "2026-08", "profile": {"rows": 2}}
    loaded = panel.load_minutes("A", [source])
    assert len(loaded) == 87840 and loaded.iloc[1].isna().all()
    assert loaded.iloc[0]["taker_buy_quote"] == 100
    assert set(panel.RAW_COLUMNS) <= set(loaded)
    write([stamp, stamp])
    with pytest.raises(ValueError, match="重复"):
        panel.load_minutes("A", [source])
    with gzip.open(path, "wt") as stream:
        stream.write("wrong\n")
    with pytest.raises(ValueError, match="字段"):
        panel.load_minutes("A", [source])


def seal(root):
    save(
        root / "receipts.json",
        {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
    )


def synthetic_sources(tmp_path):
    source, typed = tmp_path / "capture", tmp_path / "typed"
    source.mkdir()
    typed.mkdir()
    symbols = ["A", "UNKNOWNUSDT"]
    tasks = [
        {"id": f"1m_{s}_{m}", "symbol": s, "month": m, "kind": "1m"}
        for s in symbols
        for m in panel.MONTHS
    ]
    plan = {
        "schema": "ta-fixed-historical-public-capture.v1",
        "start_ms": panel.START_MS,
        "end_exclusive_ms": panel.END_MS,
        "months": list(panel.MONTHS),
        "symbols": symbols,
        "original_universe": ["A"],
        "archive_tasks": tasks,
        "orders_enabled": False,
        "leader_labels_read": False,
        "returns_evaluated": False,
    }
    save(source / "capture_plan.json", plan)
    save(source / "result.json", {"capture_completed": True})
    (source / "inputs").mkdir()
    rows = [
        {
            "symbol": s,
            "snapshot_symbol_found": s == "A",
            "snapshot": {
                "symbol": s,
                "contractType": "PERPETUAL",
                "onboardDate": 0,
                "deliveryDate": 0,
            }
            if s == "A"
            else None,
            "original_748_member": s == "A",
        }
        for s in symbols
    ]
    save(source / "inputs/eligibility_symbol_metadata.json", rows)
    comparisons = [
        {"symbol": s, "month": m, "status": "UNPROVEN_MISSING_SOURCE"}
        for s in symbols
        for m in panel.MONTHS
    ]
    save(source / "funding_month_comparisons.json", comparisons)
    for task in tasks:
        folder = source / "archive_jobs" / task["id"]
        folder.mkdir(parents=True)
        save(
            folder / "attempt_0000.json",
            {
                **task,
                "task_id": task["id"],
                "plan_sha256": sha(source / "capture_plan.json"),
                "status": "NOT_LISTED",
                "files": {},
                "profile": None,
            },
        )
    seal(source)
    save(
        typed / "protocol.json",
        {
            "schema": "ta-historical-typed-funding-view.v1",
            "input_identities": {str(source / "receipts.json"): sha(source / "receipts.json")},
        },
    )
    save(typed / "result.json", {"status": "COMPLETED_WITH_UNKNOWN"})
    cov = [r for s in symbols for r in coverage(s, (False, False))]
    for row in cov:
        row["original_comparison"] = next(
            r for r in comparisons if (r["symbol"], r["month"]) == (row["symbol"], row["month"])
        )
    save(typed / "coverage.json", cov)
    (typed / "typed_cash_events.jsonl").write_text("")
    seal(typed)
    return source, typed


def test_double_table_run_retains_all_unknown_rows_and_seals_only_actual_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(panel, "SYMBOL_COUNT", 2)
    monkeypatch.setattr(panel, "ORIGINAL_COUNT", 1)
    source, typed = synthetic_sources(tmp_path)
    output = tmp_path / "result"
    result = panel.run(source, typed, output)
    assert result["status"] == "COMPLETED_WITH_UNKNOWN"
    features, labels = (
        pd.read_parquet(output / name) for name in ("features.parquet", "labels.parquet")
    )
    assert len(features) == 2 * 1464 and len(labels) == len(features)
    assert features[list(panel.IDENTITY)].equals(labels[list(panel.IDENTITY)])
    assert features[list(panel.FEATURES)].isna().all().all()
    assert not features.loc[features.symbol == "UNKNOWNUSDT", "feature_eligible"].any()
    assert features.loc[features.symbol == "A", "feature_eligible"].all()
    assert not any(name.startswith(("target", "label", "holding", "endpoint")) for name in features)
    protocol = read(output / "protocol.json")
    measured = read(output / "coverage.json")
    for axis in ("by_utc_date", "by_utc_hour", "by_asset_clock"):
        assert sum(row["scheduled_rows"] for row in measured[axis]) == len(features)
        assert sum(row["finite_nine"] for row in measured[axis]) == 0
    for horizon in panel.HORIZONS:
        assert sum(measured["label_status_counts"][str(horizon)].values()) == len(features)
        first = labels.loc[labels.clock_ms == panel.START_MS]
        assert (
            first[f"label_end_{horizon}_ms"].eq(panel.START_MS + (horizon + 1) * panel.MINUTE).all()
        )
        assert first[f"label_status_{horizon}"].notna().all()
    assert protocol["study_plan_sha256"] == sha(
        Path(panel.__file__).parents[2] / "research/quant/plans/extended_funding_plan.md"
    )
    verify_receipts(output)
    with pytest.raises(FileExistsError):
        panel.run(source, typed, output)


def test_receipt_tamper_is_rejected_before_calculation(tmp_path):
    source, _ = synthetic_sources(tmp_path)
    bound = panel.BoundInputs(source, {})
    (source / "result.json").write_text(json.dumps({"capture_completed": False}))
    with pytest.raises(ValueError, match="身份变化"):
        bound.add("result.json")


def test_changed_source_after_freeze_retains_failed_partial_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "SYMBOL_COUNT", 2)
    monkeypatch.setattr(panel, "ORIGINAL_COUNT", 1)
    source, typed = synthetic_sources(tmp_path)
    original = panel.price_features

    def changed(frame):
        (source / "result.json").write_text('{"capture_completed":false}')
        return original(frame)

    monkeypatch.setattr(panel, "price_features", changed)
    output = tmp_path / "failed"
    result = panel.run(source, typed, output)
    assert result["status"] == "FAILED_INTEGRITY"
    assert (output / "partial_features.parquet").exists()
    assert (output / "partial_labels.parquet").exists()
    assert not (output / "features.parquet").exists()
    assert read(output / "errors.json")[0]["stage"] == "PANEL"
    verify_receipts(output)


def test_missing_month_delays_funding_change_until_full_new_coverage():
    split = panel.month_bounds("2026-09")[0]
    events = [event(split + 1), event(split + 24 * panel.HOUR + 1, "-.0002")]
    clocks = [split + 48 * panel.HOUR + 300000, split + 48 * panel.HOUR + 300001]
    result = panel.regular_factors(clocks, "A", events, counted(events, (False, True)))
    assert result.iloc[0][panel.FUNDING_NAMES[1]] == pytest.approx(0.0003)
    assert np.isnan(result.iloc[1][panel.FUNDING_NAMES[0]])
    assert np.isnan(result.iloc[1][panel.FUNDING_NAMES[1]])
