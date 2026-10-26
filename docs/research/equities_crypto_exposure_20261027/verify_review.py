"""Recheck public source lineage, clocks, and HAC math; not external attestation."""

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equity_crypto_exposure_20261027_v001"
DATA = ROOT / "data/research/equity_crypto_exposure_20261027_v001"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


reg = json.loads((RUN / "registration.json").read_text())
assert all(sha(ROOT / p) == h for p, h in reg["inputs_sha256"].items())
assert (
    sha(RUN / "analysis_source.py")
    == reg["source_sha256"]
    == sha(ROOT / "tradingagents/research/equities/crypto_exposure.py")
)
manifest = json.loads((DATA / "coinbase-source-manifest.json").read_text())
assert (
    manifest["requested_hour_count"] == 13080 and manifest["observed_requested_hour_count"] == 13077
)
assert manifest["complete"] is False
assert manifest["missing_hours_utc"] == [
    "2023-03-04T18:00:00+00:00",
    "2023-03-04T19:00:00+00:00",
    "2023-03-04T20:00:00+00:00",
]
normalized = json.loads((DATA / "hourly-candles.json").read_text())
assert len(normalized) == manifest["observed_requested_hour_count"]
expected_hours = set(
    range(
        int(datetime(2022, 12, 30, tzinfo=UTC).timestamp()),
        int(datetime(2024, 6, 27, tzinfo=UTC).timestamp()),
        3600,
    )
)
missing_hours = {int(datetime.fromisoformat(x).timestamp()) for x in manifest["missing_hours_utc"]}
assert {e["candle"][0] for e in normalized} == expected_hours - missing_hours
collection = json.loads((DATA / "collection-plan.json").read_text())
assert collection["source_sha256"] == sha(DATA / "collector_source.py")
probes = json.loads((DATA / "binance-fixed-probes.json").read_text())
for probe in probes["records"]:
    assert (
        json.loads(probe["result"]["content"][0]["text"])["result"]
        == probe["result"]["structuredContent"]["result"]
    )
raw = {}
for receipt in manifest["receipts"]:
    path = ROOT / receipt["file"]
    assert sha(path) == receipt["sha256"]
    raw[receipt["file"]] = json.loads(path.read_text())
marks = json.loads((RUN / "clock-marks.json").read_text())
calendar = {
    r["date"]: r
    for r in csv.DictReader(
        (ROOT / "data/research/equity_pit_20261026_v001/prices/signal_sessions.csv").open()
    )
}
for mark in marks:
    cutoff = datetime.fromisoformat(calendar[mark["date"]]["close"])
    assert (
        cutoff
        == datetime.fromisoformat(mark["stock_close_utc"])
        == datetime.fromisoformat(mark["bucket_end_exclusive_utc"])
    )
    assert cutoff - timedelta(hours=1) == datetime.fromisoformat(mark["bucket_start_utc"])
    assert (
        cutoff.astimezone(ZoneInfo("America/New_York")).isoformat() == mark["stock_close_new_york"]
    )
    for ref in mark["raw_lineage"]:
        candle = raw[ref["file"]][ref["row"]]
        assert (
            candle[0] == int((cutoff - timedelta(hours=1)).timestamp())
            and candle[4] == mark["BTCUSD_close"]
        )
        assert ref["sha256"] == sha(ROOT / ref["file"])
assert len(marks) == 373
by_day = {m["date"]: m for m in marks}
assert datetime.fromisoformat(by_day["2023-07-03"]["stock_close_utc"]).hour == 17
assert datetime.fromisoformat(by_day["2023-11-24"]["stock_close_utc"]).hour == 18
assert datetime.fromisoformat(by_day["2023-03-10"]["stock_close_utc"]).hour == 21
assert datetime.fromisoformat(by_day["2023-03-13"]["stock_close_utc"]).hour == 20
incident = json.loads((DATA / "coverage-probes/coinbase-incident.json").read_text())["incident"]
assert incident["id"] == "5rzjgq4mgdy3" and incident["status"] == "resolved"
closed = [
    u for u in incident["incident_updates"] if "all functionality has been disabled" in u["body"]
]
opened = [
    u
    for u in incident["incident_updates"]
    if u["body"].rstrip().endswith("BTC-USD") and "full-trading mode" in u["body"]
]
assert len(closed) == len(opened) == 1
closed_at = datetime.fromisoformat(closed[0]["created_at"]).astimezone(UTC)
opened_at = datetime.fromisoformat(opened[0]["created_at"]).astimezone(UTC)
assert all(
    closed_at <= datetime.fromisoformat(missing) < opened_at
    for missing in manifest["missing_hours_utc"]
)
frame = pd.read_csv(RUN / "aligned-returns.csv", index_col="date")
assert list(frame.index) == [m["date"] for m in marks[1:]]
expected = np.array(
    [marks[i]["BTCUSD_close"] / marks[i - 1]["BTCUSD_close"] - 1 for i in range(1, len(marks))]
)
assert np.max(np.abs(frame["BTCUSD_close_to_close_return"].to_numpy() - expected)) < 1e-14
reported = json.loads((RUN / "diagnostics.json").read_text())
assert (
    reported["full_hourly_coverage_complete"] is False
    and not reported["profit_goal_met"]
    and not reported["fit_or_execute_allocations"]
)
math_checks = {}
for outcome in ["focal_net_account", "focal_minus_fixed_equity_control", "COIN_raw_price"]:
    y = frame[outcome].to_numpy()
    for label, cols in [
        ("two_factor", ["SPY_gross_total_return", "QQQ_minus_SPY_gross_return"]),
        (
            "three_factor",
            [
                "SPY_gross_total_return",
                "QQQ_minus_SPY_gross_return",
                "BTCUSD_close_to_close_return",
            ],
        ),
    ]:
        X = np.column_stack([np.ones(len(frame)), frame[cols].to_numpy()])
        n, k = X.shape
        inverse = np.linalg.inv(X.T @ X)
        beta = inverse @ X.T @ y
        score = X * (y - X @ beta)[:, None]
        S = score.T @ score
        for lag in range(1, 6):
            gamma = score[lag:].T @ score[:-lag]
            S += (1 - lag / 6) * (gamma + gamma.T)
        covariance = n / (n - k) * inverse @ S @ inverse
        se = np.sqrt(np.diag(covariance))
        critical = t.ppf(0.975, n - k)
        ci = np.column_stack([beta - critical * se, beta + critical * se])
        z = reported["models"][outcome][label]["coefficients"]
        names = ["intercept_daily_return", *cols]
        error = max(
            np.max(np.abs(beta - [z[c]["value"] for c in names])),
            np.max(np.abs(se - [z[c]["standard_error_hac"] for c in names])),
            np.max(np.abs(ci - np.array([z[c]["confidence95"] for c in names]))),
        )
        assert error < 1e-10
        math_checks[outcome + "_" + label] = {
            "max_coefficient_se_interval_error": float(error),
            "passed": True,
        }
prior = json.loads(
    (ROOT / "results/research/equity_depth_attribution_20261026_v002/diagnostics.json").read_text()
)
for old_key, new_key in [
    ("net_account_regression", "focal_net_account"),
    ("paired_net_difference_regression", "focal_minus_fixed_equity_control"),
]:
    for name, coefficient in prior[old_key]["coefficients"].items():
        assert (
            abs(
                coefficient["value"]
                - reported["models"][new_key]["two_factor"]["coefficients"][name]["value"]
            )
            < 1e-12
        )
result = {
    "schema": "equity-crypto-exposure-verification.v1",
    "role": "Internal source, clock and independent matrix-method checks; not independent external strategy attestation.",
    "source_batch_hashes_verified": len(raw),
    "stock_close_endpoints_raw_bound": len(marks),
    "same_frozen_return_observations": len(frame),
    "missing_hour_count_retained": 3,
    "official_maintenance_interval_contains_all_three_missing_hours": True,
    "BTC_USD_full_trading_resume_posted_at": opened_at.isoformat(),
    "first_all_functionality_disabled_posted_at": closed_at.isoformat(),
    "source_amendment_explicit": True,
    "no_required_endpoint_removed": True,
    "old_two_factor_results_unchanged": True,
    "matrix_HAC5_crosschecks": math_checks,
    "new_economic_accounts": 0,
    "external_independent_attestation": False,
    "profit_goal_met": False,
}
(ROOT / "docs/research/equities_crypto_exposure_20261027/verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
