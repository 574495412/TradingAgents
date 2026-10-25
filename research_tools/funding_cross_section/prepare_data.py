"""Seal actual past funding and a separate API price revision before return scoring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(collection, output):
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    registered = read(collection / "collection_plan.json")
    if registered["plan_sha256"] != sha(plan_path):
        raise ValueError("Preregistered study changed")
    parent = Path(plan["data"])
    funding = Path(plan["parent_actual_funding"])
    verify_manifest(parent)
    verify_manifest(funding)
    recovery_path = Path(__file__).with_name("collection_recovery.json")
    recovery = read(recovery_path)
    if recovery["parent_plan_sha256"] != sha(plan_path) or recovery[
        "failure_receipt_sha256"
    ] != sha(collection / "raw/bars_BNBUSDT.json"):
        raise ValueError("Transport recovery identity changed")
    save(
        output / "plan.json",
        {
            "study_plan_sha256": sha(plan_path),
            "collection_plan_sha256": sha(collection / "collection_plan.json"),
            "recovery_plan_sha256": sha(recovery_path),
            "source_code_sha256": sha(Path(__file__)),
            "parent_market_completion_sha256": sha(parent / "completion.json"),
            "parent_funding_completion_sha256": sha(funding / "completion.json"),
            "financial_returns_computed": False,
        },
    )
    symbols = registered["symbols"]
    reports = []
    bindings = {}
    expected = pd.date_range("2025-01-29", periods=288, freq="5min", tz="UTC")
    for s in symbols:
        path = collection / "raw" / ("funding_" + s + ".json")
        raw = read(path)
        bindings[str(path)] = sha(path)
        if raw["response"].get("isError"):
            raise ValueError("Funding collector failed: " + s)
        records = raw["response"]["structuredContent"]["result"]
        if not 0 < len(records) < 1000:
            raise ValueError("Funding missing or page truncated")
        f = pd.DataFrame(
            {
                "funding_rate": [float(r["fundingRate"]) for r in records],
                "mark": [float(r["markPrice"]) for r in records],
            },
            index=pd.DatetimeIndex(
                pd.to_datetime([r["fundingTime"] for r in records], unit="ms", utc=True),
                name="date",
            ),
        )
        if (
            f.index.has_duplicates
            or not f.index.is_monotonic_increasing
            or not np.isfinite(f.to_numpy()).all()
            or (f.mark <= 0).any()
            or f.index[0] < pd.Timestamp("2024-12-25", tz="UTC")
            or f.index[-1] >= pd.Timestamp("2025-01-01", tz="UTC")
        ):
            raise ValueError("Past funding clock/mark integrity failed")
        old = pd.read_parquet(parent / (s + "_funding.parquet")).loc[
            lambda x: (
                (x.index >= pd.Timestamp("2024-12-25", tz="UTC"))
                & (x.index < pd.Timestamp("2025-01-01", tz="UTC"))
            )
        ]
        if not f.index.floor("h").equals(old.index) or not np.allclose(
            f.funding_rate, old.funding_rate, atol=1e-12, rtol=0
        ):
            raise ValueError("Past official versus sealed funding calendar/rates differ")
        combined = pd.concat([f, pd.read_parquet(funding / (s + "_exact.parquet"))])
        if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
            raise ValueError("Combined funding calendar invalid")
        combined.to_parquet(output / (s + "_exact.parquet"))
        path = collection / "raw" / ("bars_" + s + ".json")
        raw = read(path)
        if raw["response"].get("isError"):
            bindings[str(path)] = sha(path)
            if s != "BNBUSDT":
                raise ValueError("Unexpected bar transport failure")
            failed_retry = collection / "recovery/bars_BNBUSDT_attempt1.json"
            bindings[str(failed_retry)] = sha(failed_retry)
            if not read(failed_retry)["response"].get("isError"):
                raise ValueError("First retry unexpectedly succeeded")
            path = collection / "recovery/bars_BNBUSDT_attempt2.json"
            raw = read(path)
        bindings[str(path)] = sha(path)
        if raw["response"].get("isError"):
            raise ValueError("Revised market snapshot incomplete")
        records = raw["response"]["structuredContent"]["result"]
        r = pd.DataFrame(records)
        if r.shape != (288, 12):
            raise ValueError("Incomplete API revision calendar")
        index = pd.DatetimeIndex(pd.to_datetime(r[0].to_numpy(), unit="ms", utc=True), name="date")
        if not index.equals(expected):
            raise ValueError("API revision timestamps differ")
        replacement = pd.DataFrame(
            {
                name: r[i].astype(float).to_numpy()
                for name, i in [
                    ("open", 1),
                    ("high", 2),
                    ("low", 3),
                    ("close", 4),
                    ("volume", 5),
                    ("quote_volume", 7),
                ]
            },
            index=index,
        )
        if (
            not np.isfinite(replacement.to_numpy()).all()
            or (replacement[["open", "high", "low", "close"]] <= 0).any().any()
            or (replacement[["volume", "quote_volume"]] < 0).any().any()
            or (replacement.high < replacement[["open", "low", "close"]].max(axis=1)).any()
            or (replacement.low > replacement[["open", "high", "close"]].min(axis=1)).any()
        ):
            raise ValueError("API revision numerical integrity failed")
        replacement.to_parquet(output / (s + "_api_revision_5m.parquet"))
        original = pd.read_parquet(parent / (s + "_5m.parquet")).loc[index]
        delta = (original[replacement.columns] - replacement).abs()
        changed = (delta > np.maximum(1e-8, original[replacement.columns].abs() * 1e-12)).any(
            axis=1
        )
        reports.append(
            {
                "symbol": s,
                "past_events": len(f),
                "combined_events": len(combined),
                "nominal_calendar_rates_match": True,
                "actual_marks_and_ms_retained": True,
                "api_revision_bars": len(replacement),
                "changed_bars": int(changed.sum()),
                "max_errors": {k: float(delta[k].max()) for k in delta},
                "api_daily_close_matches_archive": bool(
                    replacement.close.iloc[-1] == original.close.iloc[-1]
                ),
            }
        )
    save(output / "raw_source_bindings.json", bindings)
    save(
        output / "summary.json",
        {
            "schema": "pure-funding-sealed-input-data.v1",
            "symbols": len(symbols),
            "reports": reports,
            "past_funding_events": sum(r["past_events"] for r in reports),
            "combined_funding_events": sum(r["combined_events"] for r in reports),
            "initial_funding_requests": 40,
            "initial_bar_requests": 40,
            "additional_bar_transport_attempts": 2,
            "funding_data_proxy_marks": False,
            "source_versions": ["archive", "api_revision"],
            "source_versions_agree": all(r["changed_bars"] == 0 for r in reports),
            "financial_accounts": 0,
            "profit_goal_met": False,
            "returns_computed": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "data_preparation_complete": True,
            "financial_accounts": 0,
        },
    )
    print(
        {
            "symbols": len(symbols),
            "past_funding_events": sum(r["past_events"] for r in reports),
            "revised_bars": sum(r["changed_bars"] for r in reports),
            "financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.collection, a.output)
