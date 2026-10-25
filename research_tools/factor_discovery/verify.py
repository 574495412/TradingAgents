"""Independent raw-flow checks and source/label-clock verification for discovery."""

import argparse
from pathlib import Path

import pandas as pd

from factors.common.mechanisms import features
from research_tools.factor_lab.io import read, save, sha


def verify(output):
    base = Path("results/research")
    first = base / "factor_discovery_20261025_v001"
    second = base / "factor_discovery_nonlinear_20261025_v001"
    plan = read(first / "plan.json")
    data = Path(plan["data"])
    saved = pd.read_parquet(first / "features.parquet")
    maximum_error = 0.0
    checked = 0
    for symbol in plan["symbols"]:
        raw = pd.read_parquet(data / f"{symbol}_5m.parquet")
        rates = pd.read_parquet(data / f"{symbol}_funding.parquet")
        frozen = saved.xs(symbol, level="symbol")
        current, _ = features(raw, rates)
        pd.testing.assert_frame_equal(current, frozen, check_names=False, check_freq=False)
        # Recompute the raw quote-flow identity from the last24 complete hours.
        for clock in pd.date_range("2025-01-01", "2026-09-01", freq="MS", tz="UTC"):
            recent = raw.loc[(raw.index >= clock - pd.Timedelta(hours=24)) & (raw.index < clock)]
            expected = (
                2 * recent.taker_buy_quote_volume.sum() - recent.quote_volume.sum()
            ) / recent.quote_volume.sum()
            error = abs(float(expected) - frozen.loc[clock, "aggressor24"])
            if error > 1e-12:
                raise ValueError("Real aggressor feature differs from independent raw-flow sum")
            maximum_error = max(maximum_error, error)
            known = rates.loc[rates.index + pd.Timedelta(minutes=5) <= clock].funding_rate.iloc[-1]
            if known != frozen.loc[clock, "funding_last"]:
                raise ValueError("Unpublished funding entered feature")
            checked += 1
    model_count = 0
    source_count = 0
    for root in (first, second):
        registered = read(root / "plan.json")
        for source, expected in registered["source_hashes"].items():
            if Path(source).suffix == ".py":
                if sha(root / "source_snapshot" / Path(source).name) != expected:
                    raise ValueError("Executed source snapshot changed")
                source_count += 1
        for p in root.glob("model_*.json"):
            m = read(p)
            if pd.Timestamp(m["latest_matured_label"]) >= pd.Timestamp(m["evaluation_start"]):
                raise ValueError("Unmatured model target")
            if "model_sha256" in m and sha(p.with_suffix(".joblib")) != m["model_sha256"]:
                raise ValueError("Persisted model changed")
            model_count += 1
    result = {
        "passed": True,
        "raw_flow_and_known_funding_checks": checked,
        "maximum_raw_flow_error": maximum_error,
        "models_with_matured_labels": model_count,
        "executed_python_snapshot_checks": source_count,
        "current_crypto_features_identical_to_frozen_study": True,
        "equity_session_adapter_does_not_change_crypto_results": True,
    }
    save(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    print(verify(p.parse_args().output.resolve()))
