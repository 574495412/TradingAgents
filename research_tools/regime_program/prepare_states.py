"""Seal causal market states and official event availability before account tests."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from factors.common.github_formulas import daily
from factors.common.market_states import classify_market, event_risk
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources


def prepare(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    data = Path(plan["data"])
    sources = Path(plan["source_pack"])
    source_receipt = read(sources / "completion.json")
    if sha(sources / "source_catalog.json") != source_receipt["catalog_sha256"]:
        raise ValueError("Source catalog changed")
    source_catalog = read(sources / "source_catalog.json")
    for record in source_catalog["records"]:
        if sha(Path(record["local"])) != record["sha256"]:
            raise ValueError("Pinned reference changed")
    news = Path("data/research/factor_lab_news_20261025_v002")
    paths = [
        Path(__file__),
        plan_path,
        Path("factors/common/market_states.py"),
        Path("factors/common/github_formulas.py"),
        Path("research_tools/github_factor_study/run.py"),
        Path("research_tools/factor_lab/io.py"),
        data / "completion.json",
        sources / "source_catalog.json",
        news / "events.json",
        news / "coverage.json",
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "before_state_computation": True,
            "before_account_returns": True,
        },
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for filename, checksum in read(data / "completion.json")["manifest"].items():
        if Path(filename).name != filename or sha(data / filename) != checksum:
            raise ValueError("Sealed market data changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    frames = {s: daily(pd.read_parquet(data / (s + "_5m.parquet"))) for s in symbols}
    states, eligible = classify_market(frames, plan["state_parameters"])
    checks = []
    for cut in (450, 730, 1000):
        prefix = {s: f.iloc[:cut] for s, f in frames.items()}
        small, small_eligible = classify_market(prefix, plan["state_parameters"])
        pd.testing.assert_frame_equal(states.iloc[:cut], small)
        pd.testing.assert_frame_equal(eligible.iloc[:cut], small_eligible)
        future = {s: f.copy() for s, f in frames.items()}
        for f in future.values():
            f.iloc[cut:, f.columns.get_loc("close")] *= 1000
            f.iloc[cut:, f.columns.get_loc("quote_volume")] = 0
        changed, _ = classify_market(future, plan["state_parameters"])
        pd.testing.assert_frame_equal(states.iloc[: cut + 1], changed.iloc[: cut + 1])
        checks.append(
            {
                "prefix_rows": cut,
                "prefix_equal": True,
                "future_and_current_day_perturbation_equal": True,
            }
        )
    events = read(news / "events.json")
    for event in events:
        if sha(news / event["raw_file"]) != event["sha256"]:
            raise ValueError("Official event source changed")
    clocks = pd.date_range("2024-01-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    flags = event_risk(clocks, events)
    states.to_parquet(root / "states_daily.parquet")
    eligible.to_parquet(root / "eligible_daily.parquet")
    flags.to_frame("covered_fomc_post_release").to_parquet(root / "event_risk_hourly.parquet")
    profile = []
    for year in (2024, 2025, 2026):
        sample = states.loc[states.index.year == year]
        profile.append(
            {
                "year": year,
                "observed_days": len(sample),
                "states": sample.state.value_counts().to_dict(),
                "minimum_eligible": float(sample.eligible_count.min()),
                "maximum_eligible": float(sample.eligible_count.max()),
            }
        )
    save(
        root / "summary.json",
        {
            "schema": "causal-market-state-evidence.v1",
            "profile": profile,
            "source_records": len(source_catalog["records"]),
            "prefix_checks": checks,
            "verified_official_releases": len(events),
            "flagged_hourly_decisions": int(flags.sum()),
            "event_coverage": read(news / "coverage.json"),
            "account_returns_computed": False,
            "strategy_profit_proven": False,
            "orders_enabled": False,
            "production_eligible": False,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    filenames = [
        "states_daily.parquet",
        "eligible_daily.parquet",
        "event_risk_hourly.parquet",
        "summary.json",
        "plan.json",
        "registration.json",
    ]
    save(
        root / "completion.json",
        {
            "manifest": {name: sha(root / name) for name in filenames},
            "summary_sha256": sha(root / "summary.json"),
        },
    )
    print(
        {
            "root": str(root),
            "profile": profile,
            "causality_checks": len(checks),
            "event_flags": int(flags.sum()),
            "seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
