"""Freeze exact inputs and a single stop-policy alternative before account runs."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.funding_pool_bridge.prepare import old_bars
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_stop_audit.stop_audit import audit_stops


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    inputs = plan["inputs"]
    for key in ["targets", "old_market", "exact_funding"]:
        verify_manifest(Path(inputs[key]))
    checks_path = Path(inputs["targets"] + "_audit.json")
    checks = read(checks_path)
    assert checks["passed"] and checks["input_completion_sha256"] == sha(
        Path(inputs["targets"]) / "completion.json"
    )
    engine_review = Path("results/research/trend_coststop_engine_review_20261027_v002")
    verify_manifest(engine_review)
    for path, digest in read(engine_review / "verification.json")["source_hashes"].items():
        assert sha(Path(path)) == digest
    sources = Path("data/research/forecast_sizing_sources_20261026_v001")
    source_review = read(sources / "source_review.json")
    assert (
        sha(sources / "source_review.json")
        == read(sources / "completion.json")["source_review_sha256"]
    )
    for row in source_review["records"]:
        assert sha(Path(row["local"])) == row["sha256"]
    target_path = Path(inputs["targets"]) / "trend_risk16_targets.parquet"
    symbols = read(Path(inputs["old_market"]) / "prepare_plan.json")["symbols"]
    paths = [
        p,
        Path(__file__),
        Path(__file__).with_name("stop_audit.py"),
        Path("research_tools/funding_pool_bridge/prepare.py"),
        Path("../freqtrade/research/crypto/sparse_calendar_v2.py"),
        Path("../freqtrade/research/crypto/sparse_coststop_v1.py"),
        Path("../freqtrade/research/crypto/sparse_sources_v2.py"),
        checks_path,
        target_path,
        engine_review / "completion.json",
        sources / "source_review.json",
        *[Path(v) / "completion.json" for v in inputs.values()],
    ]
    shutil.copyfile("docs/research/regime_program/targets.json", output / "protocol_snapshot.json")
    plan = {
        **plan,
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "symbols": symbols,
        "source_hashes": {str(x): sha(x) for x in paths},
        "protocol_snapshot_sha256": sha(output / "protocol_snapshot.json"),
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {"before_new_targets_and_returns": True, "plan_sha256": sha(output / "plan.json")},
    )
    scores = pd.read_parquet(target_path)
    a, b = (
        pd.Timestamp(plan["window"]["start"], tz="UTC"),
        pd.Timestamp(plan["window"]["end"], tz="UTC"),
    )
    calendar = pd.date_range(a, b, freq="h", inclusive="left")
    assert scores.index.equals(calendar)
    assert set(scores.filter(like="target_")) == {"target_" + s for s in symbols}
    assert np.isfinite(scores.to_numpy(dtype=float)).all()
    scores["rebalance"] = scores.index.hour == 0
    scores.to_parquet(output / "targets.parquet")
    references, comparison, bars = {}, [], {}
    for s in symbols:
        bp = Path(inputs["old_market"]) / (s + "_5m.parquet")
        op = Path(inputs["old_market"]) / (s + "_funding.parquet")
        ep = Path(inputs["exact_funding"]) / (s + "_exact.parquet")
        references[s] = {str(path): sha(path) for path in [bp, op, ep]}
        bars[s] = old_bars(bp, a - pd.Timedelta(hours=1), b)
        old, exact = pd.read_parquet(op).loc[a:b], pd.read_parquet(ep).loc[a:b]
        floor = exact.copy()
        floor.index = floor.index.floor("5min")
        aligned = old.index.equals(floor.index)
        common = old.index.intersection(floor.index)
        comparison.append(
            {
                "symbol": s,
                "old_events": len(old),
                "exact_events": len(exact),
                "same_nominal_calendar_after_floor": aligned,
                "actual_events_with_subbar_offset": int(
                    (exact.index != exact.index.floor("5min")).sum()
                ),
                "revised_mark_or_rate_rows": int(
                    (old.loc[common] != floor.loc[common]).any(axis=1).sum()
                ),
                "quality_price_false_rows": int((~bars[s].price_usable).sum()),
                "quality_execution_false_rows": int((~bars[s].execution_liquidity_usable).sum()),
            }
        )
    save(output / "market_bindings.json", references)
    save(output / "funding_source_comparison.json", comparison)
    legacy = (
        Path(inputs["legacy_run"]) / "accounts/continuous2025_2026_trend_risk16_cap10000_slip10"
    )
    result = read(legacy / "result.json")
    for name, digest in result["source_hashes"].items():
        assert sha(legacy / name) == digest
    diagnostic = audit_stops(read(legacy / "events.json"), result["open_positions"], bars, 0.08)
    save(output / "legacy_stop_trace.json", diagnostic)
    save(
        output / "summary.json",
        {
            "hourly_rows": len(scores),
            "frozen_signal_exactly_reused": True,
            "daily_rebalance_flags_added_without_signal_change": True,
            "legacy_stop_trace_passed": diagnostic["passed"],
            "legacy_retroactive_requests": len(diagnostic["retroactive_new_stop_requests"]),
            "new_financial_accounts_computed": 0,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {"passed": True, "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()}},
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
