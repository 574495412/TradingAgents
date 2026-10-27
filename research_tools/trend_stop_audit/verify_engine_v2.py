"""Real boundary fixtures for the versioned stop policy, not profit evidence."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor


def fixture():
    start, end = "2025-01-01T00:00:00Z", "2025-01-02T02:00:00Z"
    times = pd.date_range("2024-12-31T22:00:00Z", end, freq="5min", inclusive="left")
    price = np.full(len(times), 100.0)
    price[times >= pd.Timestamp("2025-01-02T00:00:00Z")] = 120.0
    price[times >= pd.Timestamp("2025-01-02T00:05:00Z")] = 105.0
    bars = pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1e7,
            "quote_volume": 1e9,
            "price_usable": True,
            "execution_liquidity_usable": True,
        },
        index=times,
    )
    calendar = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    scores = pd.DataFrame(
        {"target_BTCUSDT": 0.5, "risk_scale": 1.0, "rebalance": calendar.hour == 0}, index=calendar
    )
    scores.loc[scores.index >= pd.Timestamp("2025-01-02T00:00:00Z"), "target_BTCUSDT"] = 1.0
    rates = pd.DataFrame({"mark": [], "funding_rate": []}, index=pd.DatetimeIndex([], tz="UTC"))
    config = {
        "capital": 10000,
        "fee_bps": 5,
        "slippage_bps": 10,
        "participation": 0.01,
        "gross": 0.8,
        "max_positions": 10,
        "rebalance_hours": 24,
        "threshold": 0.0032,
        "stop_loss": 0.08,
        "drawdown_halt": 0.15,
        "weighted_cost_stop": False,
    }
    return {"BTCUSDT": bars}, {"BTCUSDT": rates}, scores, config, start, end


def worker(root):
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    engine = executor(str(Path("../freqtrade/research/crypto/sparse_coststop_v1.py").resolve()))
    bars, rates, scores, config, _, end = fixture()
    config["weighted_cost_stop"] = True
    split = "2025-01-02T00:05:00Z"
    result = engine.replay(
        bars, rates, scores, config, split, end, initial_state=read(root / "checkpoint.json")
    )
    result["curve"].to_parquet(root / "worker_curve.parquet", index=False)
    save(
        root / "worker.json",
        {
            "pid": os.getpid(),
            "events": result["events"],
            "positions": result["positions"],
            "state": result["resume_state"],
            "requests": result["requests"],
        },
    )


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    base_path = Path("../freqtrade/research/crypto/sparse_calendar_v2.py").resolve()
    new_path = base_path.with_name("sparse_coststop_v1.py")
    original, new = executor(str(base_path)), executor(str(new_path))
    bars, rates, scores, config, start, end = fixture()
    baseline = original.replay(bars, rates, scores, config, start, end)
    control = new.replay(bars, rates, scores, config, start, end)
    pd.testing.assert_frame_equal(baseline["curve"], control["curve"])
    assert baseline["events"] == control["events"] and baseline["positions"] == control["positions"]
    config["weighted_cost_stop"] = True
    full = new.replay(bars, rates, scores, config, start, end)
    assert any(e.get("reason") == "stop" for e in control["events"])
    assert not any(e.get("reason") == "stop" for e in full["events"])
    p = full["positions"]["BTCUSDT"]
    assert abs(p["stop"] - p["price"] * (1 - p["side"] * 0.08)) < 1e-10
    split = "2025-01-02T00:05:00Z"
    before = new.replay(bars, rates, scores, config, start, split)
    assert before["positions"] and before["positions"]["BTCUSDT"]["price"] > 100
    save(output / "checkpoint.json", before["resume_state"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research_tools.trend_stop_audit.verify_engine_v2",
            "--worker",
            str(output),
        ],
        check=True,
    )
    resumed = read(output / "worker.json")
    assert resumed["pid"] != os.getpid()
    joined = pd.concat(
        [before["curve"].iloc[:-1], pd.read_parquet(output / "worker_curve.parquet")],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(joined, full["curve"])
    assert before["events"] + resumed["events"] == full["events"]
    assert before["requests"] + resumed["requests"] == full["requests"]
    assert resumed["positions"] == full["positions"] and resumed["state"] == full["resume_state"]
    save(
        output / "verification.json",
        {
            "passed": True,
            "baseline_flag_false_matches_original": True,
            "weighted_cost_anchor_recomputed": True,
            "new_process_holding_restart_equal": True,
            "parent_pid": os.getpid(),
            "worker_pid": resumed["pid"],
            "source_hashes": {str(p): sha(p) for p in [Path(__file__), base_path, new_path]},
            "profit_evidence": False,
        },
    )
    save(
        output / "completion.json",
        {"passed": True, "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()}},
    )
    print(read(output / "verification.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    worker(a.worker) if a.worker else verify(a.output)
