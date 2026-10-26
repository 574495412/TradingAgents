"""Actual-source dense/sparse parity and a nonflat on-disk process restoration."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    fq = Path("../freqtrade").resolve()
    scores = Path(
        "results/research/max_effect_inputs_20261026_v002/screen2025_momentum28_control_targets.parquet"
    )
    plan = {
        "schema": "sparse-engine-real-source-regression.v1",
        "start": "2025-01-01",
        "end": "2025-01-02",
        "split": "2025-01-01T08:05:00Z",
        "old_market": "data/research/github_factor_market_20261025_v001",
        "old_funding": "data/research/funding_cross_section_data_20261026_v001",
        "additional_market": "data/research/sparse_addition_market_20261026_v001",
        "bundle": "data/research/sparse_market_bundle_20261026_v001",
        "scores": str(scores),
        "registry": "results/research/sparse_identity_registry_20261026_v002/registry.json",
        "dense_engine": str(fq / "research/crypto/calendar_portfolio.py"),
        "sparse_engine": str(fq / "research/crypto/sparse_calendar_v2.py"),
        "config": read(Path("research_tools/sparse_market/plan.json"))["account"],
        "purpose": "Existing frozen momentum book first24h repeated only to verify execution mechanics. No new alpha or annualized result; not the expanded strategy experiment.",
        "profit_goal_met": False,
    }
    paths = [
        Path(__file__),
        Path(__file__).with_name("engineering_worker.py"),
        Path(__file__).with_name("identity.py"),
        scores,
        Path(plan["registry"]),
        Path(plan["dense_engine"]),
        Path(plan["sparse_engine"]),
        fq / "research/crypto/sparse_sources_v2.py",
        Path("research_tools/expanded_market/ingest.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
    ]
    paths += [
        Path(plan[k]) / "completion.json"
        for k in ["old_market", "old_funding", "additional_market", "bundle"]
    ]
    plan["source_hashes"] = {str(p): sha(p) for p in paths}
    save(output / "plan.json", plan)
    common = [
        sys.executable,
        "-m",
        "research_tools.sparse_market.engineering_worker",
        "--run",
        str(output),
    ]
    for mode in ["dense", "sparse", "extended"]:
        subprocess.run(
            common
            + [
                "--mode",
                mode,
                "--start",
                plan["start"],
                "--end",
                plan["end"],
                "--output",
                str(output / mode),
            ],
            check=True,
        )
    for mode in ["sparse", "extended"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(output / "dense/curve.parquet"),
            pd.read_parquet(output / mode / "curve.parquet"),
        )
        for key in ["events", "requests", "trades"]:
            if read(output / "dense" / (key + ".json")) != read(output / mode / (key + ".json")):
                raise ValueError("Actual-source dense/sparse regression differs: " + key)
    first, second = output / "checkpoint", output / "restored"
    subprocess.run(
        common
        + [
            "--mode",
            "extended",
            "--start",
            plan["start"],
            "--end",
            plan["split"],
            "--output",
            str(first),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            "--mode",
            "extended",
            "--start",
            plan["split"],
            "--end",
            plan["end"],
            "--restore",
            str(first / "resume_state.json"),
            "--output",
            str(second),
        ],
        check=True,
    )
    pd.testing.assert_frame_equal(
        pd.concat(
            [
                pd.read_parquet(first / "curve.parquet"),
                pd.read_parquet(second / "curve.parquet").iloc[1:],
            ],
            ignore_index=True,
        ),
        pd.read_parquet(output / "extended/curve.parquet"),
    )
    for key in ["events", "requests", "trades"]:
        if read(first / (key + ".json")) + read(second / (key + ".json")) != read(
            output / "extended" / (key + ".json")
        ):
            raise ValueError("New interpreter changes actual ledger")
    if read(second / "resume_state.json") != read(output / "extended/resume_state.json"):
        raise ValueError("New interpreter changes risk/inventory/final state")
    a, b = read(first / "receipt.json"), read(second / "receipt.json")
    if a["pid"] == b["pid"] or not b["restored_from_disk"] or b["open_positions_at_start"] <= 0:
        raise ValueError("Restoration was not a nonflat new interpreter")
    events = read(output / "extended/events.json")
    result = {
        "schema": plan["schema"],
        "passed": True,
        "real_source_window": [plan["start"], plan["end"]],
        "curve_points": len(pd.read_parquet(output / "extended/curve.parquet")),
        "financial_cells_repeated": 3,
        "actual_funding_events": sum(e["type"] == "funding" for e in events),
        "fills": sum(e["type"] != "funding" for e in events),
        "ledger_and_curve_exact_equality": True,
        "unused_unlisted_names_added": 3,
        "actual_process_restore": {
            "first_pid": a["pid"],
            "restored_pid": b["pid"],
            "open_positions": b["open_positions_at_start"],
            "checkpoint_sha256": b["checkpoint_sha256"],
        },
        "expanded_alpha_accounts": 0,
        "independent_financial_audit_complete": False,
        "profit_goal_met": False,
    }
    save(output / "verification.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    verify(p.parse_args().output)
