"""Preserve core funding snapshot and add only the inclusive valuation boundary."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def complete(output):
    parent = Path("data/research/max_effect_actual_funding_20261026_v001")
    terminal = Path("data/research/max_effect_funding_terminal_20261026_v001")
    plan = read(Path(__file__).with_name("plan.json"))
    source = read(parent / "completion.json")
    for name, digest in source["manifest"].items():
        if sha(parent / name) != digest:
            raise ValueError("Core funding bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "raw", output / "raw")
    shutil.copytree(terminal, output / "terminal")
    shutil.copyfile(parent / "collection.json", output / "core_collection.json")
    symbols = read(parent / "collection.json")["symbols"]
    boundary = pd.Timestamp(plan["funding_end_exclusive"])
    counts = []
    for s in symbols:
        extra = read(terminal / (s + ".json"))
        rows = extra["response"]["result"]
        request = extra["request"]
        if (
            request["startTime"] != int(boundary.timestamp() * 1000)
            or request["endTime"] != request["startTime"]
            or any(r["fundingTime"] != request["startTime"] or r["symbol"] != s for r in rows)
        ):
            raise ValueError("Supplement is outside exact terminal boundary")
        frame = pd.read_parquet(parent / (s + "_exact.parquet"))
        if rows:
            patch = pd.DataFrame(
                {
                    "funding_rate": [float(r["fundingRate"]) for r in rows],
                    "mark": [float(r["markPrice"]) for r in rows],
                },
                index=pd.DatetimeIndex(
                    pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True),
                    name="date",
                ),
            )
            frame = pd.concat([frame, patch])
        if frame.index.has_duplicates or frame.index[-1] > boundary or (frame.mark <= 0).any():
            raise ValueError("Invalid supplemented funding calendar")
        frame.to_parquet(output / (s + "_exact.parquet"))
        records = [
            r
            for f in sorted((output / "raw").glob(s + "_*.json"))
            for r in read(f)["response"]["result"]
        ] + rows
        if (
            frame.index.as_unit("ms").asi8.tolist() != [r["fundingTime"] for r in records]
            or frame.mark.tolist() != [float(r["markPrice"]) for r in records]
            or frame.funding_rate.tolist() != [float(r["fundingRate"]) for r in records]
        ):
            raise ValueError("Raw exact funding reconstruction differs")
        counts.append(
            {
                "symbol": s,
                "events": len(frame),
                "terminal_events": len(rows),
                "independent_raw_values_match": True,
            }
        )
    summary = {
        "schema": "max-effect-actual-funding.v2",
        "passed": True,
        "symbols": len(symbols),
        "events": sum(r["events"] for r in counts),
        "core_events": read(parent / "summary.json")["events"],
        "requests": read(parent / "collection.json")["requests"] + len(symbols),
        "reports": counts,
        "parent_completion_sha256": sha(parent / "completion.json"),
        "terminal_settlement_inclusive": boundary.isoformat(),
        "proxy_marks_used": False,
        "source_sha256": sha(Path(__file__)),
    }
    if summary["requests"] > plan["funding_request_budget"]:
        raise ValueError("Funding source budget exceeded")
    save(output / "summary.json", summary)
    save(output / "plan.json", plan)
    files = [p for p in output.rglob("*") if p.is_file()]
    save(
        output / "completion.json",
        {
            "manifest": {str(p.relative_to(output)): sha(p) for p in files},
            "audit_passed": True,
            "events": summary["events"],
        },
    )
    print(
        {
            "complete": True,
            "events": summary["events"],
            "requests": summary["requests"],
            "terminal_additions": sum(r["terminal_events"] for r in counts),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    complete(a.output)
