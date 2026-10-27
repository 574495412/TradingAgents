"""Acquire only four necessary archives, bind raw case sources before financial replay."""

import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def parse(body, kind, case):
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        if len(z.namelist()) != 1:
            raise ValueError("Unexpected source ZIP members")
        f = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if str(f.iloc[0, 0]) == "open_time":
        f = f.iloc[1:].reset_index(drop=True)
    if f.shape[1] != 12:
        raise ValueError("Unexpected raw source schema")
    unit = "us" if kind == "spot" else "ms"
    opening = pd.DatetimeIndex(
        pd.to_datetime(f[0].astype("int64"), unit=unit, utc=True), name="date"
    ).as_unit("ns")
    closing = pd.DatetimeIndex(pd.to_datetime(f[6].astype("int64"), unit=unit, utc=True)).as_unit(
        "ns"
    )
    if opening.has_duplicates or not opening.is_monotonic_increasing:
        raise ValueError("Actual source clocks duplicated or out of order")
    frame = f[[1, 2, 3, 4, 5, 7]].astype(float)
    frame.columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame.index = opening
    start = pd.Timestamp(case["decision"]) - pd.Timedelta(minutes=5)
    end = pd.Timestamp(case["exit_deadline"])
    wanted = pd.date_range(start, end, freq="5min", inclusive="left")
    if len(wanted.difference(frame.index)):
        raise ValueError("Missing real source case bar; no synthetic repair")
    sel = opening.isin(wanted)
    gaps = opening[sel] + pd.Timedelta(minutes=5) - closing[sel]
    if not ((gaps > pd.Timedelta(0)) & (gaps <= pd.Timedelta(milliseconds=1))).all():
        raise ValueError("Case contains shortened or invalid source interval")
    frame = frame.loc[wanted]
    if (
        not np.isfinite(frame.to_numpy()).all()
        or not frame.iloc[:, :4].gt(0).all().all()
        or not frame.high.ge(frame[["open", "low", "close"]].max(axis=1)).all()
        or not frame.low.le(frame[["open", "high", "close"]].min(axis=1)).all()
        or (frame[["volume", "quote_volume"]] < 0).any().any()
    ):
        raise ValueError("Actual source price/volume envelope invalid")
    return frame


def prepare(plan_path, output):
    plan = read(plan_path)
    for key in ["parent_boundaries", "parent_horizon", "parent_bundle", "perpetual_raw"]:
        verify_manifest(Path(plan[key]))
    failure = Path("data/research/carry_exit_cases_20261026_v001")
    verify_manifest(failure)
    plan = {
        **plan,
        "source_recovery": {
            "failed_completion_sha256": sha(failure / "completion.json"),
            "repair": "Normalize timestamp storage unit only; original failure preserved; both source windows, costs and risk unchanged.",
            "prior_http_calls": 4,
        },
    }
    goal = read(Path(plan["formal_goal"]))
    if goal["version"] != "20261026_v003":
        raise ValueError("Depth-first formal goal missing")
    output.mkdir(parents=True, exist_ok=False)
    for name in ["raw", "symbols", "funding"]:
        (output / name).mkdir()
    paths = [
        plan_path,
        Path(__file__),
        Path(plan["engine"]),
        Path(plan["formal_goal"]),
        *[
            Path(plan[k]) / "completion.json"
            for k in ["parent_boundaries", "parent_horizon", "parent_bundle", "perpetual_raw"]
        ],
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_new_case_spot_and_mark_requests": True,
            "new_financial_case_paths": 0,
        },
    )
    save(output / "plan.json", plan)
    requests_count = byte_count = 0
    receipts = []
    for case in plan["cases"]:
        s, month = case["symbol"], case["month"]
        for kind in ["spot", "mark"]:
            url = plan[kind + "_archive"].format(symbol=s, month=month)
            folder = output / "raw" / kind
            folder.mkdir(exist_ok=True)
            values = []
            for suffix in [".CHECKSUM", ""]:
                r = requests.get(url + suffix, timeout=(10, 30))
                requests_count += 1
                byte_count += len(r.content)
                path = folder / (url.rsplit("/", 1)[1] + suffix)
                path.write_bytes(r.content)
                values.append(path)
                receipts.append(
                    {
                        "kind": kind,
                        "symbol": s,
                        "url": r.url,
                        "status": r.status_code,
                        "bytes": len(r.content),
                        "path": str(path.relative_to(output)),
                        "sha256": sha(path),
                        "captured_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                )
                r.raise_for_status()
                if (
                    requests_count > plan["budget"]["maximum_http_requests"]
                    or byte_count > plan["budget"]["maximum_source_bytes"]
                ):
                    raise ValueError("Registered source budget exceeded")
            expected = values[0].read_text().split()[0]
            if expected != sha(values[1]):
                raise ValueError("Official source checksum differs")
            frame = parse(values[1].read_bytes(), kind, case)
            frame.to_parquet(output / "symbols" / (s + "_" + kind + "_5m.parquet"))
        parent_report = read(Path(plan["perpetual_raw"]) / "reports" / (s + ".json"))
        chosen = next(r for r in parent_report["sources"] if r["month"] == month)
        parent = Path(chosen["path"])
        if (
            sha(parent) != chosen["sha256"]
            or sha(parent.with_suffix(".zip.CHECKSUM")) != chosen["checksum_sha256"]
        ):
            raise ValueError("Sealed perpetual source changed")
        raw = output / "raw" / "perpetual"
        raw.mkdir(exist_ok=True)
        for p in [parent, parent.with_suffix(".zip.CHECKSUM")]:
            os.link(p, raw / p.name)
        # Perpetual archive timestamps use milliseconds, just like the mark source.
        frame = parse(parent.read_bytes(), "perpetual", case)
        joined = pd.read_parquet(Path(plan["parent_bundle"]) / "symbols" / (s + "_5m.parquet")).loc[
            frame.index, frame.columns
        ]
        joined.index = joined.index.as_unit("ns")
        pd.testing.assert_frame_equal(frame, joined, check_exact=True, check_freq=False)
        frame.to_parquet(output / "symbols" / (s + "_perpetual_5m.parquet"))
        funding = pd.read_parquet(Path(plan["parent_bundle"]) / "funding" / (s + "_exact.parquet"))
        start = pd.Timestamp(case["decision"]) - pd.Timedelta(days=7)
        end = pd.Timestamp(case["exit_deadline"])
        funding = funding.loc[(funding.index >= start) & (funding.index <= end)]
        if not len(funding) or funding.index.has_duplicates or not funding.mark.gt(0).all():
            raise ValueError("Unknown/duplicate actual funding marks in selected window")
        funding.to_parquet(output / "funding" / (s + "_exact.parquet"))
        for p in sorted(Path(plan["funding_raw"]).glob(s + "_page*.json")):
            os.link(p, output / "funding" / p.name)
    save(output / "source_receipts.json", receipts)
    save(
        output / "summary.json",
        {
            "schema": "carry-case-inputs.v1",
            "cases": plan["cases"],
            "new_http_requests": requests_count,
            "new_source_bytes": byte_count,
            "new_spot_archives": 2,
            "new_mark_archives": 2,
            "financial_case_paths": 0,
            "financial_strategy_accounts": 0,
            "independent_oos": False,
            "profit_goal_met": False,
            "portfolio_actual": None,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "financial_case_paths": 0,
        },
    )
    print(read(output / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.plan, args.output)
