"""Retain six real shortened daily candles without treating them as full days."""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

EXPECTED = {
    ("BAKEUSDT", "2025-09"),
    ("BNXUSDT", "2025-03"),
    ("FTMUSDT", "2025-01"),
    ("HIFIUSDT", "2025-09"),
    ("OMUSDT", "2026-03"),
    ("TONUSDT", "2026-06"),
}


def recover(parent, output):
    verify_manifest(parent)
    summary = read(parent / "summary.json")
    failed = [r for r in summary["records"] if r["state"] == "unknown_source_error"]
    if {(r["symbol"], r["month"]) for r in failed} != EXPECTED:
        raise ValueError("Partial-day amendment does not match the six retained failures")
    output.mkdir(parents=True, exist_ok=False)
    skip_records = {r["symbol"] + "_" + r["month"] + ".json" for r in failed}
    for p in parent.rglob("*"):
        relative = p.relative_to(parent)
        if not p.is_file() or relative.parts[0] == "symbols":
            continue
        if str(relative) in ["summary.json", "completion.json", "registration.json"]:
            continue
        if relative.parts[0] == "records" and p.name in skip_records:
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, target)
    registration = read(parent / "registration.json")
    registration["source_hashes"][str(Path(__file__))] = sha(Path(__file__))
    registration["parent_source_completion_sha256"] = sha(parent / "completion.json")
    registration["amendment"] = (
        "Only preserve6actual shortened source candles as feature-ineligible; no OHLC, selection, cost or risk change, no network repair and no financial returns. Prior full rows from the same months become observable again."
    )
    save(output / "registration.json", registration)
    replacements = {}
    for row in failed:
        s, month = row["symbol"], row["month"]
        path = output / "archives" / f"{s}-1d-{month}.zip"
        checksum = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
        if sha(path) != checksum:
            raise ValueError("Retained partial source checksum differs")
        with zipfile.ZipFile(path) as z:
            f = pd.read_csv(z.open(z.namelist()[0]), header=None)
        unit = "us" if month >= "2025-01" else "ms"
        opening = pd.DatetimeIndex(pd.to_datetime(f[0], unit=unit, utc=True), name="date").as_unit(
            "ns"
        )
        closing = pd.DatetimeIndex(pd.to_datetime(f[6], unit=unit, utc=True)).as_unit("ns")
        last = opening + pd.Timedelta(days=1) - pd.Timedelta(1, unit=unit)
        full = closing == last
        if sum(~full) != 1 or ((closing <= opening) | (closing > last)).any():
            raise ValueError("Source is not a single real shortened daily interval")
        start = pd.Timestamp(month + "-01", tz="UTC")
        calendar = pd.date_range(start, start + pd.offsets.MonthBegin(), freq="D", inclusive="left")
        if (
            opening.has_duplicates
            or not opening.is_monotonic_increasing
            or len(opening.difference(calendar))
        ):
            raise ValueError("Partial source daily opening calendar differs")
        out = f[[1, 2, 3, 4, 5, 7]].astype(float)
        out.columns = ["open", "high", "low", "close", "volume", "quote_volume"]
        out.index = opening
        tolerance = 0.01 + 1e-6 * (out.high * out.volume).abs()
        out["feature_usable"] = (
            full
            & np.isfinite(out.to_numpy()).all(axis=1)
            & out.iloc[:, :4].gt(0).all(axis=1)
            & out.volume.gt(0)
            & out.quote_volume.gt(0)
            & out.high.ge(out[["open", "low", "close"]].max(axis=1))
            & out.low.le(out[["open", "high", "close"]].min(axis=1))
            & out.quote_volume.ge(out.low * out.volume - tolerance)
            & out.quote_volume.le(out.high * out.volume + tolerance)
        )
        out.to_parquet(output / "months" / (s + "_" + month + ".parquet"))
        i = int(np.flatnonzero(~full)[0])
        record = {
            "symbol": s,
            "month": month,
            "state": "verified_source",
            "rows": len(out),
            "unit": unit,
            "official_sha256": checksum,
            "url": "https://data.binance.vision/data/spot/monthly/klines/" + s + "/1d/" + path.name,
            "invalid_or_nonpositive_volume_rows": int((~out.feature_usable).sum()),
            "partial_daily_labels": [
                {
                    "open": opening[i].isoformat(),
                    "actual_close": closing[i].isoformat(),
                    "feature_usable": False,
                }
            ],
            "absent_daily_labels": [t.isoformat() for t in calendar.difference(opening)],
            "source": "same verified retained officialZIP; no network or OHLC change",
        }
        replacements[(s, month)] = record
        save(output / "records" / (s + "_" + month + ".json"), record)
    records = [replacements.get((r["symbol"], r["month"]), r) for r in summary["records"]]
    (output / "symbols").mkdir()
    for s in sorted({r["symbol"] for r in records}):
        paths = sorted((output / "months").glob(s + "_*.parquet"))
        if paths:
            f = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
            if f.index.has_duplicates:
                raise ValueError("Merged real observations duplicated")
            f.to_parquet(output / "symbols" / (s + "_1d.parquet"))
    summary.update(
        records=records,
        verified_months=sum(r["state"] == "verified_source" for r in records),
        unknown_months=sum(r["state"] != "verified_source" for r in records),
        unknown_source_errors=sum(r["state"] == "unknown_source_error" for r in records),
        source_rows=sum(r.get("rows", 0) for r in records),
        recovered_partial_days=6,
        new_http_requests=0,
        raw_source_changed=False,
    )
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k != "records"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.parent, args.output)
