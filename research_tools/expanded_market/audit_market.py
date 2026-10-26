"""Independently reconcile retained archives, sparse calendars and daily features."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(root):
    manifest = read(root / "completion.json")["manifest"]
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise ValueError("Sealed source changed: " + str(root / name))
    return len(manifest)


def require_close(actual, expected):
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True
    ):
        raise ValueError("Retained numerical source differs")


def audit(market, cohort, output):
    output.mkdir(parents=True, exist_ok=False)
    counts = {str(r): check_manifest(r) for r in [market, cohort]}
    plan = read(cohort / "plan.json")
    start, end = [pd.Timestamp(plan[k], tz="UTC") for k in ["bars_start", "end_exclusive"]]
    reports = read(market / "download_summary.json")["records"]
    if sorted(r["symbol"] for r in reports) != read(cohort / "symbols.json"):
        raise ValueError("Acquisition roster differs")
    registration = {
        "schema": "expanded-market-independent-audit.v1",
        "source_sha256": digest(Path(__file__)),
        "market_completion_sha256": digest(market / "completion.json"),
        "cohort_completion_sha256": digest(cohort / "completion.json"),
        "daily_completion_sha256": digest(Path(plan["daily_source"]) / "completion.json"),
        "numeric_rtol": 1e-12,
        "numeric_atol": 1e-10,
        "daily_comparison_tolerance": "prices1e-9 relative +1e-10; quote0.01 +1e-9 relative",
        "no_source_replacement": True,
        "new_financial_accounts": 0,
    }
    (output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    results, conflicts, gaps = [], [], []
    for i, report in enumerate(sorted(reports, key=lambda r: r["symbol"])):
        s = report["symbol"]
        f = pd.read_parquet(market / "symbols" / (s + "_5m.parquet"))
        if (
            f.index.has_duplicates
            or not f.index.is_monotonic_increasing
            or not f.index.floor("5min").equals(f.index)
            or f.index[0] < start
            or f.index[-1] >= end
        ):
            raise ValueError("Wrong sparse time grain: " + s)
        matched = 0
        if report["source_kind"] == "sealed_six_column_reference":
            source = Path(report["path"])
            if digest(source) != report["sha256"]:
                raise ValueError("Adopted reference changed")
            old = pd.read_parquet(source).loc[lambda a: (a.index >= start) & (a.index < end)]
            if not f.index.equals(old.index):
                raise ValueError("Adopted sparse index differs")
            require_close(f[old.columns].to_numpy(), old.to_numpy())
            matched = len(old)
        else:
            detail = read(market / "reports" / (s + ".json"))
            for item in detail["sources"]:
                path = Path(item["path"])
                checksum = path.with_name(path.name + ".CHECKSUM").read_text().split()
                if checksum[0] != digest(path) or checksum[1].lstrip("*") != path.name:
                    raise ValueError("Independent SHA verification differs")
                with zipfile.ZipFile(path) as z:
                    if z.testzip() is not None or len(z.namelist()) != 1:
                        raise ValueError("Independent ZIP integrity differs")
                    raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
                # Official modern archives have named columns. Retain legacy headerless rows.
                if "open_time" not in raw.columns:
                    with zipfile.ZipFile(path) as z:
                        raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None)
                raw = raw.apply(pd.to_numeric)
                epoch = raw.iloc[:, 0].to_numpy(dtype="int64")
                unit = "us" if epoch.min() > 100_000_000_000_000 else "ms"
                idx = pd.DatetimeIndex(pd.to_datetime(epoch, unit=unit, utc=True))
                keep = (idx >= start) & (idx < end)
                idx, raw = idx[keep], raw.loc[keep]
                actual = f.reindex(idx)
                cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trades",
                    "taker_base_volume",
                    "taker_quote_volume",
                ]
                require_close(
                    actual[cols].to_numpy(), raw.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10]].to_numpy()
                )
                close_time = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 6], unit=unit, utc=True))
                if not close_time.equals(pd.DatetimeIndex(actual.close_time)):
                    raise ValueError("Source candle close time changed")
                matched += len(raw)
        if matched != len(f):
            raise ValueError("Missing or extra transformed rows: " + s)
        p = f[["open", "high", "low", "close"]]
        price = np.isfinite(p).all(axis=1) & p.gt(0).all(axis=1)
        price &= f.high.ge(p.max(axis=1)) & f.low.le(p.min(axis=1))
        liq = price & np.isfinite(f[["volume", "quote_volume"]]).all(axis=1)
        liq &= f.volume.gt(0) & f.quote_volume.gt(0)
        tol = 1e-4 + 1e-6 * (f.high * f.volume).abs()
        quote_bad = f.quote_volume.lt(f.low * f.volume - tol) | f.quote_volume.gt(
            f.high * f.volume + tol
        )
        liq &= ~quote_bad
        if "close_time" in f:
            duration = f.close_time.ge(
                f.index + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
            )
            duration &= f.close_time.lt(f.index + pd.Timedelta(minutes=5))
            liq &= duration
        if not np.array_equal(price, f.price_usable) or not np.array_equal(
            liq, f.execution_liquidity_usable
        ):
            raise ValueError("Independent quality flags differ: " + s)
        expected = pd.date_range(f.index[0], f.index[-1], freq="5min")
        missing = expected.difference(f.index)
        for group in pd.Series(missing).groupby(pd.Series(missing).dt.floor("D")):
            day, times = group
            if len(times):
                gaps.append(
                    {
                        "symbol": s,
                        "day": day.isoformat(),
                        "bars": len(times),
                        "first": times.iloc[0].isoformat(),
                        "last": times.iloc[-1].isoformat(),
                    }
                )
        daily = pd.read_parquet(Path(plan["daily_source"]) / "symbols" / (s + "_1d.parquet"))
        agg = f.resample("D").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            rows=("close", "size"),
        )
        complete = agg.loc[agg.rows.eq(288)].index.intersection(daily.index)
        old, new = daily.loc[complete], agg.loc[complete]
        differences = {}
        for col in ["open", "high", "low", "close", "quote_volume"]:
            limit = (0.01 if col == "quote_volume" else 1e-10) + 1e-9 * old[col].abs()
            differences[col] = (old[col] - new[col]).abs().gt(limit)
        bad = pd.DataFrame(differences).any(axis=1)
        for at in complete[bad]:
            changed = {
                col: {"daily": float(old.at[at, col]), "five_minute": float(new.at[at, col])}
                for col, flags in differences.items()
                if flags.at[at]
            }
            conflicts.append(
                {
                    "symbol": s,
                    "date": at.isoformat(),
                    "fields": changed,
                    "daily_feature_usable": bool(old.at[at, "price_liquidity_feature_usable"]),
                }
            )
        results.append(
            {
                "symbol": s,
                "rows": len(f),
                "source_kind": report["source_kind"],
                "source_rows_reconciled": matched,
                "interior_missing_bars": len(missing),
                "unusable_price_bars": int((~price).sum()),
                "unusable_execution_bars": int((~liq).sum()),
                "nonpositive_volume_bars": int(f.volume.le(0).sum()),
                "quote_envelope_bad_bars": int(quote_bad.sum()),
                "complete_days_compared": len(complete),
                "conflicting_days": int(bad.sum()),
                "close_conflicting_days": int(differences["close"].sum()),
            }
        )
        if (i + 1) % 20 == 0:
            print({"audited_symbols": i + 1, "conflict_days": len(conflicts)}, flush=True)
    for name, value in [
        ("symbols.json", results),
        ("daily_conflicts.json", conflicts),
        ("gaps.json", gaps),
    ]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    summary = {
        "schema": "expanded-market-independent-reconciliation.v1",
        "source_integrity_passed": True,
        "manifest_files_checked": counts,
        "symbols": len(results),
        "rows": sum(r["rows"] for r in results),
        "raw_or_adopted_numeric_rows_reconciled": sum(r["source_rows_reconciled"] for r in results),
        "interior_missing_bars": sum(r["interior_missing_bars"] for r in results),
        "unusable_execution_bars": sum(r["unusable_execution_bars"] for r in results),
        "complete_days_compared": sum(r["complete_days_compared"] for r in results),
        "daily_conflicting_days": len(conflicts),
        "gap_days": len(gaps),
        "daily_and_five_minute_sources_concordant": not conflicts,
        "historical_tradability_certified": False,
        "financial_accounts": 0,
        "profit_goal_met": False,
        "no_prices_imputed": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "completion.json").write_text(json.dumps({"manifest": manifest}, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.market, a.cohort, a.output)
