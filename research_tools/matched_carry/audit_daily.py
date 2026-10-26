"""Independently read raw CSV/Decimal values and assess data-only slot feasibility."""

import argparse
import csv
import io
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from factors.crypto.matched_spot_carry import known_funding7, prior_spot_window
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def raw_check(source, record):
    symbol, month = record["symbol"], record["month"]
    filename = record["url"].rsplit("/", 1)[1]
    path = source / "archives" / filename
    expected_hash = path.with_suffix(".zip.CHECKSUM").read_text().split()[0]
    if sha(path) != expected_hash or expected_hash != record["official_sha256"]:
        raise ValueError("Independent raw checksum differs")
    with zipfile.ZipFile(path) as z, io.TextIOWrapper(z.open(z.namelist()[0])) as f:
        rows = list(csv.reader(f))
    actual = pd.read_parquet(source / "months" / (symbol + "_" + month + ".parquet"))
    if len(actual) != len(rows):
        raise ValueError("Raw monthly row count differs")
    multiplier = 1000 if month >= "2025-01" else 1_000_000
    clocks, values, usable = [], [], []
    for row in rows:
        opening = int(row[0]) * multiplier
        if int(row[6]) * multiplier != opening + 86_400_000_000_000 - multiplier:
            raise ValueError("Raw daily interval or timestamp unit differs")
        o, h, low, c, volume, quote = [Decimal(row[k]) for k in [1, 2, 3, 4, 5, 7]]
        tol = Decimal("0.01") + Decimal("0.000001") * abs(h * volume)
        good = (
            all(v.is_finite() for v in [o, h, low, c, volume, quote])
            and min(o, h, low, c, volume, quote) > 0
            and h >= max(o, low, c)
            and low <= min(o, h, c)
            and low * volume - tol <= quote <= h * volume + tol
        )
        clocks.append(opening)
        values.append([float(v) for v in [o, h, low, c, volume, quote]])
        usable.append(good)
    if not np.array_equal(np.array(clocks), actual.index.as_unit("ns").asi8):
        raise ValueError("Independent raw daily clock differs")
    numbers = actual[["open", "high", "low", "close", "volume", "quote_volume"]].to_numpy()
    if not np.allclose(numbers, np.array(values), rtol=1e-14, atol=1e-12, equal_nan=True):
        raise ValueError("Independent raw numeric value differs")
    if usable != actual.feature_usable.tolist():
        raise ValueError("Independent Decimal daily qualification differs")
    return len(rows)


def audit(registration, source, output):
    verify_manifest(registration)
    verify_manifest(source)
    parent = read(source / "registration.json")
    if parent["parent_completion_sha256"] != sha(registration / "completion.json"):
        raise ValueError("Data source bound to a different frozen registration")
    for p, h in parent["source_hashes"].items():
        if sha(p) != h:
            raise ValueError("Frozen acquisition source changed")
    plan, summary = read(registration / "plan.json"), read(source / "summary.json")
    expected = read(registration / "daily_requests.json")
    records = summary["records"]
    if sorted((r["symbol"], r["month"]) for r in records) != sorted(
        (r["symbol"], r["month"]) for r in expected
    ):
        raise ValueError("Missing or extra registered source request")
    output.mkdir(parents=True, exist_ok=False)
    raw_rows = 0
    for r in records:
        if r["state"] == "verified_source":
            raw_rows += raw_check(source, r)
        elif r["state"] == "unknown_archive_absent_from_current_catalog":
            cat = read(source / "catalogs" / (r["symbol"] + ".json"))
            if (
                cat["error"]
                or sha(source / "catalogs" / (r["symbol"] + ".xml")) != cat["receipt"]["sha256"]
            ):
                raise ValueError("Unverified catalog absence")
            key = r["url"].split("data.binance.vision/", 1)[1]
            if key in cat["keys"] and key + ".CHECKSUM" in cat["keys"]:
                raise ValueError("Missing archive observation contradicts retained catalog")
    symbols = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (source / "symbols").glob("*_1d.parquet")
    }
    # Check merged data is exactly all retained months, never an interpolated calendar.
    for s, f in symbols.items():
        merged = pd.concat(
            [pd.read_parquet(p) for p in sorted((source / "months").glob(s + "_*.parquet"))]
        ).sort_index()
        pd.testing.assert_frame_equal(f, merged, check_exact=True)
    rates = {}
    rows, perturbations = [], []
    excluded = set(read(registration / "registration.json")["excluded_unit_mismatch_symbols"])
    for anchor in read(registration / "slots.json"):
        clock = pd.Timestamp(anchor["decision"])
        for s in anchor["slots"]:
            row = {
                **{k: anchor[k] for k in ["variant", "window", "decision"]},
                "symbol": s,
                "mapping_status": "unresolved_no_financial_execution",
                "actual_target": None,
            }
            if s in excluded:
                row.update(data_status="excluded_multiplier_unit", data_eligible=False)
            elif s not in symbols:
                row.update(data_status="no_observed_spot_daily_source", data_eligible=False)
            else:
                feature = prior_spot_window(symbols[s], clock)
                # Independently use Python index membership and positive flags for28 days.
                dates = [clock - pd.Timedelta(days=d) for d in range(28, 0, -1)]
                f = symbols[s]
                direct_valid = all(
                    t in f.index and t != f.index[0] and bool(f.loc[t, "feature_usable"])
                    for t in dates
                )
                direct_quote = (
                    sum(float(f.loc[t, "quote_volume"]) for t in dates) / 28
                    if direct_valid
                    else None
                )
                if feature["data_eligible"] != bool(direct_valid and direct_quote >= 10_000_000):
                    raise ValueError("Independent28day spot eligibility differs")
                row.update(data_status="observed_source", **feature)
                if s not in rates:
                    rates[s] = pd.read_parquet(
                        Path(plan["parent_bundle"]) / "funding" / (s + "_exact.parquet")
                    )
                row.update(known_funding7(rates[s], clock))
                if len(perturbations) < 6 and feature["data_eligible"]:
                    changed = f.copy()
                    changed.loc[changed.index >= clock, "quote_volume"] *= 1000
                    changed.loc[changed.index >= clock, "feature_usable"] = False
                    altered_rates = rates[s].copy()
                    altered_rates.loc[altered_rates.index >= clock, "funding_rate"] += 100
                    if prior_spot_window(changed, clock) != feature or known_funding7(
                        altered_rates, clock
                    ) != known_funding7(rates[s], clock):
                        raise ValueError("Future source values affect past spot/funding features")
                    perturbations.append(
                        {"symbol": s, "decision": clock.isoformat(), "passed": True}
                    )
            row["raw_data_only_candidate"] = bool(
                row["data_eligible"] and row.get("positive") is True
            )
            row["guarded_data_only_candidate"] = bool(
                row["data_eligible"] and row.get("cost_guard_passed") is True
            )
            rows.append(row)
    save(output / "slot_data_feasibility.json", rows)
    result = {
        "schema": "matched-carry-daily-independent-audit.v1",
        "source_arithmetic_passed": True,
        "raw_verified_months": summary["verified_months"],
        "raw_verified_rows": raw_rows,
        "unknown_months": summary["unknown_months"],
        "unknown_source_errors": summary["unknown_source_errors"],
        "slot_rows": len(rows),
        "data_eligible_slots": sum(r["data_eligible"] for r in rows),
        "positive_funding_data_only_slots": sum(r["raw_data_only_candidate"] for r in rows),
        "guarded_data_only_slots": sum(r["guarded_data_only_candidate"] for r in rows),
        "future_perturbations": perturbations,
        "mapping_certified": False,
        "financial_accounts": 0,
        "net_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe": None,
        "capacity_usdt": None,
        "profit_goal_met": False,
        "portfolio_actual": None,
        "next_action": "Resolve same-asset spot/perpetual episodes, then acquire actual5m spot and mark data. Data-only flags cannot enter an account until mapping and held-source gates pass.",
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("factors/crypto/matched_spot_carry.py"),
                registration / "completion.json",
                source / "completion.json",
            ]
        },
    }
    save(output / "verification.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.registration, args.source, args.output)
