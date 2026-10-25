"""Independent raw-observation signal, funding, timing and universe reconstruction."""

import argparse
import math
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(root, output):
    for name, digest in read(root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Input receipt changed")
    plan = read(root / "plan.json")
    original = read(Path(plan["formation_plan"]))
    formation = Path(plan["formation"])
    data = Path(plan["data"])
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Input source changed after registration")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Data changed")
    symbols = read(root / "summary.json")["symbol_universe"]
    raw = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(data / (s + "_funding.parquet")) for s in symbols}
    clocks = pd.date_range("2025-01-01", "2026-09-01", freq="4h", inclusive="left", tz="UTC")
    checked = 0
    max_error = 0.0
    for candidate in original["candidates"]:
        rows = read(root / (candidate + ".json"))
        if [r["time"] for r in rows] != [t.isoformat() for t in clocks]:
            raise ValueError("Decision calendar incomplete")
        for t, row in zip(clocks, rows, strict=True):
            name = (
                "fixed_formation.json" if candidate == "pairs_fixed" else f"rolling_{t:%Y%m}01.json"
            )
            fit = read(formation / name)
            models = (
                {m["x"] + "/" + m["y"]: m for m in fit["selected"]} if candidate != "cash" else {}
            )
            if [p["pair_key"] for p in row["pairs"]] != list(models):
                raise ValueError("Formation selection changed")
            for signal in row["pairs"]:
                m = models[signal["pair_key"]]
                if (
                    signal["model_id"] != sha(formation / name)
                    or pd.Timestamp(signal["fit_at"]) > t
                ):
                    raise ValueError("Unfrozen or future model")
                prices = {}
                for s in (m["x"], m["y"]):
                    sample = raw[s].loc[t - pd.Timedelta(hours=4) : t - pd.Timedelta(minutes=5)]
                    if len(sample) != 48:
                        raise ValueError("Incomplete signal price window")
                    prices[s] = float(sample.close.iloc[-1])
                residual = (
                    math.log(prices[m["y"]]) - m["beta"] * math.log(prices[m["x"]]) - m["alpha"]
                )
                z = (residual - m["mean"]) / m["sigma"]
                error = abs(z - signal["z"])
                max_error = max(max_error, error)
                if (
                    error > 1e-10
                    or prices != signal["reference_prices"]
                    or not signal["signal_valid"]
                ):
                    raise ValueError("Raw price/z reconstruction differs")
                means = {}
                for s in prices:
                    per_day = {}
                    for day in pd.date_range(
                        t.floor("D") - pd.Timedelta(days=5), periods=5, freq="D"
                    ):
                        sample = rates[s].loc[
                            (rates[s].index >= day)
                            & (rates[s].index < day + pd.Timedelta(days=1))
                            & (rates[s].index + pd.Timedelta(minutes=5) <= t)
                        ]
                        if len(sample):
                            per_day[day] = math.fsum(sample.funding_rate.to_list())
                    means[s] = math.fsum(per_day.values()) / 5 if len(per_day) == 5 else None
                    actual = signal["funding_daily_means"][s]
                    if (means[s] is None) != (actual is None) or (
                        actual is not None and abs(actual - means[s]) > 1e-14
                    ):
                        raise ValueError("Funding daily sums differ")
                known = all(v is not None for v in means.values())
                direction = 1 if z > 0 else -1 if z < 0 else 0
                projected = (
                    max(
                        0,
                        7
                        * (direction * m["beta"] * means[m["x"]] - direction * means[m["y"]])
                        / (1 + m["beta"]),
                    )
                    if known
                    else None
                )
                roundtrip = (
                    2
                    * (
                        original["signal"]["stress_fee_bps"]
                        + original["signal"]["stress_slippage_bps"]
                    )
                    / 10000
                )
                hurdle = max(2 * roundtrip, roundtrip + projected) if known else None
                edge = abs(residual - m["mean"]) / (1 + m["beta"])
                eligible = (
                    known
                    and original["signal"]["z_entry"] <= abs(z) < original["signal"]["z_stop"]
                    and edge > hurdle
                )
                if eligible != signal["entry_eligible"] or (
                    hurdle is not None and abs(hurdle - signal["hurdle_unit_gross"]) > 1e-14
                ):
                    raise ValueError("Entry/cost hurdle differs")
                if pd.Timestamp(signal["maximum_price_observation"]) >= t or (
                    signal["maximum_funding_observation"] is not None
                    and pd.Timestamp(signal["maximum_funding_observation"])
                    + pd.Timedelta(minutes=5)
                    > t
                ):
                    raise ValueError("Signal used unavailable observation")
                checked += 1
    save(
        output,
        {
            "passed": True,
            "schema": "independent-pair-input-audit.v1",
            "input_completion_sha256": sha(root / "completion.json"),
            "pair_observations": checked,
            "maximum_z_error": max_error,
            "auditor_sha256": sha(Path(__file__)),
            "account_returns_computed": False,
            "orders_enabled": False,
        },
    )
    print({"passed": True, "pair_observations": checked, "maximum_z_error": max_error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
