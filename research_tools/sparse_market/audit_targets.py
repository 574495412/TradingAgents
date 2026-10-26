"""Direct NumPy and Decimal reconstruction of frozen family books and clocks."""

import argparse
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw


def audit(inputs, output):
    if output.exists():
        raise FileExistsError("Preserve independent target audit")
    verify_manifest(inputs)
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Prepared target source changed")
    cohort, bundle = Path(plan["cohort"]), Path(plan["bundle"])
    registry = read(cohort / "registry.json")
    raw = prepare_raw(plan, cohort)
    symbols = read(cohort / "symbols.json")
    rates = {s: pd.read_parquet(bundle / "funding" / (s + "_exact.parquet")) for s in symbols}
    anchors = read(inputs / "anchors.json")
    groups = {}
    for r in anchors:
        clock, selected = pd.Timestamp(r["decision"]), r["selected"]
        candidate = r["candidate"]
        values = {}
        for s in selected:
            f = raw[r["variant"]][s]
            valid = (
                f.price_liquidity_feature_usable.eq(True) & f.volume.gt(0) & f.quote_volume.gt(0)
            )
            valid.iloc[0] = False
            launches = [
                c
                for c in registry["launch_cards"]
                if c["symbol"] == s
                and pd.Timestamp(c["available_at_assumed"]) <= clock
                and pd.Timestamp(c["launch_at"]) <= clock
            ]
            start = (
                max(pd.Timestamp(c["launch_at"]).ceil("D") for c in launches)
                if launches
                else f.index[0]
            )
            close = (
                f.close.where(valid)
                .loc[(f.index >= start) & (f.index < clock)]
                .iloc[-29:]
                .to_numpy(dtype=float)
            )
            if len(close) != 29 or not np.isfinite(close).all():
                raise ValueError("Target lacks full prior identity price window")
            if candidate.startswith("funding7"):
                observations = rates[s].loc[
                    (rates[s].index >= clock - pd.Timedelta(days=7)) & (rates[s].index < clock),
                    "funding_rate",
                ]
                if not len(observations):
                    raise ValueError("Target has no actual past funding")
                values[s] = round(
                    float(sum((Decimal(str(v)) for v in observations), Decimal(0))), 12
                )
            elif candidate == "max28":
                values[s] = float(np.max(close[1:] / close[:-1] - 1))
            else:
                values[s] = float(close[-1] / close[0] - 1)
        ascending = candidate.startswith("funding7")
        ordered = sorted(selected, key=lambda s: (values[s] if ascending else -values[s], s))
        expected = dict.fromkeys(ordered[:4], 0.125) | dict.fromkeys(ordered[-4:], -0.125)
        if candidate.startswith("funding7"):
            if max(values.values()) == min(values.values()):
                expected = {}
            forecast = -math.fsum(w * values[s] for s, w in expected.items())
            if candidate == "funding7_cost_floor" and forecast <= 0.00375:
                expected = {}
        if r["book"] != expected:
            raise ValueError(
                "Independent family direction/rank differs: " + str((candidate, clock))
            )
        groups.setdefault((r["variant"], r["window"], candidate), {})[clock] = expected
    cells = 0
    for (variant, window_id, candidate), clocks in groups.items():
        window = next(w for w in plan["windows"] if w["id"] == window_id)
        actual = pd.read_parquet(inputs / f"{variant}_{window_id}_{candidate}_targets.parquet")
        expected_calendar = pd.date_range(
            window["start"], window["end"], freq="h", inclusive="left", tz="UTC"
        )
        if not actual.index.equals(expected_calendar) or not actual.rebalance.equals(
            pd.Series(actual.index.isin(clocks), index=actual.index)
        ):
            raise ValueError("Continuous hourly book/weekly rebalance clock differs")
        ordered = sorted(clocks)
        previous = None
        for clock in actual.index:
            if clock in clocks:
                previous = clocks[clock]
            row = actual.loc[clock]
            if row.risk_scale != 1 or any(
                float(row["target_" + s]) != previous.get(s, 0.0) for s in symbols
            ):
                raise ValueError("Held targets differ from most recent actual weekly book")
            cells += 1
    result = {
        "schema": "sparse-target-independent-audit.v1",
        "passed": True,
        "source_sha256": sha(Path(__file__)),
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "anchor_source_candidate_cases": len(anchors),
        "hourly_source_candidate_rows": cells,
        "direct_price_rank_and_decimal_funding_checked": True,
        "financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.output)
