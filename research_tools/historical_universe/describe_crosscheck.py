"""Seal contradictory source evidence without turning disagreement into a passing gate."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.historical_universe.crosscheck_sources import parse

FIELDS = {
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "quote_volume": 7,
    "trades": 8,
    "taker_base_volume": 9,
    "taker_quote_volume": 10,
}


def describe(raw, api, output):
    output.mkdir(parents=True, exist_ok=False)
    p = read(raw / "plan.json")
    plan_path = Path(__file__).with_name("discrepancy_plan.json")
    plan = read(plan_path)
    cases = p["contract_days"]
    save(
        output / "plan.json",
        {
            "raw": str(raw),
            "api": str(api),
            "raw_plan_sha256": sha(raw / "plan.json"),
            "scope_sha256": sha(plan_path),
            "source_sha256": sha(Path(__file__)),
            "parser_sha256": sha(Path(__file__).with_name("crosscheck_sources.py")),
            "before_comparison": True,
            "contradictions_are_not_passes": True,
        },
    )
    oldroot, daily = Path(plan["old_5m"]), Path(plan["parent_daily"])
    bars = {
        s: pd.read_parquet(oldroot / (s + "_5m.parquet"))
        for s in {r["symbol"] for r in cases}
        if (oldroot / (s + "_5m.parquet")).is_file()
    }
    dailybars = {
        s: pd.read_parquet(daily / "symbols" / (s + "_1d.parquet"))
        for s in {r["symbol"] for r in cases}
    }
    results = []
    bindings = {}
    frames = {}
    for case in cases:
        s, d = case["symbol"], case["date"]
        path = raw / "raw" / s / f"{s}-5m-{d}.zip"
        check = path.with_name(path.name + ".CHECKSUM")
        parts = check.read_text().split()
        if (
            len(parts) != 2
            or parts[0].lower() != hashlib.sha256(path.read_bytes()).hexdigest()
            or parts[1].lstrip("*") != path.name
        ):
            raise ValueError("Checksum disagreement")
        bindings[str(path)] = sha(path)
        bindings[str(check)] = sha(check)
        f = parse(path.read_bytes(), s, d)
        frames[(s, d)] = f
        aggregate = {
            k: float(f[i].sum())
            for k, i in FIELDS.items()
            if k not in ["open", "high", "low", "close"]
        }
        aggregate.update(
            open=float(f.iloc[0, 1]),
            high=float(f[2].max()),
            low=float(f[3].min()),
            close=float(f.iloc[-1, 4]),
        )
        day = dailybars[s].loc[d]
        old_errors = None
        if s in bars:
            o = bars[s].loc[d]
            if not o.index.equals(f.index):
                raise ValueError("Old calendar differs")
            old_errors = {
                k: float(np.abs(o[k].to_numpy() - f[i].to_numpy()).max())
                for k, i in FIELDS.items()
                if k in o
            }
        differences = {k: aggregate[k] - float(day[k]) for k in aggregate}
        results.append(
            {
                "symbol": s,
                "date": d,
                "rows": len(f),
                "zip_sha256": sha(path),
                "daily5m_aggregate": aggregate,
                "old5m_max_absolute_errors": old_errors,
                "source1d_difference": differences,
                "old_price_ohlc_identical": old_errors is not None
                and all(old_errors[k] < 1e-10 for k in ["open", "high", "low", "close"]),
                "old_quote_identical": old_errors is not None and old_errors["quote_volume"] < 1e-7,
            }
        )
    api_checks = []
    for case in plan["api_reference_days"]:
        s, d = case["symbol"], case["date"]
        path = api / "api" / (s + "_" + d + ".json")
        source = read(path)
        a = np.array(source["response"]["structuredContent"]["result"], dtype=float)
        f = frames[(s, d)].to_numpy(dtype=float)
        if a.shape != (288, 12) or a.shape != f.shape or not np.array_equal(a[:, 0], f[:, 0]):
            raise ValueError("API shape/time invalid")
        bindings[str(path)] = sha(path)
        delta = np.abs(a - f)
        rows = np.any(delta > np.maximum(1e-8, np.abs(f) * 1e-12), axis=1)
        api_checks.append(
            {
                "symbol": s,
                "date": d,
                "all_numeric_fields_agree": not bool(rows.any()),
                "different5m_bars": int(rows.sum()),
                "different_bar_times": [
                    pd.Timestamp(int(t), unit="ms", tz="UTC").isoformat() for t in a[rows, 0]
                ],
                "max_errors_by_field": {k: float(delta[:, i].max()) for k, i in FIELDS.items()},
                "api_daily_quote_volume": float(a[:, 7].sum()),
                "archive_daily_quote_volume": float(f[:, 7].sum()),
                "zip_vs_api_same_daily_close": bool(a[-1, 4] == f[-1, 4]),
            }
        )
    summary = {
        "schema": "source-conflict-description.v1",
        "evidence_inspection_complete": True,
        "all_sources_concordant": all(r["all_numeric_fields_agree"] for r in api_checks),
        "contract_days": len(results),
        "cases_with_old40_source": sum(r["old5m_max_absolute_errors"] is not None for r in results),
        "old40_identical_daily5m_price_cases": sum(r["old_price_ohlc_identical"] for r in results),
        "old40_identical_daily5m_quote_cases": sum(r["old_quote_identical"] for r in results),
        "api_reference_days": len(api_checks),
        "api_reference_disagreement_days": sum(
            not r["all_numeric_fields_agree"] for r in api_checks
        ),
        "api_checks": api_checks,
        "cases": results,
        "automatic_source_replacement": False,
        "source_selection_by_returns": False,
        "original_datasets_modified": False,
        "financial_accounts": 0,
        "profit_goal_met": False,
        "next_action": "Source versions remain distinct. Treat conflicting prices/volumes as uncertainty and require candidate sensitivity rather than selecting the source with the highest return.",
    }
    save(output / "source_bindings.json", bindings)
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {q.name: sha(q) for q in output.iterdir() if q.is_file()},
            "comparison_complete": True,
            "sources_agree": summary["all_sources_concordant"],
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k not in ["cases", "api_checks"]}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--api", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    describe(a.raw, a.api, a.output)
