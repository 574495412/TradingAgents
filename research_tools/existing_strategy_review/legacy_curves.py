"""Retain legacy saved curves without assuming an undocumented capital or cost model."""

import argparse
import json
from pathlib import Path

import pandas as pd

from research_tools.existing_strategy_review.normalize import curve_stats, number, sha


def inspect(inventory_path, output):
    inventory = json.loads(inventory_path.read_text())
    rows = []
    seen = set()
    for alias in inventory["aliases"]:
        root = Path(alias["canonical"])
        for curve in sorted(root.rglob("*equity.csv")):
            if curve.resolve() in seen:
                continue
            seen.add(curve.resolve())
            protocol = None
            contract = None
            for parent in [curve.parent, *list(curve.parents)[:3]]:
                p = parent / "protocol.json"
                if p.is_file():
                    value = json.loads(p.read_text())
                    if number(value.get("initial_equity")) is not None:
                        protocol = value
                        contract = p
                        break
            row = {
                "curve_path": str(curve),
                "curve_sha256": sha(curve),
                "study": root.name,
                "strategy": curve.stem.removesuffix("_equity"),
                "capital": protocol.get("initial_equity") if protocol else None,
                "kind": "legacy_equity_contract_incomplete",
                "net_return": None,
                "funding_scope": "unknown or source-specific proxy; not relabeled as complete native funding",
                "head_to_head_eligible": False,
            }
            if protocol:
                frame = pd.read_csv(curve)
                stats = curve_stats(frame, float(protocol["initial_equity"]))
                if stats and "last_equity" in stats:
                    row.update(
                        kind="legacy_saved_equity",
                        saved_curve=stats,
                        net_return=stats["last_equity"] / float(protocol["initial_equity"]) - 1,
                        protocol_path=str(contract),
                        protocol_sha256=sha(contract),
                        fee_bps=protocol.get("fee_bps"),
                        slippage_bps=protocol.get("slippage_bps"),
                        adverse_carry_proxy_bps=protocol.get("adverse_carry_bps_per_8h"),
                    )
            else:
                row["gap"] = (
                    "No adjacent protocol with explicit initial equity; do not divide by first post-cost observation."
                )
            rows.append(row)
    output.write_text(
        json.dumps(
            {"schema": "legacy-strategy-curve-review.v1", "curves": rows, "orders_enabled": False},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        {
            "legacy_curves": len(rows),
            "explicit_capital_curves": sum(r["net_return"] is not None for r in rows),
            "missing_capital_contract": sum(r["capital"] is None for r in rows),
            "highest_saved_curve_return": max(
                (r["net_return"] for r in rows if r["net_return"] is not None), default=None
            ),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    inspect(a.inventory, a.output)
