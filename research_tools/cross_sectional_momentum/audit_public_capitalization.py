"""Independently validate the small public provider sample, not a strategy."""

import argparse
from decimal import Decimal, localcontext
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def validate_rows(data):
    rows = data.get("data", [])
    if len(rows) != 2 or {r.get("asset") for r in rows} != {"btc", "eth"}:
        raise ValueError("Missing/duplicate/unexpected sample identity")
    checked = []
    with localcontext() as context:
        context.prec = 100
        for row in rows:
            if row.get("time") != "2025-01-07T00:00:00.000000000Z":
                raise ValueError("Unexpected historical observation time")
            values = {}
            for key in ["PriceUSD", "SplyCur", "CapMrktCurUSD", "CapMrktEstUSD"]:
                try:
                    value = Decimal(str(row.get(key)))
                except Exception as error:
                    raise ValueError("Missing source numeric input") from error
                if not value.is_finite() or value <= 0:
                    raise ValueError("Null/nonfinite/nonpositive source input")
                values[key] = value
            difference = values["CapMrktCurUSD"] - values["PriceUSD"] * values["SplyCur"]
            if difference != 0:
                raise ValueError("Provider current-cap definition does not reconcile exactly")
            checked.append(
                {
                    **row,
                    "current_supply_product_error": str(difference),
                    "current_minus_estimated_usd": str(
                        values["CapMrktCurUSD"] - values["CapMrktEstUSD"]
                    ),
                    "estimated_to_current_ratio": str(
                        values["CapMrktEstUSD"] / values["CapMrktCurUSD"]
                    ),
                    "two_cap_definitions_equal": values["CapMrktCurUSD"] == values["CapMrktEstUSD"],
                    "historical_publication_time_certified": False,
                    "full_universe_or_coin_identity_history_certified": False,
                    "independent_circulating_supply_verification": False,
                }
            )
    return checked


def audit(source, current_source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    verify_manifest(current_source)
    receipt = read(source / "response.receipt.json")
    assert receipt["status"] == 200
    data = read(source / "response.json")
    import json

    assert data == json.loads((source / "response.body").read_bytes())
    checked = validate_rows(data)
    old_rows = {r["asset"]: r for r in read(current_source / "historical.json")["data"]}
    consistency = all(
        all(
            row[k] == old_rows[row["asset"]][k]
            for k in ["PriceUSD", "SplyCur", "CapMrktCurUSD", "time"]
        )
        for row in checked
    )
    save(output / "checked_rows.json", checked)
    summary = {
        "public_no_key_sample_available": True,
        "valid_historical_asset_observations": len(checked),
        "current_supply_cap_identity_exact": True,
        "previous_current_cap_probe_fields_unchanged": consistency,
        "current_and_estimated_cap_differ_in_both_rows": all(
            not r["two_cap_definitions_equal"] for r in checked
        ),
        "original_author_CoinGecko_raw_panel_obtained": False,
        "full_formation_universe_ready": False,
        "historical_availability_certified": False,
        "new_financial_accounts": 0,
        "new_predictive_models": 0,
        "external_review_performed": False,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "verification.json",
        {
            "source_completion_sha256": sha(source / "completion.json"),
            "current_sample_completion_sha256": sha(current_source / "completion.json"),
            "audit_source_sha256": sha(Path(__file__)),
            "decimal_precision": 100,
            "mathematical_consistency_is_not_independent_economic_truth": True,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--current-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.current_source, a.output)
