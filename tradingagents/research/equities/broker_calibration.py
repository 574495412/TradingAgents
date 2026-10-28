"""Project a published broker commission on frozen fills, not a new account.

The result does not certify historical fee validity, order grouping, borrow
availability, executable fills or customer cash. No NAV is recalculated.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/equities_french_calibration_20261027"
SOURCE = ROOT / "data/research/equity_french_calibration_20261027_v001/broker"
PARENT = ROOT / "results/research/equities_pit_stocks_context_20261027_v001"
D = Decimal


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commission(quantity, price):
    return min(max(D(quantity) * D(".005"), D(1)), D(quantity) * D(str(price)) * D(".01"))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for path in (
        DOC / "execution-reference-plan.json",
        SOURCE / "commission.htm",
        SOURCE / "commission.receipt.json",
        PARENT / "completion.json",
        PARENT / "summary.json",
    ):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    assert read(SOURCE / "commission.receipt.json")["sha256"] == sha(SOURCE / "commission.htm")
    soup = BeautifulSoup((SOURCE / "commission.htm").read_bytes(), "html.parser")
    candidates = [
        t for t in soup.find_all("table") if "IBKR Pro - Fixed" in t.get_text(" ", strip=True)
    ]
    assert len(candidates) == 1
    text = candidates[0].get_text(" ", strip=True)
    for value in ("USD 0.005", "USD 1.00", "1% of Trade Value"):
        assert value in text, value
    assert "calculated maximum per order is less than the minimum per order" in soup.get_text(
        " ", strip=True
    )
    examples = []
    for quantity, price, expected in [
        (10, ".20", ".02"),
        (100, "100", "1"),
        (1000, "10", "5"),
        (1000, "100", "5"),
    ]:
        actual = commission(quantity, price)
        assert actual == D(expected)
        examples.append({"shares": quantity, "price_usd": price, "commission_usd": str(actual)})
    for path, digest in read(PARENT / "completion.json")["manifest"].items():
        assert sha(PARENT / path) == digest
    rows, totals = [], []
    for account in read(PARENT / "summary.json")["records"]:
        path = PARENT / account["name"] / "state.json"
        state = read(path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        legacy, published, seen = D(0), D(0), set()
        for fill in state["fills"]:
            key = (fill["day"], fill["symbol"], fill["side"])
            assert key not in seen
            seen.add(key)
            old, new = (
                D(str(fill["fee_parts"]["commission"])),
                commission(fill["quantity"], fill["price"]),
            )
            legacy += old
            published += new
            rows.append(
                {
                    "account": account["name"],
                    "day": fill["day"],
                    "symbol": fill["symbol"],
                    "side": fill["side"],
                    "shares": fill["quantity"],
                    "price": fill["price"],
                    "old_commission": str(old),
                    "published_unrounded_commission": str(new),
                    "difference": str(new - old),
                }
            )
        totals.append(
            {
                "account": account["name"],
                "fills": len(state["fills"]),
                "legacy_commission_usd": str(legacy),
                "published_unrounded_commission_usd": str(published),
                "difference_usd": str(published - legacy),
            }
        )
    result = {
        "schema": "published-broker-frozen-fill-fee-calibration.v1",
        "broker_reference": "IBKR Pro US stocks fixed, USD RegT margin research reference",
        "published_formula": "min(max(0.005*shares,1),0.01*shares*price)",
        "captured_rule_not_certified_for_historical_trade_dates": True,
        "regulatory_fees_not_replaced_or_assumed_zero": True,
        "billing_rounding_and_order_grouping_not_certified": True,
        "formula_examples": examples,
        "accounts": totals,
        "fills_checked": len(rows),
        "broker_authenticated": False,
        "historical_short_borrow_data_available": False,
        "real_execution_verified": False,
        "new_financial_accounts": 0,
        "counterfactual_nav_or_cagr_generated": False,
        "profit_goal_met": False,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "fills.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (output / "registration.json").write_text(
        json.dumps(
            {
                "inputs_sha256": inputs,
                "analysis_sha256": sha(Path(__file__)),
                "external_attestation": False,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
