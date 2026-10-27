"""Compare original exported trades/orders and independently reconcile closed cash."""

import argparse
import hashlib
import json
import math
import zipfile
from decimal import Decimal, localcontext
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right, path=""):
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append({"path": path, "reason": "keys differ"})
        for key in left.keys() & right.keys():
            differences.extend(compare(left[key], right[key], path + "/" + key))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "reason": "length differs"})
        for i, (a, b) in enumerate(zip(left, right, strict=False)):
            differences.extend(compare(a, b, path + "/" + str(i)))
    elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if not (left == right or math.isclose(left, right, rel_tol=0, abs_tol=1e-8)):
            differences.append({"path": path, "left": left, "right": right})
    elif left != right:
        differences.append({"path": path, "left": left, "right": right})
    return differences


def cash_from_orders(trade):
    with localcontext() as context:
        context.prec = 50
        def d(value):
            return Decimal(str(value))
        entries = [o for o in trade["orders"] if o["ft_is_entry"]]
        exits = [o for o in trade["orders"] if not o["ft_is_entry"]]
        entry_value = sum((d(o["amount"]) * d(o["safe_price"]) for o in entries), Decimal(0))
        exit_value = sum((d(o["amount"]) * d(o["safe_price"]) for o in exits), Decimal(0))
        quantity_residual = sum((d(o["amount"]) for o in entries), Decimal(0)) - sum(
            (d(o["amount"]) for o in exits), Decimal(0)
        )
        commission = entry_value * d(trade["fee_open"]) + exit_value * d(trade["fee_close"])
        price_pnl = (-1 if trade["is_short"] else 1) * (exit_value - entry_value)
        expected = price_pnl - commission + d(trade["funding_fees"])
        return {
            "pair": trade["pair"],
            "open_date": trade["open_date"],
            "quantity_residual": str(quantity_residual),
            "price_pnl": str(price_pnl),
            "commission": str(commission),
            "funding_model_cash": str(trade["funding_fees"]),
            "reconstructed_net_pnl": str(expected),
            "reported_net_pnl": str(trade["profit_abs"]),
            "absolute_error": str(abs(expected - d(trade["profit_abs"]))),
        }


def audit(author, local_zip, output):
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads(author.read_text())["strategy"]["NostalgiaForInfinityX8"]
    with zipfile.ZipFile(local_zip) as archive:
        name = next(
            n for n in archive.namelist() if n.endswith(".json") and not n.endswith("_config.json")
        )
        local = json.loads(archive.read(name))["strategy"]["NostalgiaForInfinityX8"]
    fields = [
        "trades",
        "pairlist",
        "results_per_pair",
        "total_trades",
        "wins",
        "losses",
        "profit_total_abs",
        "final_balance",
        "daily_profit",
        "wallet_stats",
        "max_drawdown_account",
        "rejected_signals",
    ]
    differences = {field: compare(original[field], local[field], field) for field in fields}
    cash = [cash_from_orders(trade) for trade in local["trades"]]
    cash_max = max(Decimal(row["absolute_error"]) for row in cash)
    quantity_max = max(abs(Decimal(row["quantity_residual"])) for row in cash)
    result = {
        "schema": "original-nfi-source-account-audit.v1",
        "author_json_sha256": sha(author),
        "local_zip_sha256": sha(local_zip),
        "source_fields_compared": fields,
        "original_trade_order_and_account_match": not any(differences.values()),
        "differences": differences,
        "trades": len(cash),
        "orders": sum(len(t["orders"]) for t in local["trades"]),
        "max_independent_cash_error_USDT": str(cash_max),
        "cash_tolerance_USDT": "0.000001",
        "cash_reconciliation_passed": cash_max <= Decimal("0.000001"),
        "max_inventory_quantity_residual": str(quantity_max),
        "inventory_reconciled": quantity_max <= Decimal("0.000000001"),
        "net_pnl_USDT": local["profit_total_abs"],
        "funding_scope": "Exported original model funding amounts, not actual customer receipts.",
        "historical_original_month_only": True,
        "third_party_review_performed": False,
        "unseen_validation_performed": False,
        "project_profit_goal_met": False,
        "orders_enabled": False,
    }
    (output / "cash_reconciliation.json").write_text(json.dumps(cash, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["differences", "source_fields_compared"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--local-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.author, args.local_zip, args.output)
