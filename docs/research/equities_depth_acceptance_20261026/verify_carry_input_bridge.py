"""Check frozen carry-case input clocks and their scope for the equity goal.

No case ledger, strategy account, trade or profit calculation is executed.
The separately owned carry-case preparation and outputs remain read-only.
"""

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/research/carry_exit_cases_20261026_v003"
BOUNDARIES = ROOT / "results/research/matched_carry_boundaries_20261026_v001/boundaries.json"
STOCK = ROOT / "results/research/equities_pit_stocks_context_20261027_v001"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock(value):
    return datetime.fromisoformat(value)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    completion = read(SOURCE / "completion.json")
    for path, digest in completion["manifest"].items():
        assert sha(SOURCE / path) == digest, path
    plan, input_summary = read(SOURCE / "plan.json"), read(SOURCE / "summary.json")
    cards = {r["symbol"]: r for r in read(BOUNDARIES)}
    stock_state = STOCK / "US_french_mom_12_2_10000_15bps/state.json"
    stock_curve = read(stock_state)["equity_curve"]
    assert "day" not in stock_curve[0] and stock_curve[0]["clock"] < stock_curve[1]["clock"]
    stock_days = {r["day"] for r in stock_curve[1:]}
    checks, inputs = [], {str(BOUNDARIES.relative_to(ROOT)): sha(BOUNDARIES)}
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for case in plan["cases"]:
        symbol = case["symbol"]
        start, end = clock(case["decision"]), clock(case["exit_deadline"])
        begin = start - timedelta(minutes=5)
        expected = []
        t = begin
        while t < end:
            expected.append(t)
            t += timedelta(minutes=5)
        markets = []
        for market in ("spot", "perpetual", "mark"):
            archive = SOURCE / "raw" / market / f"{symbol}-5m-{case['month']}.zip"
            assert sha(archive) == archive.with_suffix(".zip.CHECKSUM").read_text().split()[0]
            with zipfile.ZipFile(archive) as z:
                assert len(z.namelist()) == 1
                with io.TextIOWrapper(z.open(z.namelist()[0])) as f:
                    rows = list(csv.reader(f))
            if rows[0][0] == "open_time":
                rows = rows[1:]
            unit = "microseconds" if market == "spot" else "milliseconds"
            scale = 1 if market == "spot" else 1000
            clocks, close_gaps_us = [], []
            for row in rows:
                opening = epoch + timedelta(microseconds=int(row[0]) * scale)
                if begin <= opening < end:
                    closing = epoch + timedelta(microseconds=int(row[6]) * scale)
                    gap = opening + timedelta(minutes=5) - closing
                    gap_us = gap // timedelta(microseconds=1)
                    assert 0 < gap_us <= 1000
                    clocks.append(opening)
                    close_gaps_us.append(gap_us)
            assert clocks == expected
            derived = pd.read_parquet(SOURCE / "symbols" / f"{symbol}_{market}_5m.parquet")
            assert list(derived.index.to_pydatetime()) == clocks
            assert clocks[-1] + timedelta(minutes=5) == end
            assert end not in clocks
            markets.append(
                {
                    "market": market,
                    "raw_epoch_unit": unit,
                    "rows": len(clocks),
                    "first_open": clocks[0].isoformat(),
                    "last_open": clocks[-1].isoformat(),
                    "last_completion_boundary": end.isoformat(),
                    "raw_close_gap_microseconds": sorted(set(close_gaps_us)),
                    "deadline_open_bar_excluded": True,
                }
            )
        card = cards[symbol]
        available_deadlines = []
        for available_key, cut_key in (
            ("spot_available_at_assumed", "spot_entry_cut_assumed"),
            ("perpetual_available_at_assumed", "perpetual_entry_cut_at"),
        ):
            if clock(card[available_key]) <= start:
                available_deadlines.append(clock(card[cut_key]))
        assert min(available_deadlines) == end
        funding = pd.read_parquet(SOURCE / "funding" / f"{symbol}_exact.parquet")
        assert funding.index.is_unique and funding.index.is_monotonic_increasing
        assert funding.index.max().to_pydatetime() <= end
        cash_window = funding.loc[(funding.index > start) & (funding.index <= end)]
        case_days = {t.date().isoformat() for t in expected if t >= start}
        checks.append(
            {
                "symbol": symbol,
                "decision": start.isoformat(),
                "deadline": end.isoformat(),
                "markets": markets,
                "reported_funding_events_after_decision_through_deadline": len(cash_window),
                "future_events_are_not_decision_features": True,
                "known_market_specific_earliest_cutoff_used": True,
                "later_perpetual_stop_notice_available_at_decision": (
                    clock(card["perpetual_available_at_assumed"]) <= start
                ),
                "common_calendar_dates_with_frozen_stock_account": len(case_days & stock_days),
                "source_available_clock_is_current_body_plus_five_minute_assumption": True,
                "historical_notice_version_certified": False,
                "customer_payable_cash_certified": False,
            }
        )
    assert (
        input_summary["financial_case_paths"] == input_summary["financial_strategy_accounts"] == 0
    )
    inputs[str(stock_state.relative_to(ROOT))] = sha(stock_state)
    inputs[str((SOURCE / "completion.json").relative_to(ROOT))] = sha(SOURCE / "completion.json")
    inputs.update(
        {str((SOURCE / p).relative_to(ROOT)): h for p, h in completion["manifest"].items()}
    )
    result = {
        "schema": "equity-carry-input-bridge.v1",
        "input_scope": str(SOURCE.relative_to(ROOT)),
        "manifest_files_verified": len(completion["manifest"]),
        "checks": checks,
        "two_selected_cases_are_not_independent_strategy_samples": True,
        "previous_2024_training_carry_date_overlap_not_overridden": True,
        "admitted_crypto_modules_added": 0,
        "new_financial_accounts": 0,
        "financial_case_paths_executed_by_this_bridge": 0,
        "unified_portfolio_evaluated": False,
        "external_independent_attestation": False,
        "profit_goal_met": False,
        "scope": "Input-clock and evidence-scope check only; original notices are current-body captures, public rates are not customer receipts, bar-volume capacity is not actual execution. No profit or module admission follows.",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "registration.json").write_text(
        json.dumps(
            {
                "inputs_sha256": inputs,
                "analysis_sha256": sha(Path(__file__)),
                "hash_scope": "Current internal consistency, not historical publication or external identity proof.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
