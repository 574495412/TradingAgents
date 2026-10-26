"""Re-use sealed official bodies to distinguish spot and perpetual end schedules."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

BOUNDARIES = [
    ("FTMUSDT", "aec6fcbc84b749eeab6690e6bcac2f3d", "2025-01-13 03:00"),
    ("BNXUSDT", "2f963977c7274e0583f16f2e26987b61", "2025-03-18 03:00"),
    ("BAKEUSDT", "b601f1d986ad403b8c667dd0ca17e2e7", "2025-09-17 03:00"),
    ("HIFIUSDT", "b601f1d986ad403b8c667dd0ca17e2e7", "2025-09-17 03:00"),
    ("OMUSDT", "4d13052b4be943f5b155cb9150ca902c", "2026-03-02 03:00"),
    ("TONUSDT", "fe307fba935b44698fde4db01e84a7eb", "2026-06-30 03:00"),
]


def build(source, registry, feasibility, output):
    for p in [source, registry, feasibility]:
        verify_manifest(p)
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "registration.json",
        {
            "purpose": "Six observed partial days, source-derived schedules only; no financial outcome or strategy parameter change.",
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    source / "completion.json",
                    registry / "completion.json",
                    feasibility / "completion.json",
                ]
            },
            "fixed_scope": BOUNDARIES,
            "maximum_new_http_requests": 0,
        },
    )
    cards = read(registry / "registry.json")["cards"]
    raw_root = Path("data/research/contract_lifecycle_notices_20261026_v001/details")
    rows, references = [], {}
    daily = read(source / "summary.json")["records"]
    for symbol, code, stop in BOUNDARIES:
        raw, text_path = raw_root / (code + ".raw"), raw_root / (code + ".txt")
        receipt = read(raw_root / (code + ".receipt.json"))
        if receipt["status"] != 200 or receipt["body_sha256"] != sha(raw):
            raise ValueError("Adopted official detail changed")
        body = read(raw)["data"]
        text = text_path.read_text().replace("\u00a0", " ")
        if stop not in text or symbol.removesuffix("USDT") not in text:
            raise ValueError("Explicit spot schedule or asset not present in retained source")
        if code not in references:
            references[code] = {
                "url": "https://www.binance.com/en/support/announcement/detail/" + code,
                "title": body["title"],
                "raw_path": str(raw),
                "raw_sha256": sha(raw),
                "text_path": str(text_path),
                "text_sha256": sha(text_path),
                "publish_at_claimed": pd.Timestamp(
                    body["publishDate"], unit="ms", tz="UTC"
                ).isoformat(),
                "current_body_only": True,
            }
        end = pd.Timestamp(stop, tz="UTC")
        partial = [
            x for r in daily if r["symbol"] == symbol for x in r.get("partial_daily_labels", [])
        ]
        if len(partial) != 1 or not pd.Timedelta(0) < end - pd.Timestamp(
            partial[0]["actual_close"]
        ) <= pd.Timedelta(milliseconds=1):
            raise ValueError("Real shortened source candle does not end at published spot stop")
        contracts = [r for r in cards if r["symbol"] == symbol]
        if len(contracts) != 1:
            raise ValueError("Expected exactly one frozen perpetual terminal schedule")
        contract = contracts[0]
        if sha(contract["source_raw"]) != contract["source_raw_sha256"]:
            raise ValueError("Parent perpetual schedule body changed")
        known = pd.Timestamp(references[code]["publish_at_claimed"]) + pd.Timedelta(minutes=5)
        rows.append(
            {
                "symbol": symbol,
                "spot_source_code": code,
                "spot_stop_at": end.isoformat(),
                "spot_available_at_assumed": known.isoformat(),
                "spot_entry_cut_assumed": (end - pd.Timedelta(minutes=30)).isoformat(),
                "spot_entry_cut_is_research_rule": True,
                "actual_partial_day": partial[0],
                "perpetual_source_url": contract["source_url"],
                "perpetual_source_raw": contract["source_raw"],
                "perpetual_source_raw_sha256": contract["source_raw_sha256"],
                "perpetual_stop_at": contract["settlement_at"],
                "perpetual_entry_cut_at": contract["entry_stop_at"],
                "perpetual_available_at_assumed": contract["available_at_assumed"],
                "same_market_end_time": end == pd.Timestamp(contract["settlement_at"]),
                "new_asset_alias_transfer_allowed": False,
                "automatic_settlement_cash_certified": False,
                "historical_original_body_certified": False,
            }
        )
    data = read(feasibility / "slot_data_feasibility.json")
    guarded = [
        r
        for r in data
        if r["variant"] == "daily_archive"
        and r["window"] == "continuous2025_2026"
        and r["guarded_data_only_candidate"]
    ]
    guards = []
    for r in guarded:
        card = next((c for c in rows if c["symbol"] == r["symbol"]), None)
        if card is None:
            raise ValueError("Guarded slot has no reviewed boundary")
        t = pd.Timestamp(r["decision"])
        clock = min(
            pd.Timestamp(card["spot_entry_cut_assumed"]),
            pd.Timestamp(card["perpetual_entry_cut_at"]),
        )
        guards.append(
            {
                "symbol": r["symbol"],
                "decision": r["decision"],
                "past7_rate": r["known_rate_sum"],
                "hours_until_first_leg_exit_cut": (clock - t).total_seconds() / 3600,
                "forecast_is_not_cash": True,
                "account_executed": False,
            }
        )
    save(output / "boundaries.json", rows)
    save(output / "references.json", list(references.values()))
    save(
        output / "summary.json",
        {
            "schema": "matched-carry-spot-perp-boundaries.v1",
            "reviewed_symbols": len(rows),
            "all_six_spot_ends_match_actual_shortened_source": True,
            "same_market_end_times": sum(r["same_market_end_time"] for r in rows),
            "guarded_continuous_primary_slots": guards,
            "adopted_official_details": len(references),
            "new_http_requests": 0,
            "mapping_for_all_candidates_complete": False,
            "financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--feasibility", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.registry, args.feasibility, args.output)
