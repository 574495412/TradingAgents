"""Register ten additional end clauses and one explicitly separated AERGO launch."""

import argparse
import copy
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

ADDITIONS = [
    (
        "ALPACAUSDT",
        "0274f9d47da1437990bc13eb17b0ec99",
        "2025-04-30 09:00",
        "2025-04-30 08:30",
        "ALPACA",
        True,
    ),
    (
        "DEGOUSDT",
        "d067b373496c44a4ab8d28f3e90b66ad",
        "2026-04-21 09:00",
        "2026-04-21 08:30",
        "DEGO",
        True,
    ),
    (
        "HIGHUSDT",
        "24aced8c698649d9a552a47eeead5e34",
        "2026-06-11 09:00",
        "2026-06-11 08:30",
        "HIGH",
        True,
    ),
    (
        "LRCUSDT",
        "ddac99c0e5044c13a13c7c113e530e2b",
        "2026-03-24 09:00",
        "2026-03-24 08:30",
        "LRC",
        True,
    ),
    (
        "ICXUSDT",
        "d72915ed7a60473b92f0818d959a227a",
        "2026-08-26 09:00",
        "2026-08-26 08:30",
        "ICX",
        True,
    ),
    (
        "STORJUSDT",
        "d72915ed7a60473b92f0818d959a227a",
        "2026-08-26 09:00",
        "2026-08-26 08:30",
        "STORJ",
        True,
    ),
    (
        "VANRYUSDT",
        "64cf7544b98e4f2ca7f54ba11fd72d9d",
        "2026-08-07 09:00",
        "2026-08-07 08:30",
        "VANRY",
        True,
    ),
    (
        "BTTUSDT",
        "2722e6da4f5141dd9b2fb07b5b1f3f75",
        "2022-01-14 02:00",
        None,
        "BTT/USDT futures contract",
        False,
    ),
    (
        "TOMOUSDT",
        "034eee2293964e26b915b1ec5c42972a",
        "2023-11-14 09:00",
        "2023-11-14 08:30",
        "USDT-Margined TOMO Perpetual Contract",
        False,
    ),
    (
        "FRONTUSDT",
        "3ab5488a00e04d4fb338c77ea28326a8",
        "2024-08-23 09:00",
        "2024-08-23 08:30",
        "FRONTUSDT",
        False,
    ),
]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    parent = Path("data/research/contract_lifecycle_notices_20261026_v001")
    extra = Path("data/research/contract_segment_followup_20261026_v002")
    daily = Path("data/research/historical_perpetual_daily_20261026_v002")
    for root in [parent, extra, daily]:
        verify_manifest(root)
    old_path = Path(__file__).with_name("event_cards.json")
    cards = copy.deepcopy(read(old_path)["cards"])
    meta_path = Path("data/research/historical_universe_sources_20261026_v001/exchange_info.json")
    metadata = {r["symbol"]: r for r in read(meta_path)["symbols"]}
    bindings = {}
    for s, code, end, cut, product, indirect in ADDITIONS:
        source = parent / "details" if indirect else extra
        raw, text_path = source / (code + ".raw"), source / (code + ".txt")
        data, text = read(raw)["data"], text_path.read_text()
        if product not in text or "Binance Futures" not in text:
            raise ValueError("Explicit token/product reference absent")
        region = text
        if indirect:
            at = text.index("\nFutures\n")
            following = text.find("\nFunding Rate Arbitrage Bot\n", at)
            region = text[at : following if following >= 0 else None]
            if "contracts of the aforementioned token" not in region:
                raise ValueError("Missing explicit anaphoric futures scope")
            m = metadata[s]
            if (
                m["contractType"] != "PERPETUAL"
                or m["quoteAsset"] != "USDT"
                or m["marginAsset"] != "USDT"
            ):
                raise ValueError("Current product type does not corroborate USD-M USDT")
        if end not in region or (cut and cut not in region):
            raise ValueError("End or cutoff not in bound futures clause")
        pub = pd.Timestamp(data["publishDate"], unit="ms", tz="UTC")
        c = {
            "id": s + ":" + code,
            "symbol": s,
            "episode": s + ":" + end[:10],
            "contract_type": "USD_M_PERPETUAL",
            "kind": "terminal_schedule",
            "supersedes": None,
            "source_raw": str(raw),
            "source_raw_sha256": sha(raw),
            "source_text": str(text_path),
            "source_text_sha256": sha(text_path),
            "source_code": code,
            "source_url": "https://www.binance.com/en/support/announcement/detail/" + code,
            "source_version": data.get("version"),
            "publish_at_claimed": pub.isoformat(),
            "available_at_assumed": (pub + pd.Timedelta(minutes=5)).isoformat(),
            "settlement_at": pd.Timestamp(end, tz="UTC").isoformat(),
            "entry_stop_at": pd.Timestamp(cut, tz="UTC").isoformat() if cut else None,
            "product_literal": product,
            "settlement_literal": end,
            "entry_literal": cut,
            "scope_match": "named token and explicit Futures anaphor; conditional applicability, current type corroboration only"
            if indirect
            else "explicit USD-M token/product clause",
            "scope_indirect": indirect,
            "historical_type_and_status_certified": False,
            "original_historical_content_certified": False,
            "original_historical_revision_time_certified": False,
            "terminal_settlement_price": None,
            "terminal_settlement_fee": None,
            "new_episode_start_certified": False,
            "observed_amendment_notes_in_current_text": False,
        }
        if indirect:
            c["current_type_metadata_sha256"] = sha(meta_path)
        cards.append(c)
        bindings.update({str(raw): sha(raw), str(text_path): sha(text_path)})
    code = "82f730b7ef444a38b323ab7a2e56b757"
    raw, text_path = extra / (code + ".raw"), extra / (code + ".txt")
    data, text = read(raw)["data"], text_path.read_text()
    if (
        "AERGOUSDT" not in text
        or "2025-04-16 11:00" not in text
        or "USDⓈ-M Perpetual Contract" not in text
    ):
        raise ValueError("AERGO launch product/date missing")
    pub = pd.Timestamp(data["publishDate"], unit="ms", tz="UTC")
    first = next(
        c
        for c in cards
        if c["symbol"] == "AERGOUSDT" and c["settlement_at"].startswith("2025-03-27")
    )
    second = next(
        c
        for c in cards
        if c["symbol"] == "AERGOUSDT" and c["settlement_at"].startswith("2026-07-24")
    )
    launch = {
        "id": "AERGOUSDT:launch:" + code,
        "symbol": "AERGOUSDT",
        "episode": second["episode"],
        "reopens_after_episode": first["episode"],
        "launch_at": "2025-04-16T11:00:00+00:00",
        "publish_at_claimed": pub.isoformat(),
        "available_at_assumed": (pub + pd.Timedelta(minutes=5)).isoformat(),
        "source_raw": str(raw),
        "source_raw_sha256": sha(raw),
        "source_text": str(text_path),
        "source_text_sha256": sha(text_path),
        "source_url": "https://www.binance.com/en/support/announcement/detail/" + code,
        "current_official_launch_clause_verified": True,
        "original_historical_content_certified": False,
        "portfolio_risk_reset": False,
        "feature_lookback_reset": True,
    }
    bindings.update({str(raw): sha(raw), str(text_path): sha(text_path)})
    registry = {
        "schema": "contract-segment-registry.v1",
        "cards": cards,
        "launch_cards": [launch],
        "source_conflicts": [
            {
                "symbol": "BLZUSDT",
                "early_end": "2022-08-25T09:00:00Z",
                "notice_id": "BLZUSDT:b312715afbff4d7493b77089967303b1",
                "reported_trades_after_notice_end": True,
                "trade_archive_summary_sha256": sha(extra / "summary.json"),
                "relisting_or_cancellation_verified": False,
                "automatic_source_choice": False,
            }
        ],
        "partial_historical_version_assumption": True,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "registry.json", registry)
    paths = [
        Path(__file__),
        old_path,
        meta_path,
        parent / "completion.json",
        extra / "completion.json",
        daily / "completion.json",
        Path(__file__).with_name("segment_plan.json"),
        Path(__file__).with_name("segment_collection_recovery.json"),
    ]
    save(
        output / "plan.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "data": str(daily),
            "parent_preview": "results/research/contract_lifecycle_timeline_20261026_v001",
            "age_policy": "For an explicitly announced new episode, start full-daily features at ceil(launch UTC,day), excluding the partial opening day; count180validfullobservations inside that segment only. No old-contract carryover or account risk reset.",
            "before_new_preview": True,
            "new_financial_accounts": 0,
        },
    )
    save(output / "bindings.json", bindings)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(
        {
            "end_cards": len(cards),
            "symbols": len({c["symbol"] for c in cards}),
            "launch_cards": 1,
            "source_conflicts": 1,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
