"""Source-bound ticker identities and type restrictions, with provisional availability."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.contract_lifecycles.segments import context_at
from research_tools.contract_lifecycles.timeline import validate
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def source(root, code, symbols, literal):
    text = (root / (code + ".txt")).read_text()
    if not all(s in text for s in symbols) or literal not in text:
        raise ValueError("Named identity/type clause differs: " + code)
    data = read(root / (code + ".raw"))["data"]
    published = pd.to_datetime(data["publishDate"], unit="ms", utc=True)
    return {
        "source_code": code,
        "source_url": "https://www.binance.com/en/support/announcement/detail/" + code,
        "source_raw": str(root / (code + ".raw")),
        "source_raw_sha256": sha(root / (code + ".raw")),
        "source_text_sha256": sha(root / (code + ".txt")),
        "publish_at_claimed": published.isoformat(),
        "available_at_assumed": (published + pd.Timedelta(minutes=5)).isoformat(),
        "original_historical_content_certified": False,
        "original_historical_revision_time_certified": False,
    }


def prepare(capture, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(capture)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    if read(capture / "registration.json")["study_plan_sha256"] != sha(p):
        raise ValueError("Frozen identity capture belongs to another study")
    registry = read(Path(plan["parent_registry"]))
    for s, code, terminal, cut, asset in [
        (
            "LITUSDT",
            "2a9feaa556f74dcdaa2192366f0e247c",
            "2025-01-31T09:00:00Z",
            "2025-01-31T08:30:00Z",
            "Litentry",
        ),
        (
            "PUMPUSDT",
            "38a74ee47cdc43ebbc672e1dc6968358",
            "2025-06-13T09:00:00Z",
            "2025-06-13T08:30:00Z",
            "PumpBTC",
        ),
    ]:
        info = source(capture, code, [s], terminal[:10] + " " + terminal[11:16])
        registry["cards"].append(
            {
                **info,
                "id": s + ":" + code,
                "symbol": s,
                "episode": s + ":" + terminal[:10],
                "underlying_identity": asset,
                "contract_type": "USD_M_PERPETUAL",
                "supersedes": None,
                "settlement_at": pd.Timestamp(terminal).isoformat(),
                "entry_stop_at": pd.Timestamp(cut).isoformat(),
                "terminal_settlement_price": None,
                "terminal_settlement_fee": None,
            }
        )
    for s, code, launch, episode, asset in [
        (
            "PUMPUSDT",
            "4a4bb43f865a4348828de4facfc9eee4",
            "2025-04-12T14:30:00Z",
            "PUMPUSDT:2025-06-13",
            "PumpBTC",
        ),
        (
            "PUMPUSDT",
            "4bc8b483d10d4619babb2015066b2d89",
            "2025-07-10T07:30:00Z",
            "PUMPUSDT:Pump.fun@2025-07-10",
            "Pump.fun",
        ),
        (
            "LITUSDT",
            "6a33be00231c4539b3a4a625538e4d1e",
            "2025-12-23T17:30:00Z",
            "LITUSDT:Lighter@2025-12-23",
            "Lighter",
        ),
    ]:
        info = source(capture, code, [s], launch[:10] + " " + launch[11:16])
        registry["launch_cards"].append(
            {
                **info,
                "id": s + ":launch:" + code,
                "symbol": s,
                "episode": episode,
                "launch_at": pd.Timestamp(launch).isoformat(),
                "underlying_identity": asset,
                "portfolio_risk_reset": False,
                "terminal_cash_values_verified": False,
            }
        )
    type_cards = []
    for code, symbols, kind, literal in [
        ("d592f6ba938746cbadaaf5a8a714abd6", ["INTCUSDT"], "equity", "Intel"),
        ("f083bdc8f37d4d45a0e0ec6fd5cd4086", ["CRCLUSDT", "MSTRUSDT"], "equity", "Common Stock"),
        ("bd92dee1018f421fae1ace119fc2f706", ["XAGUSDT"], "commodity", "Silver"),
        ("ecf7318c0d434c339e80878588e700d0", ["XAUUSDT"], "commodity", "Gold"),
    ]:
        info = source(capture, code, symbols, literal)
        type_cards.extend({**info, "symbol": s, "kind": kind} for s in symbols)
    registry["type_cards"] = type_cards
    validate(registry["cards"])
    save(output / "registry.json", registry)
    save(
        output / "registration.json",
        {
            "study_plan_sha256": sha(p),
            "capture_completion_sha256": sha(capture / "completion.json"),
            "parent_registry_sha256": sha(Path(plan["parent_registry"])),
            "source_sha256": sha(Path(__file__)),
            "historical_original_versions_certified": False,
            "financial_accounts": 0,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
        },
    )
    print(
        {
            "end_cards": len(registry["cards"]),
            "launch_cards": len(registry["launch_cards"]),
            "type_cards": len(type_cards),
        },
        flush=True,
    )


def crypto_context(registry, symbol, clock):
    context = context_at(registry, symbol, clock)
    typed = [
        r
        for r in registry["type_cards"]
        if r["symbol"] == symbol and pd.Timestamp(r["available_at_assumed"]) <= clock
    ]
    return {**context, "blocked": context["blocked"] or bool(typed), "known_noncrypto": bool(typed)}


def directives(registry, symbols):
    out = {}
    for s in symbols:
        clocks = {
            pd.Timestamp(r["available_at_assumed"])
            for r in registry["cards"] + registry["type_cards"]
            if r["symbol"] == s
        }
        clocks.update(
            max(pd.Timestamp(r["available_at_assumed"]), pd.Timestamp(r["launch_at"]))
            for r in registry["launch_cards"]
            if r["symbol"] == s
        )
        rows = []
        for clock in sorted(clocks):
            c = crypto_context(registry, s, clock)
            rows.append(
                {
                    "known_at": clock.isoformat(),
                    "episode_key": c["active_contract_key"],
                    "entry_stop_at": c["known_entry_cutoff"],
                    "terminal_at": c["known_terminal"],
                    "blocked": c["blocked"],
                    "historical_tradability_certified": False,
                }
            )
        if rows:
            out[s] = rows
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.capture, a.output)
