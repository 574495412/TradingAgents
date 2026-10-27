"""Case-window raw CSV/Decimal and official product-clause review before replay."""

import argparse
import csv
import io
import re
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.contract_lifecycles.collect import text_nodes
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def check_csv(source, case, market):
    s, month = case["symbol"], case["month"]
    archive = source / "raw" / market / f"{s}-5m-{month}.zip"
    if sha(archive) != archive.with_suffix(".zip.CHECKSUM").read_text().split()[0]:
        raise ValueError("Independent source checksum mismatch")
    with zipfile.ZipFile(archive) as z, io.TextIOWrapper(z.open(z.namelist()[0])) as f:
        rows = list(csv.reader(f))
    if rows[0][0] == "open_time":
        rows = rows[1:]
    unit = 1000 if market == "spot" else 1_000_000
    start = (pd.Timestamp(case["decision"]) - pd.Timedelta(minutes=5)).as_unit("ns").value
    finish = pd.Timestamp(case["exit_deadline"]).as_unit("ns").value
    clocks, values = [], []
    for r in rows:
        t = int(r[0]) * unit
        if not start <= t < finish:
            continue
        duration_gap = t + 300_000_000_000 - int(r[6]) * unit
        if not 0 < duration_gap <= 1_000_000:
            raise ValueError("Case raw candle is shortened or has wrong timestamp unit")
        v = [Decimal(r[i]) for i in [1, 2, 3, 4, 5, 7]]
        if not all(x.is_finite() for x in v):
            raise ValueError("Unknown actual source value")
        o, h, low, c, volume, quote = v
        if (
            min(o, h, low, c) <= 0
            or min(volume, quote) < 0
            or h < max(o, c, low)
            or low > min(o, c, h)
        ):
            raise ValueError("Raw price/volume envelope invalid")
        if market != "mark":
            tol = Decimal(".01") + Decimal(".000001") * abs(h * volume)
            if not low * volume - tol <= quote <= h * volume + tol:
                raise ValueError("Reported quote volume outside raw trade envelope")
        clocks.append(t)
        values.append([float(x) for x in v])
    frame = pd.read_parquet(source / "symbols" / f"{s}_{market}_5m.parquet")
    if not np.array_equal(clocks, frame.index.as_unit("ns").asi8) or not np.allclose(
        values, frame.to_numpy(), rtol=1e-14, atol=1e-12
    ):
        raise ValueError("Case raw numeric values or actual clocks differ from derived input")
    return len(frame)


def official(raw, code, token, market, decision):
    body = read(raw)
    receipt = read(raw.with_suffix(".receipt.json"))
    if receipt["status"] != 200 or sha(raw) != receipt["body_sha256"]:
        raise ValueError("Official source body or HTTP receipt changed")
    data = body["data"]
    import json

    text = text_nodes(json.loads(data["body"])).replace("\u00a0", " ")
    literal = token + "/USDT" if market == "spot" else token + "USDT"
    if not re.search(r"(?<![A-Z0-9])" + re.escape(literal) + r"(?![A-Z0-9])", text):
        raise ValueError(
            "Exact market-specific asset symbol absent; substring cannot approve identity"
        )
    available = pd.Timestamp(data["publishDate"], unit="ms", tz="UTC") + pd.Timedelta(minutes=5)
    if available > pd.Timestamp(decision):
        raise ValueError("Identity evidence was not visible at registered case decision")
    return {
        "url": "https://www.binance.com/en/support/announcement/detail/" + code,
        "title": data["title"],
        "raw_path": str(raw),
        "raw_sha256": sha(raw),
        "available_at_assumed": available.isoformat(),
        "product": market,
        "matched_literal": literal,
        "current_body_only": True,
    }


def audit(source, output):
    verify_manifest(source)
    plan = read(source / "plan.json")
    for p, h in read(source / "registration.json")["source_hashes"].items():
        if sha(p) != h:
            raise ValueError("Frozen preparation definition changed")
    output.mkdir(parents=True, exist_ok=False)
    cards = read(Path(plan["parent_boundaries"]) / "boundaries.json")
    horizon = read(Path(plan["parent_horizon"]) / "summary.json")["old_guarded_details"]
    checks = []
    for case in plan["cases"]:
        s = case["symbol"]
        token = s.removesuffix("USDT")
        t = pd.Timestamp(case["decision"])
        end = pd.Timestamp(case["exit_deadline"])
        counts = {
            market: check_csv(source, case, market) for market in ["spot", "perpetual", "mark"]
        }
        actual = pd.read_parquet(source / "funding" / (s + "_exact.parquet"))
        raw = []
        for p in sorted((source / "funding").glob(s + "_page*.json")):
            raw.extend(read(p)["response"]["structuredContent"]["result"])
        selected = [
            r
            for r in raw
            if t - pd.Timedelta(days=7)
            <= pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC")
            <= end
        ]
        selected.sort(key=lambda r: r["fundingTime"])
        clocks = pd.DatetimeIndex(
            pd.to_datetime([r["fundingTime"] for r in selected], unit="ms", utc=True)
        )
        if not clocks.equals(actual.index) or clocks.has_duplicates:
            raise ValueError("Raw funding event grain differs")
        for r, (_, q) in zip(selected, actual.iterrows(), strict=True):
            if (
                float(Decimal(r["fundingRate"])) != q.funding_rate
                or float(Decimal(r["markPrice"])) != q.mark
                or q.mark <= 0
            ):
                raise ValueError("Raw actual settlement rate or mark differs")
        past = sum(
            (
                Decimal(r["fundingRate"])
                for r in selected
                if t - pd.Timedelta(days=7) <= pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC")
                and pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC") + pd.Timedelta(minutes=5)
                <= t
            ),
            Decimal(0),
        )
        prior = next(r for r in horizon if r["symbol"] == s)
        if abs(float(past) - prior["known_rate_sum"]) > 1e-11:
            raise ValueError("Known past funding forecast differs")
        card = next(r for r in cards if r["symbol"] == s)
        visible = (
            [pd.Timestamp(card["spot_entry_cut_assumed"])]
            if pd.Timestamp(card["spot_available_at_assumed"]) <= t
            else []
        )
        if pd.Timestamp(card["perpetual_available_at_assumed"]) <= t:
            visible.append(pd.Timestamp(card["perpetual_entry_cut_at"]))
        if min(visible) != end:
            raise ValueError("Case end used an invisible or wrong-product deadline")
        spot_code = card["spot_source_code"]
        spot_raw = Path("data/research/contract_lifecycle_notices_20261026_v001/details") / (
            spot_code + ".raw"
        )
        refs = [official(spot_raw, spot_code, token, "spot", case["decision"])]
        if s == "BNXUSDT":
            refs.append(official(spot_raw, spot_code, token, "perpetual", case["decision"]))
        else:
            code = "60e59bfb64ce4f76a1cf22d5cb7282aa"
            path = Path("data/research/matched_carry_asset_sources_20261026_v001/details") / (
                code + ".raw"
            )
            refs.append(official(path, code, token, "perpetual", case["decision"]))
        checks.append(
            {
                "symbol": s,
                "raw_source_rows_by_market": counts,
                "raw_funding_events_including_history": len(actual),
                "past7_rate_decimal": str(past),
                "known_exit_deadline": end.isoformat(),
                "identity_status": "explicit_two_product_clauses_under_current_body_assumption",
                "identity_references": refs,
                "original_historical_version_certified": False,
                "customer_cashflow_certified": False,
            }
        )
    result = {
        "schema": "carry-case-input-arithmetic-review.v1",
        "passed": True,
        "source_completion_sha256": sha(source / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "case_checks": checks,
        "financial_case_paths": 0,
        "financial_strategy_accounts": 0,
        "external_third_party_review": False,
        "independent_oos": False,
        "profit_goal_met": False,
    }
    save(output / "verification.json", result)
    save(
        output / "completion.json",
        {
            "manifest": {"verification.json": sha(output / "verification.json")},
            "financial_case_paths": 0,
        },
    )
    print(
        {
            "cases": len(checks),
            "raw_csv_and_funding_review_passed": True,
            "financial_case_paths": 0,
            "external_third_party_review": False,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.source, args.output)
