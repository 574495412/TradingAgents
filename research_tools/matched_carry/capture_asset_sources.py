"""Bounded primary-source asset review; current metadata never filters old slots."""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def select_sources(articles, first):
    selections, missing = {}, []
    for symbol, decision in first.items():
        base = symbol.removesuffix("USDT")
        boundary = re.compile(r"(?<![A-Z0-9])" + re.escape(base) + r"(?![A-Z0-9])", re.I)
        for market in ["spot", "perpetual"]:
            options = []
            for row in articles.values():
                title = row["title"]
                at = pd.Timestamp(row["releaseDate"], unit="ms", tz="UTC") + pd.Timedelta(minutes=5)
                if at > pd.Timestamp(decision):
                    continue
                is_perp = bool(re.search(r"Futures.*(?:Launch|List).*", title, re.I))
                is_spot = bool(
                    re.search(r"Binance (?:Will )?List[s]?\b", title, re.I)
                ) and not re.search(r"Futures|Margin", title, re.I)
                if (
                    market == "perpetual"
                    and is_perp
                    and (symbol in title or boundary.search(title))
                    or market == "spot"
                    and is_spot
                    and boundary.search(title)
                ):
                    options.append(row)
            if options:
                # Most recent explicit predecision listing evidence is sufficient;
                # do not pretend the capped catalog establishes first-ever listing.
                chosen = max(options, key=lambda r: (r["releaseDate"], r["code"]))
                selection = selections.setdefault(chosen["code"], {**chosen, "uses": []})
                selection["uses"].append(
                    {"symbol": symbol, "market": market, "first_decision": decision}
                )
            else:
                missing.append({"symbol": symbol, "market": market, "first_decision": decision})
    return selections, missing


def capture(plan_path, output):
    plan = read(plan_path)
    audit = Path(plan["daily_audit"])
    verify_manifest(audit)
    slots = [r for r in read(audit / "slot_data_feasibility.json") if r["data_eligible"]]
    first = {
        s: min(r["decision"] for r in slots if r["symbol"] == s)
        for s in sorted({r["symbol"] for r in slots})
    }
    budget = plan["identity_budget"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "details").mkdir()
    save(
        output / "registration.json",
        {
            "first_data_eligible_decision": first,
            "plan": plan,
            "source_hashes": {
                str(p): sha(p)
                for p in [
                    plan_path,
                    Path(__file__),
                    Path("research_tools/contract_lifecycles/collect.py"),
                    audit / "completion.json",
                ]
            },
            "before_source_requests": True,
            "financial_accounts": 0,
        },
    )
    old = Path("data/research/contract_lifecycle_notices_20261026_v001/catalog.json")
    articles = {r["code"]: r for r in read(old)["articles"]}
    traversals = []
    exhausted = False
    for page in range(1, budget["maximum_catalog_pages"] + 1):
        data = request(
            output,
            f"listing_{page:03d}",
            "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
            {"type": 1, "pageNo": page, "pageSize": budget["page_size"], "catalogId": 48},
        )
        catalog = next(c for c in data["catalogs"] if c["catalogId"] == 48)
        rows = catalog["articles"]
        for r in rows:
            articles.setdefault(r["code"], r)
        traversals.append(
            {
                "page": page,
                "articles": len(rows),
                "total": catalog["total"],
                "oldest_claimed": min([r["releaseDate"] for r in rows], default=None),
            }
        )
        if page % 10 == 0:
            print({"catalog_pages": page, "all_articles": len(articles)}, flush=True)
        if not rows or page * budget["page_size"] >= catalog["total"]:
            exhausted = True
            break
    selected, missing = select_sources(articles, first)
    if len(selected) > budget["maximum_new_detail_requests"]:
        raise ValueError("Deterministic detail selection exceeds registered budget")
    save(
        output / "catalog.json",
        {
            "articles": list(articles.values()),
            "adopted_catalog": str(old),
            "adopted_sha256": sha(old),
            "traversals": traversals,
            "exhausted": exhausted,
        },
    )
    save(
        output / "detail_selection.json",
        {"selected": list(selected.values()), "missing": missing, "before_details": True},
    )
    results = []

    def detail(row):
        code = row["code"]
        data = request(
            output / "details",
            code,
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            {"articleCode": code},
        )
        text = text_nodes(json.loads(data["body"]))
        (output / "details" / (code + ".txt")).write_text(text)
        return {
            "code": code,
            "title": data["title"],
            "publishDate": data["publishDate"],
            "uses": row["uses"],
            "raw_sha256": sha(output / "details" / (code + ".raw")),
            "text_sha256": sha(output / "details" / (code + ".txt")),
            "current_body_only": True,
        }

    with ThreadPoolExecutor(max_workers=budget["maximum_workers"]) as pool:
        futures = {pool.submit(detail, r): r for r in selected.values()}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {"code": futures[future]["code"], "error": type(exc).__name__ + ": " + str(exc)}
                )
    results.sort(key=lambda r: r["code"])
    save(output / "details.json", results)
    save(
        output / "summary.json",
        {
            "schema": "matched-carry-identity-source-capture.v1",
            "symbols": len(first),
            "catalog_pages": len(traversals),
            "catalog_exhausted": exhausted,
            "details": len(results),
            "detail_errors": sum("error" in r for r in results),
            "missing_market_sources": missing,
            "all_identity_mappings_approved": False,
            "financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "financial_accounts": 0,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.plan, args.output)
