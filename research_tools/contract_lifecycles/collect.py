"""Bounded official CMS capture; title matches remain candidates, not tradability proof."""

import argparse
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def text_nodes(node):
    if isinstance(node, list):
        return "".join(text_nodes(v) for v in node)
    if isinstance(node, dict):
        if node.get("node") == "text":
            return html.unescape(node.get("text", ""))
        text = "".join(text_nodes(v) for k, v in node.items() if k in ["child", "children"])
        return text + ("\n" if node.get("tag") in ["p", "li", "tr", "h1", "h2", "h3"] else "")
    return ""


def request(root, stem, endpoint, params):
    raw, receipt = root / (stem + ".raw"), root / (stem + ".receipt.json")
    if raw.exists():
        r = read(receipt)
        prepared = requests.Request("GET", endpoint, params=params).prepare().url
        if r["body_sha256"] != sha(raw) or r["url"] != prepared or r["status"] != 200:
            raise ValueError("Existing response identity differs")
    else:
        r = requests.get(endpoint, params=params, timeout=25)
        raw.write_bytes(r.content)
        save(
            receipt,
            {
                "url": r.url,
                "status": r.status_code,
                "body_sha256": sha(raw),
                "captured_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        r.raise_for_status()
    result = json.loads(raw.read_text())
    if not result.get("success") or result.get("code") != "000000":
        raise ValueError("CMS failure response")
    return result["data"]


def collect(root):
    p = Path(__file__).with_name("plan.json")
    plan, registered = read(p), read(root / "plan.json")
    if sha(p) != registered["registered_plan_sha256"]:
        raise ValueError("Registered study changed")
    source = root / "collector_source.py"
    source.write_bytes(Path(__file__).read_bytes()) if not source.exists() else None
    if sha(source) != sha(Path(__file__)):
        raise ValueError("Capture implementation changed")
    save(
        root / "collection_registration.json",
        {
            "collector_sha256": sha(source),
            "plan_sha256": sha(root / "plan.json"),
            "before_catalog_traversal": True,
            "first_delisting_page_already_captured_and_adopted": True,
            "detail_priority": "priority ticker + futures first; other futures second; priority ticker third; remaining delist/rebrand titles last; within rank releaseDate then code ascending. Cap180. Full selected code set frozen before detail calls; no returns inspected.",
        },
    )
    articles, traversals = {}, []
    requests_count = 0
    for cat in plan["catalogs"]:
        done = False
        for page in range(1, cat["maximum_pages"] + 1):
            requests_count += 1
            if requests_count > plan["maximum_list_requests"]:
                raise ValueError("List budget exceeded")
            data = request(
                root,
                f"list_{cat['id']}_{page:03d}",
                plan["list_endpoint"],
                {"type": 1, "pageNo": page, "pageSize": plan["page_size"], "catalogId": cat["id"]},
            )
            catalog = next(c for c in data["catalogs"] if c["catalogId"] == cat["id"])
            rows = catalog["articles"]
            for r in rows:
                a = articles.setdefault(r["code"], {**r, "catalog_ids": []})
                if cat["id"] not in a["catalog_ids"]:
                    a["catalog_ids"].append(cat["id"])
            if len(rows) < plan["page_size"] or page * plan["page_size"] >= catalog["total"]:
                done, reason = True, "catalog_exhausted_at_capture"
            elif rows and min(r["releaseDate"] for r in rows) < 1567296000000:
                done, reason = True, "older_than_2019_09_01"
            if done:
                break
        traversals.append(
            {
                "catalog_id": cat["id"],
                "pages": page,
                "captured_total": catalog["total"],
                "done": done,
                "reason": reason if done else "page_budget_reached",
            }
        )
        print(traversals[-1], flush=True)
    save(
        root / "catalog.json",
        {
            "articles": list(articles.values()),
            "traversals": traversals,
            "list_requests": requests_count,
        },
    )
    bases = [s.removesuffix("USDT") for s in registered["priority_symbols"]]
    ticker = re.compile(
        r"(?<![A-Z0-9])(?:" + "|".join(re.escape(s) for s in bases) + r")(?:USDT)?(?![A-Z0-9])"
    )
    candidates = []
    for a in articles.values():
        title = a["title"]
        priority, futures = (
            bool(ticker.search(title.upper())),
            bool(re.search(r"futures|perpetual", title, re.I)),
        )
        if not (futures or priority or re.search(r"delist|rebrand|token swap", title, re.I)):
            continue
        rank = 0 if priority and futures else 1 if futures else 2 if priority else 3
        candidates.append({**a, "selection_rank": rank, "priority_ticker_in_title": priority})
    ordered = sorted(candidates, key=lambda a: (a["selection_rank"], a["releaseDate"], a["code"]))
    selected = ordered[: plan["maximum_detail_requests"]]
    save(
        root / "detail_selection.json",
        {
            "articles": selected,
            "candidates": len(ordered),
            "omitted_candidates": ordered[len(selected) :],
            "selected_before_detail_requests": True,
            "title_scope_is_not_full_historical_event_certification": True,
        },
    )
    results = []
    (root / "details").mkdir()
    for i, a in enumerate(selected):
        code = a["code"]
        data = request(root / "details", code, plan["detail_endpoint"], {"articleCode": code})
        requests_count += 1
        if requests_count > plan["maximum_total_requests"]:
            raise ValueError("Total request budget exceeded")
        if data["code"] != code:
            raise ValueError("CMS returned another article")
        body = json.loads(data["body"])
        text = text_nodes(body)
        (root / "details" / (code + ".txt")).write_text(text)
        hits = [
            s
            for s in registered["selected_symbols"]
            if re.search(r"(?<![A-Z0-9])" + re.escape(s) + r"(?![A-Z0-9])", text)
        ]
        results.append(
            {
                "code": code,
                "title": data["title"],
                "publish_at_claimed": data["publishDate"],
                "version": data.get("version"),
                "lastUpdateTime": data.get("lastUpdateTime"),
                "body_sha256": sha(root / "details" / (code + ".raw")),
                "text_sha256": sha(root / "details" / (code + ".txt")),
                "selected_symbol_mentions": hits,
                "original_historical_version_certified": False,
                "notice_scope_and_times_manually_verified": False,
            }
        )
        if (i + 1) % 20 == 0:
            print(
                {
                    "details": i + 1,
                    "mentions": sum(bool(r["selected_symbol_mentions"]) for r in results),
                },
                flush=True,
            )
    save(
        root / "summary.json",
        {
            "schema": "historical-contract-cms-capture.v1",
            "selected_symbols": len(registered["selected_symbols"]),
            "priority_symbols": registered["priority_symbols"],
            "catalog_articles": len(articles),
            "traversals": traversals,
            "details": len(results),
            "list_requests": requests_count - len(results),
            "total_requests": requests_count,
            "results": results,
            "unfetched_title_candidates": len(ordered) - len(selected),
            "historical_notice_completeness_certified": False,
            "all_contract_segments_certified": False,
            "financial_accounts": 0,
            "profit_goal_met": False,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    save(
        root / "completion.json",
        {
            "manifest": {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
            "capture_complete": True,
            "historical_event_coverage_certified": False,
            "financial_accounts": 0,
        },
    )
    print(
        {"capture_complete": True, "details": len(results), "requests": requests_count}, flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
