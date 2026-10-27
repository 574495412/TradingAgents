"""Resolve a failed API route from official Postman source, not guessed payouts."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def repair(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(parent)
    assert all(r["status"] == 404 for r in read(parent / "summary.json")["settlement_responses"])
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "reason": "Correct an incorrectly constructed404 route using official source; preserve original raw errors and all six market archives. No new markets, prices, account policies or economic trials.",
            "parent_completion_sha256": sha(parent / "completion.json"),
            "source_sha256": sha(Path(__file__)),
            "maximum_read_only_http_requests": 4,
            "new_market_archives": 0,
            "new_financial_accounts": 0,
        },
    )
    session = requests.Session()
    requests_made = 0

    def get(url, filename):
        nonlocal requests_made
        assert requests_made < 4
        requests_made += 1
        r = session.get(url, timeout=(10, 30), allow_redirects=False)
        (output / filename).write_bytes(r.content)
        save(
            output / (filename + ".receipt.json"),
            {
                "url": url,
                "status": r.status_code,
                "bytes": len(r.content),
                "sha256": sha(output / filename),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        assert r.status_code == 200
        return r.json()

    files = get(
        "https://api.github.com/repos/binance/binance-api-postman/contents/collections",
        "official_directory.json",
    )
    candidates = [f for f in files if "usd" in f["name"].lower() and "futures" in f["name"].lower()]
    assert len(candidates) == 1, [f["name"] for f in files]
    collection = get(candidates[0]["download_url"], "official_um_collection.json")
    routes = []

    def visit(node):
        if isinstance(node, dict):
            request = node.get("request", {})
            raw = request.get("url", {})
            raw = raw.get("raw", "") if isinstance(raw, dict) else raw
            if "delivery-price" in raw:
                assert request["method"] == "GET"
                routes.append(
                    {
                        "name": node.get("name"),
                        "raw": raw,
                        "description": request.get("description"),
                    }
                )
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(collection)
    assert len(routes) == 1, routes
    route = routes[0]["raw"]
    assert "/futures/data/delivery-price" in route
    save(
        output / "official_route.json",
        {
            "collection_blob_sha": candidates[0]["sha"],
            "request": routes[0],
            "base_url": "https://fapi.binance.com",
        },
    )
    responses = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        path = urlsplit(
            route.replace("{{url}}", "https://fapi.binance.com").replace(
                "{{baseUrl}}", "https://fapi.binance.com"
            )
        ).path
        assert path == "/futures/data/delivery-price"
        data = get(
            "https://fapi.binance.com" + path + "?" + urlencode({"pair": pair}),
            pair + "_delivery.json",
        )
        responses.append({"pair": pair, "records": data})
    session.close()
    save(
        output / "summary.json",
        {
            "requests": requests_made,
            "responses": responses,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json",
        {"capture_finished": True, "manifest": manifest, "profit_goal_met": False},
    )
    print({"requests": requests_made, "responses": responses}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repair(a.parent, a.output)
