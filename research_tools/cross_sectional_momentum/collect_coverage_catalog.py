"""Bounded current provider identity/coverage metadata, no prices or returns."""

import argparse
import json
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("coverage_catalog_plan.json")
    plan = read(path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in [path, Path(__file__)]},
            "before_new_provider_catalog_access": True,
            "economic_evaluation_allowed": False,
        },
    )
    assert len(plan["requests"]) <= plan["budget"]["maximum_requests"]
    results = []
    for item in plan["requests"]:
        name = item["name"]
        receipt = {
            "url": plan["origin"] + item["path"],
            "params": item["params"],
            "status": None,
            "sent_wall_ns": time.time_ns(),
        }
        data = None
        try:
            with requests.get(
                receipt["url"],
                params=item["params"],
                timeout=(10, 20),
                stream=True,
                allow_redirects=False,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                receipt.update(
                    status=response.status_code,
                    url=response.url,
                    headers={
                        k: response.headers[k]
                        for k in ["Date", "Age", "Content-Type", "ETag", "Last-Modified"]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_bytes_each"] + 1, decode_content=False
                )
            assert len(body) <= plan["budget"]["maximum_bytes_each"]
            (output / (name + ".body")).write_bytes(body)
            receipt.update(bytes=len(body), body_sha256=sha(output / (name + ".body")))
            data = json.loads(body)
            save(output / (name + ".json"), data)
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:200])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (name + ".receipt.json"), receipt)
        result = {
            "name": name,
            "status": receipt["status"],
            "error_type": receipt.get("error_type"),
            "records": len(data.get("data", [])) if isinstance(data, dict) else None,
            "next_page_present": bool(data.get("next_page_url"))
            if isinstance(data, dict)
            else None,
        }
        results.append(result)
        print(result, flush=True)
    save(
        output / "summary.json",
        {
            "results": results,
            "new_financial_accounts": 0,
            "historical_identities_certified": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output)
