"""Two bounded public reads: documented Community access and one historical row."""

import argparse
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("public_cap_probe_plan_v002.json")
    plan = read(path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in [path, Path(__file__)]},
            "before_new_historical_observation": True,
        },
    )
    results = []
    for name, url, params in [
        ("documentation", plan["documentation"], None),
        ("historical", plan["provider_endpoint"], plan["params"]),
    ]:
        receipt = {"url": url, "sent_wall_ns": time.time_ns(), "status": None}
        data = None
        try:
            with requests.get(
                url,
                params=params,
                timeout=(10, 15),
                stream=True,
                allow_redirects=False,
                headers={"Accept-Encoding": "identity"},
            ) as r:
                receipt.update(
                    status=r.status_code,
                    url=r.url,
                    headers={
                        k: r.headers[k]
                        for k in [
                            "Date",
                            "Age",
                            "Content-Type",
                            "Content-Encoding",
                            "ETag",
                            "Last-Modified",
                        ]
                        if k in r.headers
                    },
                )
                body = r.raw.read(plan["budget"]["maximum_bytes_each"] + 1, decode_content=False)
            assert len(body) <= plan["budget"]["maximum_bytes_each"]
            (output / (name + ".body")).write_bytes(body)
            receipt.update(bytes=len(body), sha256=sha(output / (name + ".body")))
            if name == "documentation":
                assert b"community-api.coinmetrics.io" in body
            elif receipt["status"] == 200:
                import json

                data = json.loads(body)
                save(output / "historical.json", data)
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:180])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (name + ".receipt.json"), receipt)
        results.append(
            {
                "name": name,
                "status": receipt["status"],
                "error_type": receipt.get("error_type"),
                "rows": len(data.get("data", [])) if isinstance(data, dict) else None,
            }
        )
        print(results[-1], flush=True)
        if name == "documentation" and (receipt["status"] != 200 or receipt.get("error_type")):
            break
    save(
        output / "summary.json",
        {
            "results": results,
            "new_financial_accounts": 0,
            "original_author_source_replication": False,
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    probe(p.parse_args().output)
