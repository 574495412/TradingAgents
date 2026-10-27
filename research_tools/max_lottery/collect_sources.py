"""Bounded primary-paper capture; no profitability inference or financial accounts."""

import argparse
import json
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("source_plan.json")
    plan = read(plan_path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {
                str(p): sha(p) for p in [plan_path, Path(__file__), Path(plan["reuse"])]
            },
            "before_new_native_downloads": True,
            "before_economic_registration_or_evaluation": True,
        },
    )
    assert len(plan["requests"]) <= plan["budget"]["maximum_http_requests"]
    rows = []
    for item in plan["requests"]:
        receipt = {**item, "sent_wall_ns": time.time_ns(), "status": None}
        try:
            with requests.get(
                item["url"],
                timeout=(10, 25),
                allow_redirects=False,
                stream=True,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                receipt.update(
                    status=response.status_code,
                    response_url=response.url,
                    headers={
                        k: response.headers[k]
                        for k in ["Content-Type", "Date", "Location", "Last-Modified", "ETag"]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_bytes_each"] + 1, decode_content=False
                )
            if len(body) > plan["budget"]["maximum_bytes_each"]:
                raise ValueError("Source exceeds registered byte cap")
            path = output / (item["id"] + ".body")
            path.write_bytes(body)
            receipt.update(bytes=len(body), sha256=sha(path), pdf_magic=body.startswith(b"%PDF"))
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:200])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (item["id"] + ".receipt.json"), receipt)
        rows.append(
            {k: receipt.get(k) for k in ["id", "status", "bytes", "pdf_magic", "error_type"]}
        )
        print(json.dumps(rows[-1]), flush=True)
    save(
        output / "summary.json",
        {"results": rows, "new_financial_accounts": 0, "economic_evaluation_allowed": False},
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output)
