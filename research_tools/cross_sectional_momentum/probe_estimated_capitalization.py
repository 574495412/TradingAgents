"""Test the separately documented circulating-supply metric once, without keys."""

import argparse
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import save, sha


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    reference = Path(
        "research_tools/cross_sectional_momentum/capitalization_definition_receipt.json"
    )
    assert "CapMrktEstUSD" in reference.read_text()
    plan = {
        "question": "Is the circulating-supply-derived CapMrktEstUSD available on the authorized no-key Community endpoint at the same historical sample? Never substitute current-supply cap for it.",
        "endpoint": "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
        "params": {
            "assets": "btc,eth",
            "metrics": "PriceUSD,SplyCur,CapMrktCurUSD,CapMrktEstUSD",
            "frequency": "1d",
            "start_time": "2025-01-07T00:00:00Z",
            "end_time": "2025-01-07T00:00:00Z",
            "page_size": 10,
        },
        "primary_methodology_receipt": str(reference),
        "methodology_sha256": sha(reference),
        "maximum_requests": 1,
        "retries": 0,
        "credentials_used": False,
        "new_financial_accounts": 0,
        "original_CoinGecko_replication": False,
    }
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "plan_sha256": sha(output / "plan.json"),
            "source_sha256": sha(Path(__file__)),
            "before_first_estimated_cap_request": True,
        },
    )
    receipt = {"sent_wall_ns": time.time_ns(), "status": None}
    with requests.get(
        plan["endpoint"],
        params=plan["params"],
        timeout=(10, 15),
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
        body = response.raw.read(8388609, decode_content=False)
    assert len(body) <= 8388608
    (output / "response.body").write_bytes(body)
    receipt.update(
        received_wall_ns=time.time_ns(), sha256=sha(output / "response.body"), bytes=len(body)
    )
    save(output / "response.receipt.json", receipt)
    import json

    try:
        data = json.loads(body)
    except ValueError:
        data = {}
    save(output / "response.json", data)
    valid = (
        receipt["status"] == 200
        and len(data.get("data", [])) == 2
        and all("CapMrktEstUSD" in r for r in data.get("data", []))
    )
    summary = {
        "estimated_capitalization_sample_available": valid,
        "status": receipt["status"],
        "rows": len(data.get("data", [])),
        "error": data.get("error"),
        "coverage_and_identity_panel_verified": False,
        "new_financial_accounts": 0,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    probe(p.parse_args().output)
