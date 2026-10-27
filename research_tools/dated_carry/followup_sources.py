"""Follow an observed PDF link and capture two original public API references."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def capture(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(parent)
    prior = read(parent / "summary.json")
    assert prior["http_requests"] == 13
    page = parent / prior["final_source_responses"]["paper"]["body_file"]
    paper = "https://www.bis.org/publications/working-paper-1087-crypto-carry.pdf"
    assert paper in page.read_text()
    tasks = {
        "paper": paper,
        "exchange_info": "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "official_market_sdk": "https://raw.githubusercontent.com/binance/binance-futures-connector-python/main/binance/um_futures/market.py",
    }
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "parent_completion_sha256": sha(parent / "completion.json"),
            "source_sha256": sha(Path(__file__)),
            "targets": tasks,
            "remaining_original_http_budget": 3,
            "maximum_new_market_archives": 0,
            "new_financial_accounts": 0,
        },
    )
    rows = []
    for identity, url in tasks.items():
        r = requests.get(url, timeout=(10, 30), allow_redirects=False)
        body = r.content
        assert len(body) < 12000000
        p = output / (identity + ".raw")
        p.write_bytes(body)
        row = {
            "id": identity,
            "url": url,
            "status": r.status_code,
            "content_type": r.headers.get("content-type"),
            "body_sha256": sha(p),
            "bytes": len(body),
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "pdf_header": body.startswith(b"%PDF-") if identity == "paper" else None,
        }
        save(output / (identity + ".receipt.json"), row)
        rows.append(row)
        print({k: row[k] for k in ["id", "status", "bytes", "pdf_header"]}, flush=True)
    save(
        output / "summary.json",
        {
            "responses": rows,
            "combined_http_requests": 16,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json",
        {"capture_finished": True, "manifest": manifest, "profit_goal_met": False},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    capture(a.parent, a.output)
