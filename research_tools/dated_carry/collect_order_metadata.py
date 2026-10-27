"""Bounded public history search for exact-contract original order filters."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).with_name("order_metadata_plan.json")
    plan = read(src)
    parent = Path(plan["parent"])
    verify_manifest(parent)
    save(
        output / "plan.json",
        {
            **plan,
            "source_hashes": {
                str(p): sha(p) for p in [src, Path(__file__), parent / "completion.json"]
            },
        },
    )
    save(
        output / "registration.json",
        {
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha(output / "plan.json"),
            "before_historical_fixture_access": True,
            "new_financial_accounts": 0,
        },
    )
    session, receipts, fixtures = requests.Session(), [], []

    def get(url, stem, params=None):
        assert len(receipts) < plan["budget"]["read_only_HTTP_requests_max"]
        receipt = {"url": url, "status": None, "requested_at_utc": datetime.now(UTC).isoformat()}
        decoded = None
        try:
            with session.get(
                url, params=params, timeout=(10, 25), stream=True, allow_redirects=False
            ) as response:
                receipt.update(url=response.url, status=response.status_code)
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    assert size <= plan["budget"]["response_bytes_max"]
                    chunks.append(chunk)
                body = b"".join(chunks)
            file = output / (stem + ".json")
            file.write_bytes(body)
            receipt.update(bytes=size, sha256=sha(file))
            if receipt["status"] == 200:
                decoded = read(file)
        except Exception as e:
            receipt.update(error_type=type(e).__name__, error=str(e))
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        save(output / (stem + ".receipt.json"), receipt)
        receipts.append(receipt)
        print(
            {
                "stem": stem,
                "status": receipt["status"],
                "bytes": receipt.get("bytes"),
                "error": receipt.get("error"),
            },
            flush=True,
        )
        return decoded

    base = "https://api.github.com/repos/" + plan["repository"]
    listing = get(base + "/contents/" + plan["initial_directory"], "static_listing")
    markets = next(
        (
            item
            for item in listing or []
            if item.get("name") == "markets" and item.get("type") == "dir"
        ),
        None,
    )
    market_files = get(markets["url"], "markets_listing") if markets else []
    for filename in plan["target_market_files"]:
        selected = next(
            (
                item
                for item in market_files or []
                if item.get("name") == filename and item.get("type") == "file"
            ),
            None,
        )
        if not selected:
            fixtures.append(
                {"name": filename, "found": False, "reason": "No path in discovered listing"}
            )
            continue
        commits = get(
            base + "/commits",
            filename + "_commits",
            {"path": selected["path"], "until": plan["cutoff"], "per_page": 1},
        )
        if not commits:
            fixtures.append(
                {"name": filename, "found": False, "reason": "No pre-entry commit obtained"}
            )
            continue
        commit = commits[0]
        commit_sha = commit["sha"]
        body = get(
            "https://raw.githubusercontent.com/"
            + plan["repository"]
            + "/"
            + commit_sha
            + "/"
            + selected["path"],
            filename + "_historical",
        )
        text = str(body) if body is not None else ""
        fixtures.append(
            {
                "name": filename,
                "path": selected["path"],
                "commit": commit_sha,
                "commit_date": commit["commit"]["committer"]["date"],
                "body_available": body is not None,
                "exact_contract_strings_present": {s: s in text for s in plan["contracts"]},
                "historical_order_filters_certified": False,
            }
        )
    session.close()
    save(
        output / "summary.json",
        {
            "receipts": receipts,
            "fixtures": fixtures,
            "new_financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "capture_finished": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
