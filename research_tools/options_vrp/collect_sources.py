"""Pin one author's repository and sample native public option facts, no strategy."""

import argparse
import json
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("source_plan.json")
    plan = read(path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in [path, Path(__file__)]},
            "before_new_native_access": True,
            "economic_evaluation_allowed": False,
        },
    )
    rows = []

    def capture(name, url, params=None):
        if len(rows) >= plan["budget"]["maximum_native_HTTP_requests"]:
            raise ValueError("Registered request cap reached")
        receipt = {
            "id": name,
            "url": url,
            "params": params,
            "sent_wall_ns": time.time_ns(),
            "status": None,
        }
        decoded = None
        try:
            with requests.get(
                url,
                params=params,
                timeout=(10, 25),
                stream=True,
                allow_redirects=False,
                headers={
                    "Accept-Encoding": "identity",
                    "Accept": "application/json,application/pdf,text/html,text/plain",
                },
            ) as response:
                receipt.update(
                    status=response.status_code,
                    response_url=response.url,
                    headers={
                        k: response.headers[k]
                        for k in [
                            "Content-Type",
                            "Date",
                            "ETag",
                            "Last-Modified",
                            "Location",
                            "X-RateLimit-Remaining",
                        ]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_bytes_each"] + 1, decode_content=False
                )
            if len(body) > plan["budget"]["maximum_bytes_each"]:
                raise ValueError("Raw response exceeds registered byte cap")
            file = output / (name + ".body")
            file.write_bytes(body)
            receipt.update(bytes=len(body), sha256=sha(file), pdf_magic=body.startswith(b"%PDF"))
            try:
                decoded = json.loads(body)
                save(output / (name + ".json"), decoded)
                receipt["json_rpc_error"] = (
                    decoded.get("error") if isinstance(decoded, dict) else None
                )
            except (ValueError, UnicodeError):
                pass
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:180])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (name + ".receipt.json"), receipt)
        rows.append(
            {
                k: receipt.get(k)
                for k in ["id", "status", "bytes", "pdf_magic", "json_rpc_error", "error_type"]
            }
        )
        print(rows[-1], flush=True)
        return decoded

    repo = plan["author_repository"]
    commit = capture(
        "author_commit", "https://api.github.com/repos/" + repo + "/commits", {"per_page": 1}
    )
    if isinstance(commit, list) and len(commit) == 1 and len(commit[0].get("sha", "")) == 40:
        revision = commit[0]["sha"]
        tree = capture(
            "author_tree",
            "https://api.github.com/repos/" + repo + "/git/trees/" + revision,
            {"recursive": 1},
        )
        if isinstance(tree, dict) and not tree.get("truncated", True):
            available = {r["path"] for r in tree.get("tree", []) if r["type"] == "blob"}
            selected = [p for p in plan["repository_files_max"] if p in available]
            save(
                output / "author_pin.json",
                {
                    "repository": repo,
                    "commit": revision,
                    "selected_files": selected,
                    "tree_truncated": False,
                    "raw_private_inputs_obtained": False,
                },
            )
            for n, name in enumerate(selected):
                capture(
                    "author_file_" + str(n),
                    "https://raw.githubusercontent.com/" + repo + "/" + revision + "/" + name,
                )
    capture("paper_v2", plan["paper"]["url"])
    for item in plan["document_sources"]:
        capture(item["id"], item["url"])
    for item in plan["public_read_only_market_requests"]:
        capture(item["id"], plan["production_public_root"] + item["path"], item["params"])
    save(
        output / "summary.json",
        {
            "results": rows,
            "native_HTTP_requests": len(rows),
            "new_financial_accounts": 0,
            "original_author_replication": False,
            "economic_evaluation_allowed": False,
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
