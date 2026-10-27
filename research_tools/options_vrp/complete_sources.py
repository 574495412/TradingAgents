"""Public Git transport fallback and one exact bounded dataset header prefix."""

import argparse
import re
import subprocess
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def complete(output):
    output.mkdir(exist_ok=False, parents=True)
    path = Path(__file__).with_name("source_completion_plan.json")
    plan = read(path)
    parent = Path(plan["parent"])
    assert "API rate limit exceeded" in read(parent / "author_commit.json")["message"]
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {
                str(p): sha(p) for p in [path, Path(__file__), parent / "completion.json"]
            },
            "before_new_source_access": True,
            "economic_evaluation_allowed": False,
        },
    )
    command = [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "ls-remote",
        plan["repository"],
        "HEAD",
    ]
    git = {"command": command, "sent_wall_ns": time.time_ns()}
    revision = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        git.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr[:1000])
        match = re.fullmatch(r"([0-9a-f]{40})\s+HEAD\s*", result.stdout)
        if result.returncode == 0 and match:
            revision = match.group(1)
    except Exception as error:
        git.update(error_type=type(error).__name__, error=str(error)[:180])
    git["received_wall_ns"] = time.time_ns()
    save(output / "git_receipt.json", git)
    rows = []

    def capture(name, url, prefix=False):
        receipt = {"url": url, "sent_wall_ns": time.time_ns(), "status": None}
        headers = {"Accept-Encoding": "identity"}
        if prefix:
            headers["Range"] = plan["csv_prefix"]["range"]
        cap = (
            plan["csv_prefix"]["maximum_saved_bytes"]
            if prefix
            else plan["budget"]["max_full_file_bytes"]
        )
        try:
            with requests.get(
                url, headers=headers, allow_redirects=False, timeout=(10, 25), stream=True
            ) as response:
                receipt.update(
                    status=response.status_code,
                    response_url=response.url,
                    headers={
                        k: response.headers[k]
                        for k in [
                            "Content-Type",
                            "Content-Length",
                            "Content-Range",
                            "Content-Encoding",
                            "Location",
                            "Date",
                            "ETag",
                        ]
                        if k in response.headers
                    },
                )
                body = response.raw.read(cap if prefix else cap + 1, decode_content=False)
            if not prefix and len(body) > cap:
                raise ValueError("Source file exceeds registered cap")
            file = output / (name + ".body")
            file.write_bytes(body)
            receipt.update(
                bytes=len(body),
                sha256=sha(file),
                complete_body_captured=not prefix,
                representation="bounded_first65536bytes_only_not_complete_dataset"
                if prefix
                else "full_response_body",
            )
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:180])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (name + ".receipt.json"), receipt)
        rows.append(
            {
                "id": name,
                "status": receipt["status"],
                "bytes": receipt.get("bytes"),
                "error_type": receipt.get("error_type"),
            }
        )
        print(rows[-1], flush=True)

    if revision:
        save(
            output / "author_pin.json",
            {
                "repository": plan["repository"],
                "commit": revision,
                "basis": "normal public Git HEAD read; not necessarily the exact paper-era code version",
                "private_raw_data_obtained": False,
            },
        )
        for i, name in enumerate(plan["known_files"]):
            capture(
                "author_file_" + str(i),
                "https://raw.githubusercontent.com/ratmir-miftachov/BTC-premia/"
                + revision
                + "/"
                + name,
            )
    capture("uwa_csv_prefix", plan["csv_prefix"]["url"], prefix=True)
    assert len(rows) <= plan["budget"]["native_HTTP_requests_max"]
    save(
        output / "summary.json",
        {
            "author_commit": revision,
            "git_read_commands": 1,
            "native_HTTP_requests": len(rows),
            "results": rows,
            "new_financial_accounts": 0,
            "economic_evaluation_allowed": False,
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
    complete(parser.parse_args().output)
