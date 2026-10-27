"""Capture pinned public strategy reports without importing or running their code."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests

SOURCES = {
    "Apex-prim/strategy-audit": [
        "README.md",
        "LEDGER.csv",
        "CORPUS_RUN.json",
        "BASELINE.md",
        "CORPUS.md",
        "LICENSE",
    ],
    "CacheCarti/Crypto-Strategies": ["README.md", "index.json", "FINDINGS.md", "LICENSE"],
    "PeetCrypto/freqtrade-stuff": ["README.md"],
}


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "github-strategy-source-capture.v1",
        "registered_ns": time.time_ns(),
        "source_files": SOURCES,
        "maximum_requests": 11,
        "maximum_bytes_per_file": 10000000,
        "source_code_executed": False,
        "new_factors": 0,
        "orders_enabled": False,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    receipts = []
    for repo, files in SOURCES.items():
        lookup = subprocess.run(
            [
                "git",
                "-c",
                "credential.helper=",
                "ls-remote",
                "--symref",
                "https://github.com/" + repo + ".git",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=35,
        )
        if lookup.returncode:
            receipts.append(
                {"repository": repo, "git_status": lookup.returncode, "error": lookup.stderr[:500]}
            )
            continue
        head = next(
            line.split()[0]
            for line in lookup.stdout.splitlines()
            if line.endswith("\tHEAD") and not line.startswith("ref:")
        )
        directory = output / repo.replace("/", "__")
        directory.mkdir()
        for file in files:
            url = f"https://raw.githubusercontent.com/{repo}/{head}/{file}"
            row = {
                "repository": repo,
                "commit": head,
                "file": file,
                "url": url,
                "requested_ns": time.time_ns(),
            }
            try:
                with requests.get(url, timeout=(10, 25), stream=True) as response:
                    response.raise_for_status()
                    body = response.raw.read(
                        plan["maximum_bytes_per_file"] + 1, decode_content=True
                    )
                    if len(body) > plan["maximum_bytes_per_file"]:
                        raise ValueError("Oversize public source")
                    dest = directory / file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                    row.update(
                        status=response.status_code,
                        bytes=len(body),
                        sha256=hashlib.sha256(body).hexdigest(),
                        local_path=str(dest),
                    )
            except (requests.RequestException, ValueError) as exc:
                row["error"] = str(exc)[:500]
            receipts.append(row)
            print(json.dumps(row), flush=True)
    (output / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output)
