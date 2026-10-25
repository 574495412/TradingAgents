#!/usr/bin/env python3
"""One-command frozen trend research, native data reuse and independent review.

This standalone coordinator imports no application package, reads no .env,
controls no existing process and has no account/trading endpoint.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(name, freqtrade_root):
    if not re.fullmatch(r"trend_mainline_[A-Za-z0-9_]{1,70}", name):
        raise ValueError(
            "Use a fresh name starting trend_mainline_ with letters/digits/underscores"
        )
    fq = freqtrade_root.resolve()
    python = fq / ".venv/bin/python"
    source = fq / "user_data/crypto_quant" / name
    output = ROOT / "results/research" / (name + "_audit")
    receipt = ROOT / "results/research" / (name + "_workflow")
    review = ROOT / "results/research" / (name + "_review")
    if any(p.exists() for p in (source, output, receipt, review)):
        raise ValueError("Existing or failed research must be preserved; choose a fresh name")
    if not python.is_file():
        raise ValueError("Freqtrade research interpreter is unavailable")
    receipt.mkdir(parents=True)
    commands = [
        (
            "register",
            [str(python), "-m", "research.crypto.trend_mainline", "register", "--run", str(source)],
        ),
        (
            "prepare",
            [str(python), "-m", "research.crypto.trend_mainline", "prepare", "--run", str(source)],
        ),
        (
            "accounts",
            [str(python), "-m", "research.crypto.trend_mainline", "run", "--run", str(source)],
        ),
        (
            "audit",
            [
                str(python),
                str(ROOT / "tradingagents/research/quant/trend_mainline_audit.py"),
                "--source",
                str(source),
                "--output",
                str(output),
            ],
        ),
    ]
    commands.append(
        (
            "review",
            [
                str(python),
                str(ROOT / "tradingagents/research/quant/trend_mainline_review.py"),
                "--source",
                str(source),
                "--audit",
                str(output),
                "--output",
                str(review),
            ],
        )
    )
    for stage, command in commands:
        print("Running " + stage + "; log: " + str(receipt / (stage + ".log")), flush=True)
        with (receipt / (stage + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=fq, stdout=log, stderr=log, check=False)
        with (receipt / (stage + ".json")).open("x") as stream:
            json.dump(
                {
                    "stage": stage,
                    "returncode": result.returncode,
                    "command": command,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - macOS Python3.9 entry
                },
                stream,
                indent=2,
            )
        if result.returncode:
            raise RuntimeError("Research stage failed; preserve evidence: " + stage)
    summary = json.loads((source / "evaluation.json").read_text())
    print(
        json.dumps(
            {
                "source": str(source),
                "review": str(output),
                "selection": summary["selected_candidate"],
                "effective_return_source": summary["valid_return_source_established"],
                "production_eligible": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--freqtrade-root", type=Path, default=ROOT.parent / "freqtrade")
    a = p.parse_args()
    try:
        run(a.name, a.freqtrade_root)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))
