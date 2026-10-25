"""Pin source sizing/risk definitions; do not execute downloaded trading programs."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from research_tools.factor_lab.io import save, sha

COMMIT = "164328bd76d2937d3dd61491377704c488283bc2"
FILES = [
    "systems/positionsizing.py",
    "systems/portfolio.py",
    "systems/buffering.py",
    "systems/forecast_scale_cap.py",
    "sysdata/config/defaults.yaml",
    "LICENSE",
]


def collect(root):
    root.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
        ),
    )
    records = []
    for name in FILES:
        url = f"https://raw.githubusercontent.com/pst-group/pysystemtrade/{COMMIT}/{name}"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        records.append(
            {
                "repository": "pst-group/pysystemtrade",
                "commit": COMMIT,
                "file": name,
                "url": url,
                "sha256": sha(path),
                "local": str(path.resolve()),
                "license": "GPL-3.0; read for independent mathematical adaptation, not vendored",
            }
        )
    save(
        root / "source_review.json",
        {
            "schema": "forecast-sizing-source-review.v1",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "records": records,
            "third_party_trading_code_executed": False,
            "observations": [
                "Subsystem position is volatility scalar times forecast divided by average_absolute_forecast.",
                "Defaults: average_absolute_forecast10, forecast_cap20, percentage_vol_target16, instrument_div_multiplier1.",
                "Default forecast buffer uses10percent of average-volatility-scaled position and trades to nearest edge.",
                "Prior local direction-only inverse-volatility normalization does not retain forecast magnitude or absolute risk budget.",
                "This is a new source-inspired sizing hypothesis, not proof of crypto profit or full upstream system reproduction.",
            ],
        },
    )
    save(
        root / "completion.json",
        {"source_review_sha256": sha(root / "source_review.json"), "files": len(records)},
    )
    print({"root": str(root), "files": len(records)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output.resolve())
