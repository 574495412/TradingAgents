"""Seal primary cointegration references; downloaded programs are never executed."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from research_tools.factor_lab.io import save, sha

FILES = {
    "stefan-jansen/machine-learning-for-trading": {
        "commit": "30db4f00946ce91fbc98fef18582b8cbc3fd8afd",
        "license": "MIT",
        "files": ["LICENSE", "09_model_based_features/14_panel_features.py"],
    },
    "je-suis-tm/quant-trading": {
        "commit": "611b73f2c3f577ac5b28aaa19ac8c43d3236c7a5",
        "license": "Apache-2.0",
        "files": ["LICENSE", "Pair trading backtest.py"],
    },
    "statsmodels/statsmodels": {
        "commit": "1107ea567121b80f90562d6c72085eed7e882113",
        "license": "BSD-3-Clause",
        "files": [
            "LICENSE.txt",
            "statsmodels/tsa/stattools.py",
            "statsmodels/tsa/vector_ar/vecm.py",
        ],
    },
}


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
    for repository, spec in FILES.items():
        for filename in spec["files"]:
            url = f"https://raw.githubusercontent.com/{repository}/{spec['commit']}/{filename}"
            r = session.get(url, timeout=30)
            r.raise_for_status()
            path = root / repository.replace("/", "__") / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            records.append(
                {
                    "repository": repository,
                    "commit": spec["commit"],
                    "license": spec["license"],
                    "file": filename,
                    "url": r.url,
                    "sha256": sha(path),
                    "local": str(path.resolve()),
                }
            )
    save(
        root / "source_review.json",
        {
            "schema": "pairs-primary-source-review.v1",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "records": records,
            "downloaded_programs_executed": False,
            "profit_proven": False,
            "method_links": [
                "https://arxiv.org/abs/2109.10662",
                "https://arxiv.org/abs/2305.06961",
            ],
            "collection_note": "Documentation host timed out; preserve incomplete v001/v002. Versioned implementation and docstrings from statsmodels official GitHub replace unavailable HTML; paper links were reviewed via web, not archived full papers.",
            "review": [
                "Jansen shows Engle-Granger/Johansen, training-block half-life, forward-only hedge filtering and entry threshold2; ETF method example, not crypto profit evidence.",
                "je-suis-tm uses full-sample price maxima to choose shares and an unadjusted residual ADF; do not import those sizing/test assumptions.",
                "Use proper cointegration p-values, declared multiple-test correction, train-only coefficients, matched pair legs, actual funding and separate closed pair lifecycles.",
                "ArXiv papers support the hypothesis and execution sensitivity; source claims and venues do not prove this Binance experiment profitable.",
            ],
        },
    )
    save(
        root / "completion.json",
        {"source_review_sha256": sha(root / "source_review.json"), "files": len(records)},
    )
    print({"source_files": len(records), "root": str(root)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output.resolve())
