"""Archive pinned public references without executing third-party programs."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from research_tools.factor_lab.io import read, save, sha

REPOS = {
    "hmmlearn/hmmlearn": {
        "ref": "0.3.3",
        "files": ["LICENSE.txt", "src/hmmlearn/base.py", "src/hmmlearn/hmm.py"],
        "use": "GaussianHMM training reference; our inference must filter past observations only",
    },
    "scikit-learn/scikit-learn": {
        "ref": "1.7.2",
        "files": ["COPYING", "sklearn/neural_network/_multilayer_perceptron.py"],
        "use": "Single fixed MLP baseline matching the isolated installed research dependency",
    },
}
DOCS = [
    "https://hmmlearn.readthedocs.io/en/0.3.3/api.html",
    "https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPClassifier.html",
    "https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/cross_validation.html",
]


def collect(root, old_pack):
    root.mkdir(parents=True, exist_ok=False)
    records = []
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
    headers = {
        "User-Agent": "TradingAgents-research-source-audit",
        "Accept": "application/vnd.github+json",
    }
    for repo, spec in REPOS.items():
        response = session.get(
            f"https://api.github.com/repos/{repo}/commits/{spec['ref']}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        commit = response.json()["sha"]
        folder = root / repo.replace("/", "__")
        folder.mkdir()
        save(folder / "commit.json", response.json())
        for name in spec["files"]:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{name}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            records.append(
                {
                    "repository": repo,
                    "ref": spec["ref"],
                    "commit": commit,
                    "file": name,
                    "url": url,
                    "local": str(path.resolve()),
                    "sha256": sha(path),
                    "use": spec["use"],
                }
            )
    for number, url in enumerate(DOCS):
        response = session.get(url, timeout=30)
        response.raise_for_status()
        path = root / f"method_{number}.html"
        path.write_bytes(response.content)
        records.append(
            {
                "url": url,
                "local": str(path.resolve()),
                "sha256": sha(path),
                "use": "method documentation, captured snapshot rather than fixed source commit",
            }
        )
    adopted = {
        "systems/provided/rules/ewmac.py",
        "systems/provided/rules/breakout.py",
        "systems/provided/rules/mr_wings.py",
        "scripts/bt_longbinance.py",
        "src/sentinel/strategy/funding_carry.py",
    }
    for record in read(old_pack / "file_receipts.json"):
        if record["file"] not in adopted:
            continue
        path = Path(record["local_path"])
        if sha(path) != record["sha256"]:
            raise ValueError("Previously pinned source changed")
        records.append(
            {
                **record,
                "local": str(path),
                "use": "reuse sealed mathematical source; no source execution",
            }
        )
    save(
        root / "source_catalog.json",
        {
            "schema": "regime-program-sources.v1",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "records": records,
            "third_party_code_executed": False,
            "profit_proven": False,
            "orders_enabled": False,
            "notes": [
                "HMM is a statistical model, not a neural network.",
                "Full-sequence Viterbi/MAP/posterior smoothing cannot determine past trading states.",
                "MLP early_stopping must be disabled; all scaling and labels mature strictly before each fit.",
                "PST GPL programs are references for independent mathematical adapters; not vendored or imported.",
                "Framework prediction capability is not proof of trading alpha.",
            ],
        },
    )
    save(
        root / "completion.json",
        {"catalog_sha256": sha(root / "source_catalog.json"), "files": len(records)},
    )
    print({"sources": len(records), "root": str(root)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-pack", type=Path, required=True)
    args = parser.parse_args()
    collect(args.output.resolve(), args.old_pack.resolve())
