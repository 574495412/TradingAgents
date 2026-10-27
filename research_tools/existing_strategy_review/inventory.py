"""Inventory existing financial evidence; no strategy import or trading execution."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

PROJECTS = {
    "CZ-Agent": Path("/Users/chrise/GitHub/CZ-Agent"),
    "TradingAgents": Path("/Users/chrise/GitHub/TradingAgents"),
    "Freqtrade": Path("/Users/chrise/GitHub/freqtrade"),
}
PRUNE = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "runtime",
    "engine_image",
    "data_image",
    "market_data",
    "lossless_cache",
    "source",
    "sources",
    "source_snapshot",
    "source_snapshots",
    "immutable",
    "legacy_source",
    "fork_workspace",
    "external_sources_20261022",
    "native_absorption_panel_v001",
    "__pycache__",
    "logs",
    "plot",
    "plots",
    "data",
    "freqaimodels",
    "hyperopt_results",
    "hyperopts",
    "strategies",
    "vendor",
}
MATCH = re.compile(
    r"(result|summary|report|account|performance|metrics|selection|audit|comparison|validation|completion|plan)",
    re.I,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(root):
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in PRUNE
            and not d.startswith(".")
            and not Path(current, d).is_symlink()
            and not d.endswith(".git")
        )
        for name in sorted(files):
            if (
                (
                    name.endswith(".json")
                    and MATCH.search(name)
                    and name not in ["settings.json", "config.json", "exampleconfig_secret.json"]
                )
                or name.startswith("backtest-result-")
                and name.endswith(".zip")
            ):
                yield Path(current, name)


def inventory(output):
    output.mkdir(parents=True, exist_ok=False)
    roots = [
        ("CZ-Agent", PROJECTS["CZ-Agent"] / "artifacts"),
        ("TradingAgents", PROJECTS["TradingAgents"] / "results/research"),
        ("TradingAgents", PROJECTS["TradingAgents"] / "data/research"),
        ("Freqtrade", PROJECTS["Freqtrade"] / "user_data/crypto_quant"),
    ]
    aliases = []
    for p in (PROJECTS["CZ-Agent"] / "artifacts").iterdir():
        if p.is_symlink():
            aliases.append({"alias": str(p), "canonical": str(p.resolve()), "exists": p.exists()})
    files = {}
    errors = []
    for owner, root in roots:
        if not root.is_dir():
            continue
        for path in walk(root):
            resolved = path.resolve()
            if resolved in files:
                continue
            size = path.stat().st_size
            row = {
                "owner": owner,
                "path": str(resolved),
                "relative_path": str(resolved.relative_to(PROJECTS[owner])),
                "bytes": size,
                "type": "native_zip" if path.suffix == ".zip" else "json",
                "sha256": None,
            }
            if row["type"] == "json" and size <= 64 * 1024 * 1024:
                try:
                    raw = path.read_bytes()
                    row["sha256"] = hashlib.sha256(raw).hexdigest()
                    v = json.loads(raw)
                    row["shape"] = (
                        list(v)[:35]
                        if isinstance(v, dict)
                        else "list:" + str(len(v))
                        if isinstance(v, list)
                        else type(v).__name__
                    )
                    row["schema"] = v.get("schema") if isinstance(v, dict) else None
                    row["has_account_markers"] = isinstance(v, dict) and any(
                        k in v
                        for k in [
                            "net_return",
                            "profit_total",
                            "strategy",
                            "accounts",
                            "account",
                            "holdout",
                            "scenarios",
                            "metrics",
                        ]
                    )
                except (OSError, ValueError) as exc:
                    errors.append({"path": str(path), "error": str(exc)[:180]})
            elif row["type"] == "json":
                row["excluded_reason"] = "JSON exceeds64MiB; review separately"
            files[resolved] = row
    value = {
        "schema": "existing-strategy-evidence-inventory.v1",
        "roots": [{"owner": owner, "path": str(root)} for owner, root in roots],
        "aliases": aliases,
        "files": list(files.values()),
        "errors": errors,
        "pruned_directories": sorted(PRUNE),
        "native_archives_executed": False,
        "orders_enabled": False,
    }
    (output / "inventory.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "files": len(files),
                "by_owner": Counter(x["owner"] for x in files.values()),
                "native_zips": sum(x["type"] == "native_zip" for x in files.values()),
                "schemas": Counter(x.get("schema") for x in files.values()).most_common(14),
                "account_marker_files": sum(
                    x.get("has_account_markers", False) for x in files.values()
                ),
                "aliases": len(aliases),
                "errors": len(errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    inventory(p.parse_args().output)
