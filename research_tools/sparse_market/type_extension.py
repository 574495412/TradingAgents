"""Close a discovered type-scope gap before reranking, without retroactive labels."""

import argparse
import json
from pathlib import Path

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import source


def extend(parent, capture, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(parent)
    verify_manifest(capture)
    registry = read(parent / "registry.json")
    targets = [
        ("40c76b4deaa247f09774e5d1ee747cb8", ["TSLAUSDT"], "equity", "Tesla"),
        ("0217a404bf984c028ebf9f92e083052c", ["XPTUSDT", "XPDUSDT"], "commodity", "Platinum"),
    ]
    save(
        output / "registration.json",
        {
            "schema": "source-type-closure.v1",
            "source_sha256": sha(Path(__file__)),
            "parent_completion_sha256": sha(parent / "completion.json"),
            "capture_completion_sha256": sha(capture / "completion.json"),
            "maximum_new_requests": 2,
            "targets": targets,
            "reason": "Initial rerank introduces TSLA; proactively cover all12equity/commodity names with180positive source days before2026Sep. CurrentMDTtype belongs to a later episode after account end and is not used as a historical exclusion. Metadata is only discovery, not a filter.",
            "financial_parameters_changed": False,
            "before_returns": True,
        },
    )
    for code, symbols, kind, literal in targets:
        data = request(
            output,
            code,
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            {"articleCode": code},
        )
        (output / (code + ".txt")).write_text(text_nodes(json.loads(data["body"])))
        info = source(output, code, symbols, literal)
        registry["type_cards"].extend({**info, "symbol": s, "kind": kind} for s in symbols)
    for code, symbols, literal in [
        ("d592f6ba938746cbadaaf5a8a714abd6", ["HOODUSDT"], "Robinhood"),
        ("f083bdc8f37d4d45a0e0ec6fd5cd4086", ["AMZNUSDT", "COINUSDT", "PLTRUSDT"], "Common Stock"),
    ]:
        info = source(capture, code, symbols, literal)
        registry["type_cards"].extend({**info, "symbol": s, "kind": "equity"} for s in symbols)
    save(output / "registry.json", registry)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_type_requests": 2,
            "type_cards": len(registry["type_cards"]),
            "financial_accounts": 0,
            "historical_original_versions_certified": False,
        },
    )
    print({"type_cards": len(registry["type_cards"]), "new_requests": 2}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    extend(a.parent, a.capture, a.output)
