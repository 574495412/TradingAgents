"""Portable offline reproduction from the sealed package; no network or orders.

This reuses the frozen calculation and book code. Running it ourselves is
reproducibility evidence, not independent third-party review or new economics.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def reproduce(root):
    ta = root / "TradingAgents"
    trial = ta / "results/research/dated_carry_prospective_entry_20261027_v001"
    source = ta / "data/research/dated_carry_prospective_entry_sources_20261027_v001"
    manifests = [
        trial,
        source,
        ta / "results/research/dated_carry_prospective_entry_sources_audit_20261027_v001",
        ta / "results/research/dated_carry_prospective_entry_audit_20261027_v001",
        ta / "data/research/dated_carry_forward_docs_20261027_v001",
    ]
    verified = 0
    for folder in manifests:
        for name, expected in read(folder / "completion.json")["manifest"].items():
            target = (folder / name).resolve()
            assert target.is_relative_to(root.resolve()) and digest(target) == expected
            verified += 1
    registration = read(trial / "registration.json")
    for name, expected in registration["source_hashes"].items():
        if name.startswith("/Users/chrise/GitHub/"):
            target = root / name.removeprefix("/Users/chrise/GitHub/")
        else:
            target = ta / name
        target = target.resolve()
        assert target.is_relative_to(root.resolve()) and digest(target) == expected
        verified += 1
    assert (
        registration["registered_wall_ns"]
        < read(source / "registration.json")["registered_wall_ns"]
    )
    obs = read(trial / "observations.json")
    # Extract only frozen pure book definitions to avoid importing optional
    # pandas/numpy/network modules. These are the same provided code, not an
    # independent algorithm or an external review.
    namespace = {"Decimal": Decimal}
    for name, wanted in [
        ("audit_depth_witness.py", {"Book"}),
        ("audit_prospective_probe.py", {"SpotBook", "reconstruct"}),
    ]:
        path = ta / "research_tools/dated_carry" / name
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted
        ]
        assert {n.name for n in nodes} == wanted
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    rows = {}
    for market in ["spot", "future"]:
        rows[market] = []
        for line in (source / (market + "_native.ndjson")).open():
            raw = json.loads(line)
            native = json.loads(raw["native_raw"])
            rows[market].append(
                {
                    "received_wall_ns": raw["received_wall_ns"],
                    "stream": native["stream"],
                    "data": native["data"],
                }
            )
    books = 0
    for asset, o in obs.items():
        for phase in ["decision", "arrival", "unwind", "mark"]:
            for market in ["spot", "future"]:
                symbol = o["symbols"][market]["symbol"].lower()
                delta = [
                    r
                    for r in rows[market]
                    if r["stream"].split("@")[0] == symbol and "@depth" in r["stream"]
                ]
                snap = read(source / f"{market}_{asset}_snapshot.decoded")
                receipt = read(source / f"{market}_{asset}_snapshot.receipt.json")
                book, known = namespace["reconstruct"](
                    delta, snap, receipt["received_wall_ns"], o[phase + "_ns"], market
                )
                rebuilt = {
                    "bids": [[str(p), str(q)] for p, q in sorted(book.bids.items(), reverse=True)],
                    "asks": [[str(p), str(q)] for p, q in sorted(book.asks.items())],
                }
                assert rebuilt == o[phase][market] and known <= o[phase + "_ns"]
                books += 1
    engine_path = root / "freqtrade/research/crypto/dated_forward_paper_v1.py"
    spec = importlib.util.spec_from_file_location("packaged_dated_paper", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    recalculated = serial(engine.run(obs, read(trial / "plan.json")["account"]))
    assert recalculated == read(trial / "account.json")
    return {
        "offline_reproduction_passed": True,
        "manifest_and_strategy_files_verified": verified,
        "raw_receipt_books_rebuilt_exactly": books,
        "account_replay_exact": True,
        "network_calls": 0,
        "new_financial_opportunities": 0,
        "external_review_performed": False,
        "profit_goal_met": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(reproduce(parser.parse_args().root), ensure_ascii=False))
