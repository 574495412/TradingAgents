"""Paired funding-budget repair; same previously rejected entry/scaling policy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .damorne_mechanisms import market_cache
from .damorne_payoff import metadata_with_ticks
from .damorne_research import millis
from .damorne_risk_ablation import control_signals
from .damorne_tail_account import run_account
from .evidence import save, sha


def study(root, output):
    output.mkdir(exist_ok=False)
    parent = root / "risk_ablation_v001"
    selection = json.loads((parent / "selection.json").read_text())
    baseline = selection["candidate"]
    repaired = {**baseline, "id": baseline["id"] + "-funding-budget", "funding_adjusted_stop": True}
    sources = [
        Path(__file__),
        Path(__file__).with_name("damorne_tail_account.py"),
        Path(__file__).with_name("damorne_risk_ablation.py"),
        Path(__file__).with_name("damorne_research.py"),
    ]
    frozen = {str(p): sha(p) for p in sources}
    save(
        output / "protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "candidates": [baseline, repaired],
            "source_hashes": frozen,
            "parent_selection_sha256": sha(parent / "selection.json"),
            "scope": "paired retrospective funding-risk repair; same rejected entry policy",
            "hypothesis": "settled funding debit tightens stop to preserve original dollar loss budget; gap losses remain uncapped",
            "selection": "no new candidate selection and no admission based on this reused evaluation period",
            "validation_start": "2026-10-10T00:00:00+00:00",
            "evaluation_start": "2026-10-14T00:00:00+00:00",
            "evaluation_end": "2026-10-24T00:00:00+00:00",
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    snapshots = output / "executed_sources"
    snapshots.mkdir()
    for p in sources:
        (snapshots / p.name).write_bytes(p.read_bytes())
    data = root / "combined_data"
    symbols = json.loads((data / "protocol.json").read_text())["training_symbols"]
    market = {
        s: f for s, f in market_cache(data, root / "combined_mechanisms").items() if s in symbols
    }
    meta = metadata_with_ticks(json.loads((data / "discovery/catalog.json").read_text()))
    signals = control_signals(market, True)
    start = millis("2026-10-10T00:00:00+00:00")
    middle = millis("2026-10-14T00:00:00+00:00")
    end = millis("2026-10-24T00:00:00+00:00")
    marks = json.loads((root / "combined_evaluation_marks.json").read_text())
    if marks["timestamp"] != end:
        raise ValueError("Terminal mark clock differs")
    endpoint = {s: marks["marks"][s] for s in market}
    mid_marks = {s: float(f.loc[middle, "mark"]) for s, f in market.items()}
    evaluations = []
    validation = []
    repaired_state = None
    for policy in [baseline, repaired]:
        score, _ = run_account(policy, market, signals, meta, start, middle, mid_marks)
        validation.append(score)
        score, state = run_account(policy, market, signals, meta, middle, end, endpoint)
        evaluations.append(score)
        print(
            {
                "candidate": policy["id"],
                "net_return": score["net_return"],
                "breaches": {
                    s: x["gap_or_funding_risk_budget_breaches"] for s, x in score["sleeves"].items()
                },
            },
            flush=True,
        )
        if policy is baseline:
            if score != json.loads((parent / "evaluation_score.json").read_text()):
                raise ValueError("Unchanged control no longer reproduces original account")
            if state != json.loads((parent / "evaluation_state.json").read_text()):
                raise ValueError("Unchanged control state differs")
        else:
            repaired_state = state
    save(output / "validation_scores.json", validation)
    save(output / "evaluation_scores.json", evaluations)
    save(output / "evaluation_state.json", repaired_state)
    checkpoint = middle + (end - middle) // 2
    _, prefix = run_account(
        repaired, market, signals, meta, middle, end, endpoint, break_ms=checkpoint
    )
    resumed, resumed_state = run_account(
        repaired,
        market,
        signals,
        meta,
        checkpoint,
        end,
        endpoint,
        state=json.loads(json.dumps(prefix)),
    )
    if resumed != evaluations[1] or resumed_state != repaired_state:
        raise ValueError("Funding repair does not survive exact restart")
    save(
        output / "restart_verification.json",
        {
            "unchanged_control_exact_state_and_score": True,
            "repaired_exact_state_and_score_after_restart": True,
            "checkpoint_ms": checkpoint,
        },
    )
    stress = []
    for extra in [2, 5]:
        policy = {
            **repaired,
            "execution_cost_bps": repaired["execution_cost_bps"] + extra,
            "fee_bps": repaired["fee_bps"] + extra,
        }
        score, _ = run_account(policy, market, signals, meta, middle, end, endpoint)
        stress.append({"extra_cost_bps_each_side": extra, **score})
    save(output / "cost_stress.json", stress)
    if any(sha(Path(p)) != digest for p, digest in frozen.items()):
        raise ValueError("Frozen source changed during funding study")
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "paper_research_gate": False,
            "original_validation_gate": selection["validation_gate"],
            "scope": "risk repair only; entry policy not requalified",
            "production_eligible": False,
            "orders_enabled": False,
            "genuinely_unseen_forward_evidence": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.root.absolute(), args.output.absolute())
