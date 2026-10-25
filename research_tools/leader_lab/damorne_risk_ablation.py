"""Frozen heuristic controls to isolate lifecycle/risk effects from payoff model rejection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .damorne_mechanisms import market_cache
from .damorne_payoff import ECONOMICS, metadata_with_ticks
from .damorne_research import millis
from .damorne_tail_account import candidates, run_account
from .evidence import save, sha


def control_signals(market, role_separated):
    predictions = {}
    for symbol, frame in market.items():
        previous = frame.shift(1)
        values = np.zeros((len(frame), 2, 4))
        for side, sign in enumerate([1, -1]):
            seed = (
                (sign * previous.return60 >= 0.01)
                & (sign * previous.return15 > 0)
                & (sign * previous.return5 < 0)
                & (sign * previous.flow > 0.1)
            )
            continuation = (
                (sign * previous.return60 > 0)
                & (sign * previous.return15 > 0)
                & (sign * previous.flow > 0)
            )
            large = (
                seed
                & (previous.volume_ratio >= 2)
                & (previous.quote60 >= 5_000_000)
                & (sign * previous.return15 >= 0.002)
            )
            # These are explicit boolean gate codes, never calibrated probabilities or alpha estimates.
            values[:, side, 0] = np.where(
                seed, 1.0, np.where(role_separated & continuation, 0.56, 0.0)
            )
            values[:, side, 1] = np.where(seed | (role_separated & continuation), 0.002, 0.0)
            values[:, side, 2] = large.astype(float)
            values[:, side, 3] = np.where(large, 0.004, 0.0)
        predictions[symbol] = values
    return predictions


def run(source, caches, output):
    output.mkdir(exist_ok=False)
    protocol = json.loads((source / "protocol.json").read_text())
    all_market = market_cache(source, caches)
    primary = {s: all_market[s] for s in protocol["training_symbols"]}
    transfer_symbols = [
        "1000XECUSDT",
        "AGTUSDT",
        "BTCUSDT",
        "BTWUSDT",
        "ETHUSDT",
        "NEARUSDT",
        "PENGUUSDT",
        "QUSDT",
        "RAYSOLUSDT",
        "RLCUSDT",
        "STRKUSDT",
        "TIAUSDT",
        "UAIUSDT",
        "UNIUSDT",
        "ZECUSDT",
        "ZROUSDT",
    ]
    transfer = {s: all_market[s] for s in transfer_symbols}
    combined = {**primary, **transfer}
    metadata = metadata_with_ticks(json.loads((source / "discovery/catalog.json").read_text()))
    variants = []
    for role_separated in [False, True]:
        for base in candidates(ECONOMICS)[:4]:
            variants.append(
                {
                    **base,
                    "id": f"heuristic-role{int(role_separated)}-add{int(base['allow_additions'])}-trim{int(base['allow_reductions'])}",
                    "role_separated": role_separated,
                    "signal_scope": "BOOLEAN_HEURISTIC_CONTROL_NOT_MODEL_PROBABILITIES",
                    "admission": "frozen trend pullback for initial seeds; optional own profitable-close continuation",
                    "production_eligible": False,
                }
            )
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("damorne_tail_account.py"),
        Path(__file__).with_name("damorne_research.py"),
    ]
    frozen = {str(p): sha(p) for p in source_paths}
    save(
        output / "candidate_freeze.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "candidates": variants,
            "source_hashes": frozen,
            "primary_symbols": sorted(primary),
            "transfer_symbols": sorted(transfer),
            "scope": "retrospective development and risk ablation; no fresh forward evidence",
            "selection_rule": "validation both sleeves net and closed PNL positive,>=20small/>=5large,DD<=8%; rank net return",
            "held_evaluation_gate": "both sleeves net and closed PNL positive,>=30small/>=15large,DD<=8%,+2bps stress positive both",
            "teacher_actions_or_fills_as_runtime_inputs": False,
        },
    )
    snapshot = output / "executed_sources"
    snapshot.mkdir()
    for p in source_paths:
        (snapshot / p.name).write_bytes(p.read_bytes())
    signals = {role: control_signals(combined, role) for role in [False, True]}
    lower, middle, end = (
        millis("2026-10-10T00:00:00+00:00"),
        millis("2026-10-14T00:00:00+00:00"),
        millis("2026-10-24T00:00:00+00:00"),
    )
    middle_marks = {s: float(f.loc[middle, "mark"]) for s, f in primary.items()}
    all_marks = json.loads((output.parent / "combined_evaluation_marks.json").read_text())["marks"]
    end_marks = {s: all_marks[s] for s in primary}
    validation = []
    for policy in variants:
        summary, _ = run_account(
            policy,
            primary,
            signals[policy["role_separated"]],
            metadata,
            lower,
            middle,
            middle_marks,
        )
        small, large = summary["sleeves"]["small"], summary["sleeves"]["large"]
        summary["selection_gate"] = bool(
            all(
                s["net_pnl"] > 0 and s["net_pnl"] - s["terminal_open_net_pnl"] > 0
                for s in [small, large]
            )
            and small["complete_positions"] >= 20
            and large["complete_positions"] >= 5
            and summary["minute_observed_mark_open_drawdown"] <= 0.08
        )
        validation.append(summary)
        print(
            {
                "validation": policy["id"],
                "net_return": summary["net_return"],
                "small": small["net_pnl"],
                "large": large["net_pnl"],
                "gate": summary["selection_gate"],
            },
            flush=True,
        )
    save(output / "validation_scores.json", validation)
    qualified = [x for x in validation if x["selection_gate"]]
    chosen = sorted(qualified or validation, key=lambda r: (-r["net_return"], r["candidate"]))[0]
    policy = next(c for c in variants if c["id"] == chosen["candidate"])
    save(
        output / "selection.json",
        {
            "candidate": policy,
            "validation_gate": bool(qualified),
            "evaluation_outcomes_read": False,
            "scope": "selected historical research candidate"
            if qualified
            else "diagnostic best validation only",
        },
    )
    result, state = run_account(
        policy, primary, signals[policy["role_separated"]], metadata, middle, end, end_marks
    )
    save(output / "evaluation_score.json", result)
    save(output / "evaluation_state.json", state)
    checkpoint = middle + (end - middle) // 120000 * 60000
    _, prefix = run_account(
        policy,
        primary,
        signals[policy["role_separated"]],
        metadata,
        middle,
        end,
        end_marks,
        break_ms=checkpoint,
    )
    resumed, tail = run_account(
        policy,
        primary,
        signals[policy["role_separated"]],
        metadata,
        checkpoint,
        end,
        end_marks,
        state=json.loads(json.dumps(prefix)),
    )
    if resumed != result or tail != state:
        raise ValueError("Risk ablation restart changes account")
    save(
        output / "restart_verification.json", {"exact_state_equal": True, "exact_score_equal": True}
    )
    stresses = []
    for extra in [2, 5]:
        c = {
            **policy,
            "execution_cost_bps": policy["execution_cost_bps"] + extra,
            "fee_bps": policy["fee_bps"] + extra,
        }
        stress, _ = run_account(
            c, primary, signals[c["role_separated"]], metadata, middle, end, end_marks
        )
        stresses.append({"extra_cost_bps_each_side": extra, **stress})
    save(output / "cost_stress.json", stresses)
    transfer_score, transfer_state = run_account(
        policy,
        transfer,
        signals[policy["role_separated"]],
        metadata,
        middle,
        end,
        {s: all_marks[s] for s in transfer},
    )
    save(output / "transfer_score.json", transfer_score)
    save(output / "transfer_state.json", transfer_state)
    small, large = result["sleeves"]["small"], result["sleeves"]["large"]
    eligible = bool(
        qualified
        and small["complete_positions"] >= 30
        and large["complete_positions"] >= 15
        and result["minute_observed_mark_open_drawdown"] <= 0.08
        and all(
            s["net_pnl"] > 0 and s["net_pnl"] - s["terminal_open_net_pnl"] > 0
            for s in [small, large]
        )
        and all(s["net_pnl"] > 0 for s in stresses[0]["sleeves"].values())
    )
    if any(sha(Path(p)) != digest for p, digest in frozen.items()):
        raise ValueError("Implementation changed during risk study")
    save(
        output / "run_status.json",
        {
            "status": "completed",
            "paper_research_gate": eligible,
            "production_eligible": False,
            "orders_enabled": False,
            "source_labels_as_runtime_seeds": False,
            "unseen_forward_evidence": False,
        },
    )
    print(
        {
            "evaluation": result,
            "transfer_net_return": transfer_score["net_return"],
            "paper_research_gate": eligible,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source.absolute(), a.cache.absolute(), a.output.absolute())
