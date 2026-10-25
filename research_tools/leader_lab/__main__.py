"""Input a leader URL/ID; profile evidence or learn and replay a frozen action candidate."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .evidence import load, resolve_identity, save, sha, source_for, write_profile


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv and sys.stdin.isatty():
        argv = ["profile", "--leader", input("输入带单员名称、币安链接或ID：").strip()]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser(
        "profile", help="Fetch/import public position summaries and find observable behavior"
    )
    learning = commands.add_parser(
        "learn", help="Fit interpretable actions and replay an independent hedged account"
    )
    for sub in [profile, learning]:
        sub.add_argument("--leader", required=True)
        sub.add_argument("--output", type=Path)
    profile.add_argument("--source", type=Path)
    profile.add_argument(
        "--refresh", action="store_true", help="Explicitly bypass retained local cache"
    )
    profile.add_argument("--episode-gap-hours", type=float, default=12)
    learning.add_argument("--panel", type=Path, required=True)
    learning.add_argument("--capital", type=float, required=True)
    learning.add_argument(
        "--stake", type=float, required=True, help="Fixed margin per independently owned side"
    )
    learning.add_argument("--leverage", type=float, required=True)
    learning.add_argument("--fee-bps", type=float, required=True)
    learning.add_argument("--slippage-bps", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        leader, lookup = resolve_identity(args.leader)
    except (ValueError, OSError) as exc:
        parser.error(f"Leader lookup failed: {exc}")
    output = args.output or Path(__file__).resolve().parents[
        2
    ] / "results/research/leader_lab" / leader / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "leader_identity.json", {"leader_id": leader, **lookup})
    root = Path(__file__).resolve().parents[2]
    sources = sorted(
        [
            *Path(__file__).parent.glob("*.py"),
            root / "research_tools/__init__.py",
            root / "scripts/leader_lab.py",
        ]
    )
    save(
        output / "source_freeze.json",
        {
            "sources": [{"path": str(p), "sha256": sha(p)} for p in sources],
            "leader_id": leader,
            "command": args.command,
            "registered_utc": datetime.now(UTC).isoformat(),
            "orders_enabled": False,
        },
    )
    try:
        if args.command == "profile":
            path, mode = source_for(leader, args.source, output, args.refresh)
            input_sha = sha(path)
            positions, info = load(path, leader)
            if info["source_sha256"] != input_sha:
                raise ValueError("Source changed during normalization")
            info.update(path=str(path), acquisition_mode=mode)
            result = write_profile(output, leader, positions, info, args.episode_gap_hours)
            if sha(path) != input_sha:
                raise ValueError("Source changed during profiling")
            summary = {
                "stage": "OBSERVABLE_BEHAVIOR_ONLY",
                "closed_summaries": result["positions"],
                "training_from_closed_only_performed": False,
            }
        else:
            from .learning import learn

            economics = {
                "capital": args.capital,
                "stake": args.stake,
                "leverage": args.leverage,
                "fee_bps": args.fee_bps,
                "slippage_bps": args.slippage_bps,
            }
            result = learn(args.panel.absolute(), output, economics, leader)
            summary = {
                "stage": "CANDIDATE_ACTION_AND_HEDGED_REPLAY_EVALUATED",
                "complete_positions": result["account"]["complete_positions"],
                "net_return": result["account"]["net_return"],
                "original_strategy_recovered": False,
                "data_kind": result["account"]["data_kind"],
            }
        frozen = json.loads((output / "source_freeze.json").read_text())
        if not all(sha(Path(s["path"])) == s["sha256"] for s in frozen["sources"]):
            raise ValueError("Implementation changed during run")
        save(
            output / "run_status.json",
            {
                "status": "completed",
                **summary,
                "leader_id": leader,
                "orders_enabled": False,
                "production_eligible": False,
            },
        )
        print(
            json.dumps(
                {"output": str(output), **summary, "production_eligible": False}, ensure_ascii=False
            )
        )
        return 0
    except Exception as exc:
        save(
            output / "run_status.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "leader_id": leader,
                "orders_enabled": False,
                "production_eligible": False,
            },
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
