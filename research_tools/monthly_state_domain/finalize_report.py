"""Correct the prose-only calendar carry claim; preserve every original account."""

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def finalize(source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    run = Path("results/research/monthly_state_accounts_20261027_v001")
    verify_manifest(run)
    original = read(source / "verification.json")
    assert original["passed"] and not original["module_admitted"]
    checks = []
    for cost in [3, 10]:
        identity = f"daily_archive_conditional_trend_cap10000_slip{cost}"
        folder = run / "accounts" / identity
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
        boundary = pd.Timestamp("2026-01-01", tz="UTC")
        events = read(folder / "events.json")
        exits = [e for e in events if e["type"] == "exit" and pd.Timestamp(e["time"]) > boundary]
        entries = [
            e for e in events if e["type"] == "entry" and pd.Timestamp(e["time"]) >= boundary
        ]
        assert len(exits) == 10 and not entries
        assert {e["time"] for e in exits} == {"2026-01-01T00:05:00+00:00"}
        nav_delta = float(curve.equity.iloc[-1] - curve.equity.loc[boundary])
        period = next(
            p
            for p in read(source / "periods.json")
            if p["id"] == identity and p["kind"] == "year" and p["label"] == "2026"
        )
        assert nav_delta == period["net_pnl"] and nav_delta < 0
        closed = math.fsum(
            t["net_pnl"]
            for t in read(folder / "trades.json")
            if pd.Timestamp(t["exit_time"]) > boundary
        )
        assert closed == period["closed_campaign_pnl_in_period"] and closed > 0
        checks.append(
            {
                "id": identity,
                "2026_new_entries": 0,
                "carry_exits": 10,
                "exit_at": exits[0]["time"],
                "2026_marked_nav_change": nav_delta,
                "2026_closed_campaign_pnl_includes_December": closed,
                "distinct_measures_not_interchangeable": True,
            }
        )
    for file in source.iterdir():
        if file.is_file() and file.name not in [
            "completion.json",
            "README.md",
            "verification.json",
        ]:
            shutil.copyfile(file, output / file.name)
    shutil.copyfile(source / "verification.json", output / "original_verification.json")
    text = (source / "README.md").read_text()
    assert text.count("2026零收益保留，不能剔除后再年化。") == 1
    text = text.replace(
        "2026零收益保留，不能剔除后再年化。",
        "2026没有新开仓，但2025年12月的10个遗留仓位到2026-01-01T00:05UTC才退出；2026段净值变化约-1.49USDT，不是零收益。全部现金时段与跨年尾部保留，不能剔除后再年化。",
    )
    text += "\n本版仅修正v001正文将2026无新入场误写成零收益的表述；原账户与年度表数字不变。跨年闭合生命周期PnL包含12月已计入净值的收益，不能当成2026新增收益。见[跨年边界核验](calendar_carry_correction.json)；原报告及其完成凭证完整保留。\n"
    (output / "README.md").write_text(text)
    save(
        output / "calendar_carry_correction.json",
        {
            "passed": True,
            "financial_reruns": 0,
            "prose_claim_corrected": True,
            "original_report": str(source),
            "original_completion_sha256": sha(source / "completion.json"),
            "cases": checks,
        },
    )
    verification = {
        **original,
        "calendar_carry_prose_corrected": True,
        "financial_reruns_for_correction": 0,
        "source_hashes": {
            **original["source_hashes"],
            str(source / "completion.json"): sha(source / "completion.json"),
            str(Path(__file__)): sha(Path(__file__)),
        },
    }
    save(output / "verification.json", verification)
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print(
        {"corrected_report": str(output), "financial_reruns": 0, "boundary_checks": checks},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    finalize(a.source, a.output)
