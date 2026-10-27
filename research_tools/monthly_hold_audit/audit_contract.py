"""Freeze direct lifecycle and holding evidence before the single policy counterfactual."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    p = Path(__file__).with_name("plan.json")
    plan = read(p)
    for key in ["targets", "reference_run", "reference_report"]:
        verify_manifest(Path(plan[key]))
    target = Path(plan["targets"])
    registry = Path("results/research/sparse_market_cohort_20261026_v003/registry.json")
    card = next(c for c in read(registry)["cards"] if c["symbol"] == "TONUSDT")
    for field in ["source_raw", "source_text"]:
        assert sha(Path(card[field])) == card[field + "_sha256"]
    text = Path(card["source_text"]).read_text()
    assert "2026-06-23 09:00" in text and "2026-06-23 08:30" in text
    sys.path.insert(0, str(Path("../freqtrade").resolve()))
    from research.crypto.sparse_sources_v2 import LifecyclePolicy

    policy = LifecyclePolicy(read(target / "lifecycle.json"), read(target / "plan.json")["symbols"])
    visible = pd.Timestamp(card["available_at_assumed"])
    cutoff, terminal = pd.Timestamp(card["entry_stop_at"]), pd.Timestamp(card["settlement_at"])
    assert policy.at("TONUSDT", visible - pd.Timedelta(milliseconds=1))["_terminal_ns"] is None
    assert policy.at("TONUSDT", visible)["_terminal_ns"] == terminal.value
    assert not policy.entry_blocked("TONUSDT", cutoff - pd.Timedelta(milliseconds=1))
    assert policy.entry_blocked("TONUSDT", cutoff)
    b = read(target / "execution_bindings.json")["TONUSDT"]
    for field in ["bar", "funding"]:
        assert sha(Path(b[field + "_path"])) == b[field + "_sha256"]
    bars = pd.read_parquet(b["bar_path"])
    selected = bars.loc[cutoff - pd.Timedelta(hours=1) : terminal - pd.Timedelta(minutes=5)]
    assert selected.index.equals(
        pd.date_range(cutoff - pd.Timedelta(hours=1), terminal, freq="5min", inclusive="left")
    )
    assert (
        selected.price_usable.all()
        and selected.execution_liquidity_usable.all()
        and selected.quote_volume.gt(0).all()
    )
    selected.to_parquet(output / "TON_preterminal_actual_bars.parquet")
    fill_bar = bars.loc[cutoff]
    assert cutoff + pd.Timedelta(minutes=5) < terminal
    rates = pd.read_parquet(b["funding_path"])
    funding = rates.loc[(rates.index >= cutoff - pd.Timedelta(days=1)) & (rates.index < terminal)]
    assert funding.held_cash_mark_usable.all()
    funding.to_parquet(output / "TON_preterminal_reported_funding.parquet")
    holds, stops = [], []
    run = Path(plan["reference_run"])
    for row in read(run / "summary.json")["account_rows"]:
        if row["reused_account"]:
            continue
        trades = read(run / "accounts" / row["id"] / "trades.json")
        durations = [
            (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 86400
            for t in trades
        ]
        stopped = [t for t in trades if t["reason"] == "stop"]
        holds.append(
            {
                "id": row["id"],
                "complete_cycles": len(trades),
                "stop_cycles": len(stopped),
                "median_complete_holding_days": float(pd.Series(durations).median()),
                "median_stop_holding_days": float(
                    pd.Series(
                        [
                            (
                                pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])
                            ).total_seconds()
                            / 86400
                            for t in stopped
                        ]
                    ).median()
                ),
            }
        )
        stops += [
            {"id": row["id"], **t}
            for t in trades
            if t["symbol"] == "TONUSDT"
            and pd.Timestamp(t["entry_time"]).month == 6
            and pd.Timestamp(t["entry_time"]).year == 2026
        ]
    paths = [
        p,
        Path(__file__),
        registry,
        Path(card["source_raw"]),
        Path(card["source_text"]),
        Path(b["bar_path"]),
        Path(b["funding_path"]),
        target / "completion.json",
        run / "completion.json",
        Path("../freqtrade/research/crypto/sparse_calendar_v2.py"),
        Path("../freqtrade/research/crypto/sparse_sources_v2.py"),
    ]
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "before_counterfactual_returns": True,
            "source_hashes": {str(p): sha(p) for p in paths},
            "new_financial_accounts": 0,
        },
    )
    save(output / "holding_durations.json", holds)
    save(output / "TON_actual_prior_exits.json", stops)
    result = {
        "passed": True,
        "announced_available_at_assumed": visible.isoformat(),
        "cutoff": cutoff.isoformat(),
        "terminal": terminal.isoformat(),
        "actual_prefill_bar_time": cutoff.isoformat(),
        "model_exit_fill_time": (cutoff + pd.Timedelta(minutes=5)).isoformat(),
        "actual_exit_bar_close": float(fill_bar.close),
        "actual_exit_bar_quote_usdt": float(fill_bar.quote_volume),
        "quote1percent_notional_proxy_usdt": float(fill_bar.quote_volume * 0.01),
        "positive_actual_preterminal_bars": len(selected),
        "known_before_cutoff": True,
        "historical_original_notice_and_customer_cash_certified": False,
        "pure_full_month_source_index_still_unknown": True,
        "new_financial_accounts": 0,
        "new_policy_variants_registered": 1,
        "profit_goal_met": False,
    }
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {q.name: sha(q) for q in output.iterdir() if q.is_file()},
            "profit_goal_met": False,
        },
    )
    print({"contract": result, "holding": holds}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args().output)
