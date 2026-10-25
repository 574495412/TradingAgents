"""Independent calendar-rolling membership and schedule supersession audit.

Does not import the lifecycle evaluator or its NumPy preview implementation.
"""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(run, output):
    if output.exists():
        raise FileExistsError("Preserve completed independent verification")
    for name, digest in read(run / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(run / name) != digest:
            raise ValueError("Scenario evidence identity differs")
    report = read(run / "summary.json")
    cards = report["cards"]
    daily = Path("data/research/historical_perpetual_daily_20261026_v002")
    original = read(daily / "eligibility_preview.json")
    stored = read(run / "preview.json")
    clocks = pd.DatetimeIndex(pd.to_datetime([r["decision"] for r in original], utc=True))
    feature_rows = {t: [] for t in clocks}
    for p in (daily / "symbols").glob("*_1d.parquet"):
        s = p.name.removesuffix("_1d.parquet")
        f = pd.read_parquet(p)
        calendar = pd.date_range(f.index[0], f.index[-1] + pd.Timedelta(days=1), freq="D")
        d = f.reindex(calendar)
        valid = d.price_liquidity_feature_usable.eq(True)
        q = d.quote_volume.where(valid).rolling(28, min_periods=28).mean().shift(1)
        observations = valid.cumsum().shift(1)
        for t in clocks.intersection(calendar):
            if pd.notna(q.loc[t]) and q.loc[t] >= 50_000_000 and observations.loc[t] >= 180:
                feature_rows[t].append((s, float(q.loc[t])))
    checks = []
    for base, expected in zip(original, stored, strict=True):
        now = pd.Timestamp(base["decision"])
        ranked = sorted(feature_rows[now], key=lambda r: (-r[1], r[0]))
        if [s for s, _ in ranked[:30]] != base["selected"]:
            raise ValueError("Independent rolling baseline membership differs")
        visible = [
            c
            for c in cards
            if pd.Timestamp(c["publish_at_claimed"]) + pd.Timedelta(minutes=5) <= now
        ]
        superseded = {c["supersedes"] for c in visible if c["supersedes"] is not None}
        active = [c for c in visible if c["id"] not in superseded]
        blocked = set()
        for c in active:
            # A revision with no new cutoff retains the old restriction as a
            # conservative research policy, not a claim about exchange admission.
            cut = c["entry_stop_at"]
            prior = c
            while cut is None and prior["supersedes"]:
                prior = next(v for v in visible if v["id"] == prior["supersedes"])
                cut = prior["entry_stop_at"]
            if now >= pd.Timestamp(c["settlement_at"]) or (
                cut is not None and now >= pd.Timestamp(cut)
            ):
                blocked.add(c["symbol"])
        selected = [s for s, _ in ranked if s not in blocked][:30]
        if selected != expected["selected"]:
            raise ValueError("Independent episode or schedule membership differs")
        if any(expected[k] is not False for k in ["historical_tradability_certified"]):
            raise ValueError("Scenario promoted into certified history")
        checks.append({"decision": base["decision"], "passed": True})
    result = {
        "schema": "independent-contract-timeline-preview.v1",
        "passed": True,
        "decisions": len(checks),
        "baseline_reproduced_without_restrictions": True,
        "schedule_revisions_and_distinct_episodes_reproduced": True,
        "source_completion_sha256": sha(run / "completion.json"),
        "verifier_sha256": sha(Path(__file__)),
        "new_financial_accounts": 0,
        "historical_tradability_certified": False,
        "profit_goal_met": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.output)
