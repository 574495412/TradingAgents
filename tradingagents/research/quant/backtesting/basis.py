"""实现 TradingAgents 预登记的单一基差回归假说及可用性、稀疏化对照。"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import deque
from pathlib import Path
from statistics import median

import pandas as pd

from tradingagents.dataflows.research.public_reference import ReferenceStore
from tradingagents.research.quant.backtesting.search import COLUMNS, run
from tradingagents.research.quant.backtesting.signals import MINUTE, PROTOCOL, Signals
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.strategy_audit import PublicPositioning, read, save, sha

NAMES = ("basis_parent", "basis_quality_only", "basis_candidate", "basis_sparse_control")
BASIS_LIMIT = 45 / 10000
RANDOM_SEED = 1729


class BasisSignals(Signals):
    """当前决策只使用至少早一分钟的指数偏离，历史窗口不包含对应当前值。"""

    def __init__(self, positioning, reference, *, probability, warmup=None):
        super().__init__(positioning)
        self.reference, self.probability = reference, probability
        self.symbols = sorted({s for s, _ in reference.paths})
        self.history = {s: deque(maxlen=60) for s in self.symbols}
        self.index_history = {s: deque(maxlen=5) for s in self.symbols}
        self.last_delta, self.older_delta = {}, {}
        self.accepted = {n: set() for n in NAMES if n != "basis_parent"}
        self.diagnostics = {
            "ready_opportunities": 0,
            "basis_pass_opportunities": 0,
            "missing_history_or_inactive_reference": 0,
            "candidate_episodes": 0,
            "random_episodes": 0,
        }
        if warmup:
            for clock in sorted(warmup):
                for symbol in self.symbols:
                    self.update_reference(symbol, clock, warmup[clock].get(symbol))

    def update_reference(self, symbol, clock, price):
        ref = self.reference.at(symbol, clock)
        basis = price / ref - 1 if price is not None and ref is not None else None
        history = self.history[symbol]
        delta = (
            basis - median(history)
            if basis is not None and len(history) == 60 and all(v is not None for v in history)
            else None
        )
        self.older_delta[symbol] = self.last_delta.get(symbol)
        self.last_delta[symbol] = delta
        history.append(basis)
        self.index_history[symbol].append(ref)

    def observe(self, clock, frame):
        original, qualified = super().observe(clock, frame)
        pulses = {n: [] for n in NAMES}
        current = {n: {} for n in NAMES}
        pulses["basis_parent"] = original["shock_fade_positioning"]
        current["basis_parent"] = qualified["shock_fade_positioning"]
        for symbol, signal in qualified["shock_fade_positioning"].items():
            d1, d2 = self.last_delta.get(symbol), self.older_delta.get(symbol)
            indices = self.index_history.get(symbol, ())
            ready = (
                d1 is not None
                and d2 is not None
                and len(indices) == 5
                and all(v is not None for v in indices)
                and len(set(indices)) > 1
            )
            if not ready:
                self.diagnostics["missing_history_or_inactive_reference"] += 1
                continue
            self.diagnostics["ready_opportunities"] += 1
            basis_pass = (
                abs(d1) >= BASIS_LIMIT
                and abs(d2) >= BASIS_LIMIT
                and d1 * d2 > 0
                and signal.direction == (-1 if d1 > 0 else 1)
            )
            if basis_pass:
                self.diagnostics["basis_pass_opportunities"] += 1
            uniform = (
                int.from_bytes(
                    hashlib.sha256(f"{RANDOM_SEED}|{signal.episode}|{clock}".encode()).digest()[:8],
                    "big",
                )
                / 2**64
            )
            conditions = {
                "basis_quality_only": True,
                "basis_candidate": basis_pass,
                "basis_sparse_control": uniform < self.probability,
            }
            for name, allowed in conditions.items():
                if not allowed:
                    continue
                current[name][symbol] = signal
                if signal.episode not in self.accepted[name]:
                    pulses[name].append(signal)
                    self.accepted[name].add(signal.episode)
                    if name == "basis_candidate":
                        self.diagnostics["candidate_episodes"] += 1
                    elif name == "basis_sparse_control":
                        self.diagnostics["random_episodes"] += 1
        # 当前指数值只进入下一分钟决策的缓存，不能影响本分钟已经生成的信号。
        for symbol in self.symbols:
            row = frame.loc[symbol]
            price = (
                float(row["close"])
                if row["eligible"]
                and row["observed"]
                and math.isfinite(row["close"])
                and row["close"] > 0
                else None
            )
            self.update_reference(symbol, clock, price)
        return pulses, current


def warmup_prices(panel, symbols):
    """从已封存原始分钟取得基差窗口，不接触研究起点之后的任何价格。"""
    manifest = read(panel / "manifest.json")
    start = int(pd.Timestamp(PROTOCOL["start"]).value // 1_000_000)
    clocks = list(range(start - 122 * MINUTE, start - MINUTE, MINUTE))
    values, sources = {clock: {} for clock in clocks}, {}
    for symbol in symbols:
        paths = [
            Path(p)
            for p in manifest["source_hashes"]
            if p.endswith(f"/raw/{symbol}/2026-09-28.json")
        ]
        if len(paths) != 1 or sha(paths[0]) != manifest["source_hashes"][str(paths[0])]:
            raise ValueError("基差预热原文缺少已封存身份")
        sources[str(paths[0])] = sha(paths[0])
        for row in read(paths[0])["rows"]:
            clock = int(row[0]) + MINUTE
            if clock in values:
                price = float(row[4])
                if not math.isfinite(price) or price <= 0:
                    raise ValueError("基差预热价格不合法")
                values[clock][symbol] = price
    return values, sources


def prepare_coverage(args, output, reference, warmup):
    """只在训练区间测量输入覆盖率，不读取未来标签或账户收益来调概率。"""
    records = read(args.public / "metrics_manifest.json")["records"]
    lookup = PublicPositioning(
        {r["path"]: r["sha256"] for r in records if r["status"] == "VERIFIED"}
    )
    signals = BasisSignals(lookup.at, reference, probability=0, warmup=warmup)
    start, cut = pd.Timestamp(PROTOCOL["start"]), pd.Timestamp(PROTOCOL["training_end"])
    manifest = read(args.panel / "manifest.json")
    for record in manifest["daily_files"]:
        if record["last_close_ms"] < int(
            (start - pd.Timedelta(minutes=1)).value // 1_000_000
        ) or record["first_close_ms"] > int(cut.value // 1_000_000):
            continue
        path = args.panel / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("覆盖率输入不符合原始身份")
        frame = pd.read_parquet(
            path,
            columns=COLUMNS,
            filters=[("bar_time", ">=", start - pd.Timedelta(minutes=1)), ("bar_time", "<=", cut)],
        )
        for at, rows in frame.groupby(level="bar_time", sort=True):
            signals.observe(int(at.value // 1_000_000), rows.droplevel("bar_time"))
    d = signals.diagnostics
    probability = (
        d["basis_pass_opportunities"] / d["ready_opportunities"]
        if d["ready_opportunities"]
        else 0.0
    )
    save(
        output / "coverage.json",
        {
            "training_inputs_only": True,
            "target_returns_read": False,
            "counts": d,
            "fixed_sparse_probability": probability,
            "seed": RANDOM_SEED,
        },
    )
    print(
        f"基差覆盖率已冻结：{d['basis_pass_opportunities']}/{d['ready_opportunities']}，随机概率 {probability:.6f}。",
        flush=True,
    )
    return probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.panel, args.public, args.reference = (
        args.panel.resolve(),
        args.public.resolve(),
        args.reference.resolve(),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    response = read(args.proposal)
    hypothesis = next(
        h
        for h in response["review"]["hypotheses"]
        if h["id"] == "basis_reversion_incremental_information"
    )
    reference = ReferenceStore(args.reference)
    warmup, warm_sources = warmup_prices(args.panel, sorted({s for s, _ in reference.paths}))
    study = {
        "schema": "cz-profit-factor-search.v6-tradingagents-basis",
        "model_proposal_sha256": sha(args.proposal),
        "model_hypothesis": hypothesis,
        "basis_deviation_bps": 45,
        "prior_median_minutes": 60,
        "confirmation_bars": 2,
        "reference_availability_lag_minutes_assumed": 1,
        "reference_activity_proxy": "at least one index-price change in prior5 complete minutes; not proof of observed publication",
        "hold_minutes": [20],
        "reentry_variants": [],
        "warmup_source_files": warm_sources,
        "reference_manifest_sha256": sha(args.reference / "manifest.json"),
        "model_proposal_is_not_verified_alpha": True,
        "unseen_20261014_returns_not_evaluated": True,
        "primary_candidate_frozen_before_replay": "basis_candidate",
        "controls_not_eligible_for_promotion": [
            "basis_parent",
            "basis_quality_only",
            "basis_sparse_control",
        ],
    }
    save(output / "hypothesis_freeze.json", study)
    probability = prepare_coverage(args, output, reference, warmup)
    study["fixed_sparse_probability"] = probability
    study["coverage_sha256"] = sha(output / "coverage.json")
    args.output = output / "accounts"
    run(
        args,
        names=NAMES,
        signals_factory=lambda positioning: BasisSignals(
            positioning, reference, probability=probability, warmup=warmup
        ),
        study_protocol=study,
        extra_sources=[__file__, source_path("public_factor_reference.py")],
        reentry_names=(),
        hold_minutes=(20,),
        selectable_names=("basis_candidate",),
    )
    if (
        any(sha(p) != checksum for p, checksum in warm_sources.items())
        or sha(args.reference / "manifest.json") != study["reference_manifest_sha256"]
    ):
        raise ValueError("基差附加输入在研究期间发生变化")
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    main()
