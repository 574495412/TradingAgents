"""用持仓量增减区分冲击后的新风险承担和去杠杆，保持原账户成本与容量。"""

from __future__ import annotations

import argparse
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from tradingagents.research.quant.backtesting.search import run
from tradingagents.research.quant.backtesting.signals import MINUTE, NAMES, Signals
from tradingagents.research.quant.strategy_audit import RATIOS, read

INTEREST_NAMES = (
    "shock_fade_oi_decrease",
    "shock_fade_oi_increase",
    "shock_fade_long_crowded",
    "shock_fade_oi_decrease_long_crowded",
    "shock_fade_oi_decrease_short_crowded",
    "shock_long_oi_increase",
    "shock_long_oi_increase_short_crowded",
)


class InterestStore:
    """只查精确的历史五分钟观测；缺失或非正持仓量不构成因子。"""

    def __init__(self, public):
        records = read(Path(public) / "metrics_manifest.json")["records"]
        self.paths = {
            (r["symbol"], r["day"]): Path(r["path"]) for r in records if r["status"] == "VERIFIED"
        }
        self.cache = {}

    def at(self, symbol, clock):
        key = symbol, pd.Timestamp(clock, unit="ms", tz="UTC").date().isoformat()
        if key not in self.paths:
            return None
        if key not in self.cache:
            with zipfile.ZipFile(self.paths[key]) as archive:
                if len(archive.namelist()) != 1:
                    raise ValueError("公开持仓量归档结构异常")
                frame = pd.read_csv(
                    archive.open(archive.namelist()[0]),
                    usecols=["create_time", "symbol", "sum_open_interest", *RATIOS],
                )
            clocks = pd.to_datetime(frame["create_time"], utc=True).astype("int64") // 1_000_000
            if not clocks.is_unique or not frame["symbol"].eq(symbol).all():
                raise ValueError("公开持仓量时钟或标的不合法")
            self.cache[key] = dict(
                zip(
                    clocks,
                    frame.loc[:, ["sum_open_interest", *RATIOS]].itertuples(index=False, name=None),
                    strict=True,
                )
            )
        values = self.cache[key].get(clock)
        if values is None or any(not math.isfinite(v) or v <= 0 for v in values):
            return None
        return values


class InterestSignals(Signals):
    """在同一冲击活动内，按自然零点固定持仓量和拥挤方向条件。"""

    def __init__(self, positioning, interest):
        super().__init__(positioning)
        self.interest = interest
        self.used.update({name: set() for name in INTEREST_NAMES})
        self.missing_interest = 0

    def observe(self, clock, frame):
        pulses, qualified = super().observe(clock, frame)
        pulses.update({name: [] for name in INTEREST_NAMES})
        qualified.update({name: {} for name in INTEREST_NAMES})
        cutoff = clock // (5 * MINUTE) * (5 * MINUTE) - 5 * MINUTE
        symbols = set(qualified["shock_confirmed_fade"]) | set(qualified["shock_fade"])
        for symbol in sorted(symbols):
            current, previous = (
                self.interest.at(symbol, cutoff),
                self.interest.at(symbol, cutoff - 15 * MINUTE),
            )
            if current is None or previous is None:
                self.missing_interest += 1
                continue
            decrease, increase = current[0] < previous[0], current[0] > previous[0]
            long_crowded, short_crowded = (
                sum(v > 1 for v in current[1:]) >= 2,
                sum(v < 1 for v in current[1:]) >= 2,
            )
            short_signal = qualified["shock_confirmed_fade"].get(symbol)
            long_signal = qualified["shock_fade"].get(symbol)
            row = frame.loc[symbol]
            long_confirm = (
                long_signal is not None
                and float(row["ret_3_pct"]) > 0
                and float(row["taker_imbalance_3"]) > 0
            )
            conditions = {
                "shock_fade_oi_decrease": short_signal is not None and decrease,
                "shock_fade_oi_increase": short_signal is not None and increase,
                "shock_fade_long_crowded": short_signal is not None and long_crowded,
                "shock_fade_oi_decrease_long_crowded": short_signal is not None
                and decrease
                and long_crowded,
                "shock_fade_oi_decrease_short_crowded": short_signal is not None
                and decrease
                and short_crowded,
                "shock_long_oi_increase": long_confirm and increase,
                "shock_long_oi_increase_short_crowded": long_confirm and increase and short_crowded,
            }
            for name, condition in conditions.items():
                if not condition:
                    continue
                signal = (
                    replace(long_signal, direction=1)
                    if name.startswith("shock_long")
                    else short_signal
                )
                qualified[name][symbol] = signal
                if signal.episode not in self.used[name]:
                    pulses[name].append(signal)
                    self.used[name].add(signal.episode)
        return pulses, qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interest = InterestStore(args.public)
    study = {
        "schema": "cz-profit-factor-search.v2-open-interest",
        "additional_factor_definitions": {
            "open_interest_change": "current contract quantity compared with observation 15 minutes earlier",
            "deleveraging": "open interest decrease plus negative r3 and negative taker3 after price shock",
            "new_short_risk": "open interest increase plus negative r3 and negative taker3 after price shock",
            "crowding_direction": "at least two of three public long/short ratios above or below neutral 1",
            "new_long_risk": "open interest increase plus positive r3 and positive taker3 during ready shock",
        },
        "extra_thresholds_calibrated_from_outcomes": False,
        "historical_windows_previously_researched": True,
    }
    run(
        args,
        names=(*NAMES, *INTEREST_NAMES),
        signals_factory=lambda positioning: InterestSignals(positioning, interest),
        study_protocol=study,
        extra_sources=[__file__],
    )


if __name__ == "__main__":
    main()
