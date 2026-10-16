"""保持入场因子不变，检验波动止损及反转失效的独立退出对照。"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from tradingagents.research.quant.backtesting.account import Account
from tradingagents.research.quant.backtesting.search import run
from tradingagents.research.quant.backtesting.signals import MINUTE, PROTOCOL, Signals

FACTORS = (
    "shock_fade_positioning",
    "shock_confirmed_fade_positioning",
    "damorne_public",
    "damorne_countermove",
)
EXITS = ("time", "atr_stop", "state_reversal")


class ExitSignals(Signals):
    """三个退出账户获得同一入场信息；未知波动尺度不能授权波动止损账户。"""

    def observe(self, clock, frame):
        original, qualified = super().observe(clock, frame)
        pulses, current = {}, {}
        for factor in FACTORS:
            for style in EXITS:
                name = f"{factor}__{style}"

                def attach(signal):
                    atr = float(frame.loc[signal.symbol, "atr_pct"]) / 100
                    return replace(
                        signal, atr_fraction=atr if math.isfinite(atr) and atr > 0 else None
                    )

                candidates = [attach(s) for s in original[factor]]
                observations = {s: attach(v) for s, v in qualified[factor].items()}
                pulses[name] = [
                    s for s in candidates if style != "atr_stop" or s.atr_fraction is not None
                ]
                current[name] = {
                    s: v
                    for s, v in observations.items()
                    if style != "atr_stop" or v.atr_fraction is not None
                }
        return pulses, current


class ExitAccount(Account):
    """只修改减仓退出逻辑，现金、成交、资金费和续开仍由共用账户处理。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = self.factor.split("__")[1]
        self.current_state = {}

    def open_phase(self, stamp, market):
        before = dict(self.pending)
        super().open_phase(stamp, market)
        for symbol, p in self.positions.items():
            if p["entry_ms"] == stamp and "entry_atr_fraction" not in p:
                # 入场尺度只能来自排单时的已收盘信号，不能取成交分钟尚未完成的特征。
                p["entry_atr_fraction"] = before[symbol]["signal"]["atr_fraction"]
                p["same_direction_flow_seen"] = False

    def close_phase(self, clock, market, pulses, qualified, *, final=False):
        for symbol, p in self.positions.items():
            reason = None
            if self.style == "atr_stop":
                adverse_return = p["direction"] * (market[symbol][1] / p["entry_price"] - 1)
                if adverse_return <= -2 * p["entry_atr_fraction"]:
                    reason = "two_entry_atr_stop"
            elif self.style == "state_reversal":
                state = self.current_state.get(symbol)
                if state:
                    r3, flow = state
                    if p["direction"] * r3 > 0 and p["direction"] * flow > 0:
                        p["same_direction_flow_seen"] = True
                    if (
                        p["same_direction_flow_seen"]
                        and p["direction"] * r3 < 0
                        and p["direction"] * flow < 0
                    ):
                        reason = "confirmed_flow_reversal_failed"
                    elif not p["same_direction_flow_seen"] and clock - p["entry_ms"] >= 20 * MINUTE:
                        reason = "no_confirmation_in_original_20m_horizon"
            if reason and symbol not in self.closes:
                self.closes[symbol] = {
                    "available_ms": clock + PROTOCOL["decision_latency_ms"],
                    "reason": reason,
                }
        return super().close_phase(clock, market, pulses, qualified, final=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = tuple(f"{factor}__{style}" for factor in FACTORS for style in EXITS)
    instances = []

    def factory(*values, **options):
        account = ExitAccount(*values, **options)
        instances.append(account)
        return account

    class ObservedSignals(ExitSignals):
        def observe(self, clock, frame):
            current = {
                s: (float(r3), float(flow))
                for s, r3, flow in zip(
                    frame.index, frame["ret_3_pct"], frame["taker_imbalance_3"], strict=True
                )
                if math.isfinite(r3) and math.isfinite(flow)
            }
            for account in instances:
                account.current_state = current
            return super().observe(clock, frame)

    study = {
        "schema": "cz-profit-factor-search.v3-exit-controls",
        "exit_controls": {
            "time": "original time/1pct-profit/account-risk control",
            "atr_stop": "two times signal-time ATR adverse price move; execute later open",
            "state_reversal": "after observing same-direction r3 and flow, exit on both reversing; otherwise original20m confirmation deadline",
        },
        "entry_factors_unchanged": True,
        "extra_thresholds_fitted_to_trade_outcomes": False,
        "new_exit_samples_previously_researched_history": True,
    }
    reentries = tuple(name for name in names if name.split("__")[0] in PROTOCOL["reentry_variants"])
    run(
        args,
        names=names,
        signals_factory=ObservedSignals,
        study_protocol=study,
        extra_sources=[__file__],
        account_factory=factory,
        reentry_names=reentries,
    )


if __name__ == "__main__":
    main()
