"""事前固定的事件阶段、公开拥挤度和流向确认候选，仅读取已收盘数据。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MINUTE = 60_000
NAMES = (
    "shock_fade",
    "shock_fade_positioning",
    "shock_confirmed_fade",
    "shock_confirmed_fade_positioning",
    "shock_positioning_long",
    "damorne_public",
    "damorne_countermove",
)
PROTOCOL = {
    "schema": "cz-profit-factor-search.v1",
    "start": "2026-09-29T00:00:00+00:00",
    "training_end": "2026-10-06T00:00:00+00:00",
    "end": "2026-10-14T00:00:00+00:00",
    "initial_equity": 1000.0,
    "position_fraction": 0.125,
    "max_positions": 4,
    "decision_latency_ms": MINUTE,
    "max_signal_age_ms": 5 * MINUTE,
    "shock_confirmation_window_ms": 60 * MINUTE,
    "profit_receipt_valid_ms": 60 * MINUTE,
    "hold_minutes": [20, 60],
    "fee_bps": 5.0,
    "slippage_bps": [3.0, 10.0],
    "take_profit_price_fraction": 0.01,
    "maximum_account_drawdown": 0.08,
    "positioning_lag_ms": 5 * MINUTE,
    "minimum_trade_notional": 10.0,
    "reentry_variants": ["shock_confirmed_fade_positioning", "damorne_countermove"],
    "gate": {
        "minimum_trades_per_phase": 30,
        "minimum_profit_factor": 1.1,
        "minimum_wilson_lower": 0.4,
        "positive_base_and_stress_both_phases": True,
    },
    "factor_definitions": {
        "shock": "TradFi, r20>=1.5%, r60>=2%, max(q20,q60)>=0.985, ATRrank>=0.93",
        "positioning": "at least two of three ratios below 1; observation at floor5min(close)-5min",
        "fade_confirmation": "first r3<0 and taker3<0 within 60min after a fresh shock",
        "damorne_public": "volume>=0.9,liquidity>=0.5; q20/residual20/taker3 same signed extreme",
        "countermove": "every5min,liquidity>=0.97,abs(res20)/(vol60*sqrt20)>=1; r3 and flow12 reverse, absflow>=0.05",
        "profit_reentry": "actual net-positive close including funding, then current original factor condition; single-use receipt",
    },
    "funding_included": True,
    "timestamp_semantics": "CANDLE_CLOSE; open=row_time-60000",
    "independent_oos": False,
    "production_eligible": False,
    "leader_labels_read": False,
    "orders_enabled": False,
    "previously_researched_history": True,
}


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: int
    priority: float
    as_of_ms: int
    episode: str
    atr_fraction: float | None = None


class Signals:
    """每个冲击活动只生成一次各类阶段触发，未知持仓不当作中性零。"""

    def __init__(self, positioning):
        self.positioning = positioning
        self.anchors, self.previous_shock, self.previous_positioned, self.previous_da = (
            {},
            set(),
            set(),
            {},
        )
        self.used = {name: set() for name in NAMES}
        self.last_ms = None
        self.missing_positioning = 0

    def observe(self, clock, frame):
        if self.last_ms is not None and clock != self.last_ms + MINUTE:
            raise ValueError("信号必须逐个真实收盘分钟推进")
        pulses, qualified = {n: [] for n in NAMES}, {n: {} for n in NAMES}
        valid = frame["eligible"].eq(True) & frame["observed"].eq(True)
        r20, r60 = frame["ret_20_pct"], frame["ret_60_pct"]
        ready = (
            valid
            & frame["asset_class"].eq("TRADIFI_PERPETUAL")
            & r20.ge(1.5)
            & r60.ge(2)
            & frame[["ret_20_rank_pct", "ret_60_rank_pct"]].max(axis=1).ge(0.985)
            & frame["atr_rank_pct"].ge(0.93)
        )
        shock_now = set(frame.index[ready])
        for symbol in shock_now - self.previous_shock:
            self.anchors[symbol] = clock
        self.anchors = {
            s: t
            for s, t in self.anchors.items()
            if s in shock_now or clock - t <= PROTOCOL["shock_confirmation_window_ms"]
        }
        positioned_now = set()
        for symbol in sorted(shock_now | set(self.anchors)):
            row = frame.loc[symbol]
            if not valid.loc[symbol]:
                continue
            anchor = self.anchors[symbol]
            values = self.positioning(symbol, clock // (5 * MINUTE) * (5 * MINUTE) - 5 * MINUTE)
            if values is None:
                self.missing_positioning += 1
                positioned = False
            else:
                positioned = sum(v < 1 for v in values) >= 2
            priority = abs(float(row["ret_20_pct"])) / max(float(row["atr_pct"]), 1e-9)
            if not math.isfinite(priority):
                continue
            episode = f"shock:{symbol}:{anchor}"
            short_confirmed = (
                clock - anchor <= PROTOCOL["shock_confirmation_window_ms"]
                and float(row["ret_3_pct"]) < 0
                and float(row["taker_imbalance_3"]) < 0
            )
            conditions = {
                "shock_fade": symbol in shock_now,
                "shock_fade_positioning": symbol in shock_now and positioned,
                "shock_confirmed_fade": short_confirmed,
                "shock_confirmed_fade_positioning": short_confirmed and positioned,
                "shock_positioning_long": symbol in shock_now and positioned,
            }
            if symbol in shock_now and positioned:
                positioned_now.add(symbol)
            for name, condition in conditions.items():
                if not condition:
                    continue
                signal = Signal(
                    symbol, 1 if name == "shock_positioning_long" else -1, priority, clock, episode
                )
                qualified[name][symbol] = signal
                if name == "shock_fade":
                    fresh = symbol not in self.previous_shock
                elif name in {"shock_fade_positioning", "shock_positioning_long"}:
                    fresh = symbol not in self.previous_positioned
                else:
                    fresh = episode not in self.used[name]
                if fresh:
                    pulses[name].append(signal)
                    self.used[name].add(episode)
        da = valid & frame["volume_ratio_rank_pct"].ge(0.9) & frame["liquidity_rank_pct"].ge(0.5)
        da_long = (
            da
            & frame["ret_20_rank_pct"].ge(0.9)
            & frame["ret_20_residual_pct"].gt(0)
            & frame["taker_imbalance_3"].gt(0)
        )
        da_short = (
            da
            & frame["ret_20_rank_pct"].le(0.1)
            & frame["ret_20_residual_pct"].lt(0)
            & frame["taker_imbalance_3"].lt(0)
        )
        da_now = {}
        for direction, mask in ((1, da_long), (-1, da_short)):
            for symbol in frame.index[mask]:
                row = frame.loc[symbol]
                signal = Signal(
                    symbol,
                    direction,
                    float(row["volume_ratio_rank_pct"])
                    + abs(float(row["ret_20_residual_pct"])) / 1000,
                    clock,
                    f"da:{symbol}:{clock}",
                )
                qualified["damorne_public"][symbol] = signal
                da_now[symbol] = direction
                if self.previous_da.get(symbol) != direction:
                    pulses["damorne_public"].append(signal)
        # 原公开规则每个资产类别只发送排名最高的新事件。
        selected = []
        for lane in ("CRYPTO_PERPETUAL", "TRADIFI_PERPETUAL"):
            choices = [
                s for s in pulses["damorne_public"] if frame.loc[s.symbol, "asset_class"] == lane
            ]
            if choices:
                selected.append(max(choices, key=lambda s: (s.priority, s.symbol)))
        pulses["damorne_public"] = selected
        vol = frame["return_volatility_60_pct"]
        counter_direction = -np.sign(frame["ret_20_residual_pct"])
        z20 = frame["ret_20_residual_pct"].abs() / (vol * math.sqrt(20))
        counter = (
            valid
            & frame["liquidity_rank_pct"].ge(0.97)
            & vol.gt(0)
            & z20.ge(1)
            & (counter_direction * frame["ret_3_pct"]).gt(0)
            & (counter_direction * frame["taker_imbalance_12"]).ge(0.05)
        )
        for symbol in frame.index[counter]:
            row = frame.loc[symbol]
            priority = float(
                z20.loc[symbol]
                + abs(row["ret_3_pct"]) / (row["return_volatility_60_pct"] * math.sqrt(3))
            )
            if not math.isfinite(priority):
                continue
            qualified["damorne_countermove"][symbol] = Signal(
                symbol,
                int(counter_direction.loc[symbol]),
                priority,
                clock,
                f"counter:{symbol}:{clock}",
            )
        if clock % (5 * MINUTE) == 0 and qualified["damorne_countermove"]:
            pulses["damorne_countermove"] = [
                max(qualified["damorne_countermove"].values(), key=lambda s: (s.priority, s.symbol))
            ]
        self.previous_shock, self.previous_positioned, self.previous_da, self.last_ms = (
            shock_now,
            positioned_now,
            da_now,
            clock,
        )
        return pulses, qualified
