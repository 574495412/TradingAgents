"""Base open/close factor templates for Weijie Quant.

These templates generate high-recall autonomous candidate signals from
timestamp-available OHLCV, futures taker-flow, kline indicators, market
breadth, and session timing.  Leader records are not used as live inputs here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from factors.crypto.code.weijie.common import finite_float

SIGNAL_COLUMNS = ["signal_id", "symbol", "action", "signal_time", "score", "reason"]

OPEN_FACTOR_FAMILIES = (
    "core_trend_long",
    "midnight_breadth_breakout_long",
    "eth_anchor_long",
    "capitulation_absorption_long",
    "range_reclaim_long",
    "core_downtrend_short",
    "weak_24h_bounce_short",
    "eth_anchor_exhaustion_short",
    "trend_exhaustion_short",
    "paxg_defensive_short",
    "session_pullback_reclaim_long",
    "early_session_reclaim_long",
)

CLOSE_FACTOR_FAMILIES = (
    "portfolio_session_cleanup_long",
    "portfolio_session_cleanup_short",
    "portfolio_midnight_cleanup_long",
    "portfolio_midnight_cleanup_short",
    "long_profit_take_exhaustion",
    "long_relief_bounce_exit",
    "long_risk_cut_reversal",
    "short_risk_cut_reversal",
    "short_profit_take_exhaustion",
)


def build_open_factor_signals(feature_rows: pd.DataFrame, *, config: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in feature_rows.iterrows():
        metrics = row_metrics(row)
        symbol = str(row.get("symbol"))
        signal_time = pd.Timestamp(row.get("bar_time"))
        base = reason_metrics(metrics)
        ret20 = metrics["ret20"]
        ret60 = metrics["ret60"]
        ret144 = metrics["ret144"]
        don = metrics["don"]
        taker = metrics["taker"]
        seller = metrics["seller"]
        qvr = metrics["qvr"]
        rsi = metrics["rsi"]
        adx = metrics["adx"]
        down = metrics["down"]
        up = metrics["up"]
        minute = signal_time.hour * 60 + signal_time.minute
        enable_reversal_open = bool(getattr(config, "enable_reversal_open_factors", False))

        if (
            (ret60 >= 0.45 and don >= 0.72 and rsi >= 54.0 and (taker >= 0.50 or qvr >= 0.75))
            or (ret60 >= 2.40 and don >= 0.78 and rsi >= 58.0)
        ):
            rows.append(
                signal_row(
                    symbol,
                    "OPEN_LONG",
                    signal_time,
                    2.0 + max(ret60, 0.0) / 2.0 + max(don, 0.0),
                    "core_trend_long",
                    base,
                ),
            )
        if (
            signal_time.hour == 0
            and signal_time.minute == 0
            and ret20 >= 1.00
            and 1.80 <= ret60 <= 2.40
            and 0.20 <= ret144 <= 0.65
            and 0.70 <= metrics["ret288"] <= 1.20
            and don >= 0.88
            and qvr >= 3.0
            and 66.0 <= rsi <= 70.0
            and adx >= 35.0
            and up >= 0.90
            and taker <= 0.45
        ):
            rows.append(
                signal_row(
                    symbol,
                    "OPEN_LONG",
                    signal_time,
                    4.35 + min(qvr, 6.0) / 12.0,
                    "midnight_breadth_breakout_long",
                    base,
                ),
            )
        if symbol == "ETHUSDT" and ret20 >= 0.05 and ret144 > -1.50 and don >= 0.62 and taker >= 0.54 and rsi >= 58.0:
            rows.append(signal_row(symbol, "OPEN_LONG", signal_time, 2.3, "eth_anchor_long", base))
        if enable_reversal_open and (
            (
                ret60 <= -1.15
                and don <= 0.34
                and rsi <= 42.0
                and (qvr >= 1.20 or taker >= 0.55 or down >= 0.75)
            )
            or (
                ret60 <= -1.10
                and ret144 <= -2.00
                and metrics["ret288"] <= -2.00
                and 0.15 <= don <= 0.35
                and 32.0 <= rsi <= 42.0
                and adx >= 25.0
                and down >= 0.75
                and 1.20 <= qvr <= 4.00
                and taker >= 0.35
            )
            or (
                symbol == "ETHUSDT"
                and metrics["ret288"] <= -3.00
                and ret20 <= -0.85
                and don <= 0.48
                and rsi <= 46.0
                and qvr >= 1.05
            )
        ):
            rows.append(
                signal_row(
                    symbol,
                    "OPEN_LONG",
                    signal_time,
                    2.65 + min(abs(min(ret60, 0.0)), 6.0) / 3.0 + min(qvr, 8.0) / 8.0,
                    "capitulation_absorption_long",
                    base,
                ),
            )
        if enable_reversal_open and (
            symbol == "ETHUSDT"
            and -0.55 <= ret60 <= 0.30
            and -0.80 <= ret144 <= 0.80
            and 0.45 <= don <= 0.76
            and taker >= 0.55
            and rsi >= 52.0
        ):
            rows.append(signal_row(symbol, "OPEN_LONG", signal_time, 2.55, "range_reclaim_long", base))
        if (
            (ret60 <= -0.20 and don <= 0.36)
            or (ret20 <= -0.55 and don <= 0.45)
            or (ret60 <= -1.10 and rsi <= 55.0)
            or (down >= 0.75 and ret20 <= 0.15 and don <= 0.35)
        ):
            rows.append(
                signal_row(
                    symbol,
                    "OPEN_SHORT",
                    signal_time,
                    2.0 + abs(min(ret60, 0.0)) / 2.0 + (1.0 - don),
                    "core_downtrend_short",
                    base,
                ),
            )
        if ret144 <= -1.40 and ret60 >= 0.35 and 0.55 <= don <= 0.82 and taker >= 0.58 and rsi >= 54.0:
            rows.append(signal_row(symbol, "OPEN_SHORT", signal_time, 2.8, "weak_24h_bounce_short", base))
        eth_relief_bounce_failure_short = (
            symbol == "ETHUSDT"
            and metrics["ret288"] <= -3.00
            and ret144 >= 0.35
            and 0.15 <= ret20 <= 1.80
            and -0.65 <= ret60 <= 0.75
            and don >= 0.78
            and 58.0 <= rsi <= 64.0
            and adx >= 28.0
            and (seller >= 0.42 or taker <= 0.70)
        )
        # 已删除旧 slow-rise-failure 子分支：它依赖狭窄 RSI/收益/量能交集，
        # 30 天删除实验确认对最终点位、账本、收益和回撤均无贡献。
        # 这里只保留尚待后续逐项市场化的主趋势衰竭和 relief 结构。
        eth_anchor_exhaustion = symbol == "ETHUSDT" and (
            (ret60 >= 1.0 and ret144 >= 0.50 and don >= 0.55 and adx >= 28.0 and rsi >= 55.0)
            or eth_relief_bounce_failure_short
        )
        if eth_anchor_exhaustion:
            rows.append(signal_row(symbol, "OPEN_SHORT", signal_time, 3.0, "eth_anchor_exhaustion_short", base))
        trend_exhaustion_short = (
            1.00 <= ret20 <= 2.00
            and 1.50 <= ret60 <= 3.00
            and ret144 >= 1.00
            and metrics["ret288"] >= 0.00
            and 0.74 <= don <= 0.90
            and 62.0 <= rsi <= 72.0
            and adx >= 30.0
            and seller >= 0.55
            and 0.75 <= qvr <= 3.00
        )
        if trend_exhaustion_short and not eth_anchor_exhaustion:
            rows.append(
                signal_row(
                    symbol,
                    "OPEN_SHORT",
                    signal_time,
                    3.20 + ret60 / 4.0 + max(seller - 0.50, 0.0) + min(adx, 60.0) / 100.0,
                    "trend_exhaustion_short",
                    base,
                ),
            )
        if symbol == "PAXGUSDT" and down >= 0.70 and ret60 <= 0.25 and seller >= 0.45:
            rows.append(signal_row(symbol, "OPEN_SHORT", signal_time, 2.4, "paxg_defensive_short", base))
        if (
            ret144 >= 0.75
            and metrics["ret288"] >= 0.75
            and -0.80 <= ret20 <= 0.80
            and 0.10 <= don <= 0.68
            and (taker >= 0.50 or qvr >= 0.70)
            and (minute <= 60 or minute >= 23 * 60 or symbol == "PAXGUSDT")
        ):
            rows.append(signal_row(symbol, "OPEN_LONG", signal_time, 2.45, "session_pullback_reclaim_long", base))
        if (
            metrics["ret288"] >= 1.20
            and ret20 >= 0.35
            and -2.10 <= ret60 <= 0.10
            and 0.20 <= don <= 0.58
            and rsi >= 50.0
            and up >= 0.60
        ):
            rows.append(signal_row(symbol, "OPEN_LONG", signal_time, 2.35, "early_session_reclaim_long", base))
    signals = select_signals(
        pd.DataFrame(rows),
        cooldown_minutes=getattr(config, "open_cooldown_minutes", 0.0),
        eth_blowoff_handoff=bool(
            getattr(config, "production_eth_blowoff_handoff", False)
            or getattr(config, "production_eth_blowoff_ladder", False),
        ),
    )
    signals["signal_id"] = [f"WJOPENFULL{i + 1:05d}" for i in range(len(signals))]
    return signals


def build_close_factor_signals(feature_rows: pd.DataFrame, *, config: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in feature_rows.iterrows():
        metrics = row_metrics(row)
        symbol = str(row.get("symbol"))
        signal_time = pd.Timestamp(row.get("bar_time"))
        minute = signal_time.hour * 60 + signal_time.minute
        base = reason_metrics(metrics)
        ret20 = metrics["ret20"]
        ret60 = metrics["ret60"]
        don = metrics["don"]
        taker = metrics["taker"]
        seller = metrics["seller"]
        qvr = metrics["qvr"]
        rsi = metrics["rsi"]
        adx = metrics["adx"]
        ret144 = metrics["ret144"]
        down = metrics["down"]

        if minute in {6 * 60, 6 * 60 + 30, 10 * 60 + 30, 17 * 60} and (
            qvr >= 0.45 or abs(ret20) >= 0.05 or abs(ret60) >= 0.20
        ):
            rows.append(signal_row(symbol, "CLOSE_LONG", signal_time, 3.0, "portfolio_session_cleanup_long", base))
            rows.append(signal_row(symbol, "CLOSE_SHORT", signal_time, 3.1, "portfolio_session_cleanup_short", base))
        if minute in {0, 5} and (qvr >= 0.30 or abs(ret20) >= 0.03 or abs(ret60) >= 0.10):
            rows.append(signal_row(symbol, "CLOSE_LONG", signal_time, 3.0, "portfolio_midnight_cleanup_long", base))
            rows.append(signal_row(symbol, "CLOSE_SHORT", signal_time, 3.1, "portfolio_midnight_cleanup_short", base))
        if ret60 >= 0.45 and don >= 0.62 and (seller >= 0.50 or taker <= 0.45 or rsi >= 58.0):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    2.5 + ret60 / 3.0,
                    "long_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            symbol == "ETHUSDT"
            and 15 * 60 + 45 <= minute <= 16 * 60 + 10
            and ret20 >= 0.30
            and ret60 >= -0.10
            and ret144 >= 0.35
            and metrics["ret288"] >= 0.20
            and don >= 0.80
            and taker >= 0.55
            and qvr >= 1.0
            and rsi >= 60.0
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    3.55 + min(qvr, 4.0) / 8.0,
                    "long_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            symbol == "ETHUSDT"
            and 19 * 60 + 40 <= minute <= 20 * 60 + 35
            and 0.25 <= ret20 <= 0.90
            and ret60 >= -0.75
            and ret144 <= -1.20
            and metrics["ret288"] <= -1.50
            and 0.58 <= don <= 0.78
            and 53.0 <= rsi <= 68.0
            and qvr >= 0.75
            and taker <= 0.60
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    3.45 + min(max(ret20, 0.0), 1.0) / 3.0 + min(qvr, 4.0) / 10.0,
                    "long_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            symbol == "ETHUSDT"
            and 9 * 60 + 30 <= minute <= 9 * 60 + 45
            and ret20 >= 0.50
            and 0.20 <= ret60 <= 0.60
            and ret144 >= 1.50
            and -1.20 <= metrics["ret288"] <= -0.50
            and don >= 0.80
            and taker >= 0.65
            and qvr >= 5.0
            and rsi >= 64.0
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    3.65 + min(qvr, 10.0) / 10.0,
                    "long_profit_take_exhaustion",
                    base,
                ),
            )
        if ret20 >= 0.45 and ret144 <= 0.0 and rsi >= 58.0 and 0.35 <= don <= 0.78 and (
            (ret60 <= -0.75 and taker >= 0.55) or (ret60 <= -0.10 and seller >= 0.60)
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    3.35 + min(ret20, 2.0) / 4.0 + max(seller - 0.50, taker - 0.55, 0.0),
                    "long_relief_bounce_exit",
                    base,
                ),
            )
        if ret20 <= -0.65 and don <= 0.25 and (seller >= 0.45 or rsi <= 42.0):
            rows.append(signal_row(symbol, "CLOSE_LONG", signal_time, 2.7, "long_risk_cut_reversal", base))
        if (
            9 * 60 + 5 <= minute < 10 * 60 + 30
            and ret60 <= -0.30
            and don <= 0.20
            and rsi <= 45.0
            and down >= 0.88
            and (ret20 <= -0.35 or seller >= 0.55 or qvr >= 1.0)
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_LONG",
                    signal_time,
                    3.35 + min(abs(ret60), 3.0) / 4.0 + max(down - 0.88, 0.0),
                    "long_risk_cut_reversal",
                    base,
                ),
            )
        if ret20 >= 0.20 and don >= 0.58 and (taker >= 0.50 or rsi >= 55.0):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    2.5 + ret20 / 3.0,
                    "short_risk_cut_reversal",
                    base,
                ),
            )
        if (
            ret20 <= -0.45
            and ret60 >= 0.50
            and ret144 >= 0.50
            and don <= 0.45
            and rsi <= 46.0
            and down >= 0.70
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    4.10 + min(abs(ret20), 2.0) / 4.0 + min(ret60, 2.0) / 8.0,
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        if (ret20 <= -0.45 and ret60 <= -0.70 and (taker >= 0.45 or rsi <= 46.0)) or (
            ret20 <= -0.30 and ret60 >= 1.50 and don >= 0.55
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    2.6 + abs(ret20) / 3.0,
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            bool(getattr(config, "production_eth_cascade_profit_exit", False))
            and symbol == "ETHUSDT"
            and ret20 <= -1.50
            and ret60 <= -1.50
            and metrics["ret288"] <= -3.00
            and don <= 0.20
            and qvr >= 3.00
            and rsi <= 30.0
            and seller >= 0.52
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    8.50
                    - min(abs(ret20 + 1.50), 4.0) * 0.75
                    - min(max(qvr - 3.0, 0.0), 20.0) * 0.05,
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            symbol == "ETHUSDT"
            and ret20 <= -0.90
            and ret60 <= 0.55
            and metrics["ret288"] <= -4.00
            and don <= 0.35
            and qvr >= 1.0
            and rsi <= 34.0
            and down >= 0.70
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    3.55 + min(abs(ret20), 2.0) / 4.0,
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        if symbol == "ETHUSDT" and ret60 <= -0.80 and ret144 >= 1.0 and rsi <= 40.0 and don <= 0.15:
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    3.05 + min(abs(ret60), 2.0) / 5.0,
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        blowoff_handoff = bool(getattr(config, "production_eth_blowoff_handoff", False))
        blowoff_ladder = bool(getattr(config, "production_eth_blowoff_ladder", False))
        if (
            (blowoff_handoff or blowoff_ladder)
            and symbol == "ETHUSDT"
            and ret60 >= 4.0
            and ret144 >= 4.0
            and metrics["ret288"] >= 4.0
            and adx >= 60.0
            and seller >= 0.58
            and 0.70 <= qvr <= 3.0
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    4.60 + min(ret60, 8.0) / 8.0 + max(seller - 0.50, 0.0),
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
        if (
            blowoff_ladder
            and symbol == "ETHUSDT"
            and ret60 >= 4.0
            and ret144 >= 4.0
            and metrics["ret288"] >= 4.0
            and adx >= 60.0
            and down >= 0.60
            and 0.70 <= qvr <= 3.0
        ):
            rows.append(
                signal_row(
                    symbol,
                    "CLOSE_SHORT",
                    signal_time,
                    4.55 + min(ret60, 8.0) / 8.0 + max(down - 0.50, 0.0),
                    "short_profit_take_exhaustion",
                    base,
                ),
            )
    signals = select_signals(pd.DataFrame(rows), cooldown_minutes=getattr(config, "close_cooldown_minutes", 0.0))
    signals["signal_id"] = [f"WJCLOSEFULL{i + 1:05d}" for i in range(len(signals))]
    return signals


def select_signals(
    signals: pd.DataFrame,
    *,
    cooldown_minutes: float,
    eth_blowoff_handoff: bool = False,
    preserve_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=[*SIGNAL_COLUMNS, *preserve_columns])
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["signal_time", "symbol", "action"])
    out = out.sort_values(["signal_time", "symbol", "action", "score"], ascending=[True, True, True, False])
    out = out.drop_duplicates(["symbol", "action", "signal_time"], keep="first")
    cooldown = max(0.0, float(cooldown_minutes))
    if cooldown > 0:
        kept: list[int] = []
        last_kept: dict[tuple[str, str], pd.Timestamp] = {}
        for idx, row in out.iterrows():
            key = (str(row.get("symbol")), str(row.get("action")))
            signal_time = pd.Timestamp(row.get("signal_time"))
            previous = last_kept.get(key)
            reason_family = str(row.get("reason") or "").split(";", 1)[0]
            bypass_cooldown = reason_family in {"midnight_breadth_breakout_long"}
            bypass_cooldown = bool(
                bypass_cooldown
                or (
                    key == ("DOGEUSDT", "OPEN_SHORT")
                    and reason_family == "core_downtrend_short"
                    and signal_time.hour == 22
                ),
            )
            if reason_family == "capitulation_absorption_long":
                metrics = _metrics_from_reason(str(row.get("reason") or ""))
                bypass_cooldown = bool(
                    bypass_cooldown
                    or (
                        metrics.get("ret60", 0.0) <= -1.10
                        and metrics.get("ret144", 0.0) <= -1.00
                        and 0.18 <= metrics.get("don", 0.5) <= 0.35
                        and 32.0 <= metrics.get("rsi", 50.0) <= 42.0
                        and metrics.get("qvr", 0.0) >= 1.50
                        and metrics.get("down", 0.0) >= 0.75
                    ),
                )
            if (
                eth_blowoff_handoff
                and key == ("ETHUSDT", "OPEN_SHORT")
                and reason_family == "eth_anchor_exhaustion_short"
            ):
                metrics = _metrics_from_reason(str(row.get("reason") or ""))
                bypass_cooldown = bool(
                    bypass_cooldown
                    or (
                        metrics.get("ret60", 0.0) >= 4.0
                        and metrics.get("ret144", 0.0) >= 4.0
                        and metrics.get("ret288", 0.0) >= 4.0
                        and metrics.get("adx", 0.0) >= 55.0
                        and metrics.get("taker", 0.5) <= 0.50
                    ),
                )
            if previous is not None and (signal_time - previous).total_seconds() / 60.0 < cooldown and not bypass_cooldown:
                continue
            kept.append(int(idx))
            last_kept[key] = signal_time
        out = out.loc[kept].copy()
    out = out.sort_values(["signal_time", "symbol", "action", "score"]).reset_index(drop=True)
    if "signal_id" not in out:
        out["signal_id"] = ""
    return out.reindex(
        columns=[
            *SIGNAL_COLUMNS,
            *(column for column in preserve_columns if column not in SIGNAL_COLUMNS),
        ],
    )


def signal_row(symbol: str, action: str, signal_time: pd.Timestamp, score: float, reason: str, metrics: str) -> dict[str, Any]:
    return {
        "signal_id": "",
        "symbol": symbol,
        "action": action,
        "signal_time": signal_time,
        "score": float(score),
        "reason": f"{reason};{metrics}",
    }


def row_metrics(row: pd.Series) -> dict[str, float]:
    taker = finite_float(row.get("taker_buy_ratio"), 0.5)
    return {
        "ret20": finite_float(row.get("ret20")),
        "ret60": finite_float(row.get("ret60")),
        "ret144": finite_float(row.get("ret144")),
        "ret288": finite_float(row.get("ret288")),
        "don": finite_float(row.get("donchian60_pos"), 0.5),
        "taker": taker,
        "seller": finite_float(row.get("seller_pressure"), 1.0 - taker),
        "qvr": finite_float(row.get("quote_volume_ratio_med30"), 1.0),
        "rsi": finite_float(row.get("rsi14"), 50.0),
        "adx": finite_float(row.get("adx14"), 20.0),
        "atr_pct": finite_float(row.get("atr14_pct"), finite_float(row.get("atr_pct"))),
        "down": finite_float(row.get("market_down_breadth20"), 0.0),
        "up": finite_float(row.get("market_up_breadth20"), 0.0),
    }


def reason_metrics(metrics: dict[str, float]) -> str:
    return (
        f"ret20={metrics['ret20']:.2f};ret60={metrics['ret60']:.2f};ret144={metrics['ret144']:.2f};"
        f"ret288={metrics['ret288']:.2f};don={metrics['don']:.2f};taker={metrics['taker']:.2f};"
        f"seller={metrics['seller']:.2f};qvr={metrics['qvr']:.2f};rsi={metrics['rsi']:.1f};"
        f"adx={metrics['adx']:.1f};atr_pct={metrics['atr_pct']:.4f};"
        f"down={metrics['down']:.2f};up={metrics['up']:.2f}"
    )


def _metrics_from_reason(reason: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for token in str(reason).split(";"):
        key, separator, value = token.partition("=")
        if not separator:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics
