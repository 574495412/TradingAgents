"""Causal technical, cross-sectional, funding and time-stamped macro factors."""

from contextvars import ContextVar

import numpy as np
import pandas as pd
import talib

_CATALOG = ContextVar("factor_catalog")


def add(frame, key, series, label, family, formula):
    frame[key] = series
    catalog = _CATALOG.get()
    if key not in {x["id"] for x in catalog}:
        catalog.append(
            {
                "id": key,
                "name": label,
                "family": family,
                "definition": formula,
                "availability": "preceding completed market bar or officially released event +5min",
                "asset_classes": ["crypto"]
                if family == "carry" or key == "funding_macd_interaction"
                else ["crypto", "equity"],
                "window_unit": "observed bars; use a market calendar, do not fill closed sessions",
                "independent_alpha_proven": False,
            }
        )


def event_features(clocks, events):
    if not events:
        return pd.DataFrame(
            np.nan,
            index=clocks,
            columns=[
                "news_attention24",
                "news_age_days",
                "news_hawkish",
                "news_uncertainty",
                "news_inflation",
                "policy_rate",
                "policy_rate_change",
                "news_count7d",
            ],
        )
    e = pd.DataFrame(events)
    e["available_at"] = pd.to_datetime(e.available_at, utc=True)
    selected = pd.merge_asof(
        pd.DataFrame({"time": clocks}),
        e.sort_values("available_at"),
        left_on="time",
        right_on="available_at",
        direction="backward",
    )
    selected.index = clocks
    age = (selected.time - selected.available_at).dt.total_seconds() / 3600
    result = pd.DataFrame(index=clocks)
    result["news_attention24"] = np.exp(-age / 24)
    result["news_age_days"] = age.clip(upper=720) / 24
    result["news_hawkish"] = selected.hawkish_score * np.exp(-age / 168)
    result["news_uncertainty"] = selected.uncertainty_score * np.exp(-age / 168)
    result["news_inflation"] = selected.inflation_emphasis * np.exp(-age / 168)
    result["policy_rate"] = selected.policy_rate_midpoint
    result["policy_rate_change"] = selected.policy_rate_change * np.exp(-age / 168)
    t = e.available_at.astype("int64").to_numpy()
    result["news_count7d"] = np.searchsorted(t, clocks.asi8, "right") - np.searchsorted(
        t, (clocks - pd.Timedelta(days=7)).asi8, "right"
    )
    return result


def calculate_market(
    market, events=None, funding=None, asset_class="crypto", benchmark_symbol=None
):
    """Calculate common factors on supplied session bars, without filling closed markets.

    Each input row is a market bar starting at its index; the row's OHLCV becomes
    usable on the next supplied bar. Rolling lengths count observed bars. Stock
    trading calendars and252-day account risk remain the caller's responsibility.
    Missing quote turnover stays unknown; crypto-only funding is omitted for equities.
    """
    catalog_token = _CATALOG.set([])
    if asset_class not in ("crypto", "equity") or not market:
        raise ValueError("Supply crypto or equity market frames")
    symbols = sorted(market)
    benchmark_symbol = benchmark_symbol or ("BTCUSDT" if "BTCUSDT" in symbols else symbols[0])
    if benchmark_symbol not in market:
        raise ValueError("Benchmark must be in the supplied panel")
    raw, frames = {}, {}
    for s in symbols:
        h = market[s].copy()
        if (
            not isinstance(h.index, pd.DatetimeIndex)
            or h.index.tz is None
            or h.index.has_duplicates
            or not h.index.is_monotonic_increasing
        ):
            raise ValueError("Market bars require a unique sorted timezone-aware calendar")
        if not {"open", "high", "low", "close", "volume"}.issubset(h.columns):
            raise ValueError("Market bars require OHLCV")
        if "quote_volume" not in h:
            h["quote_volume"] = np.nan
        # Row t represents the completed interval ending at t. Nothing from [t,t+1h) is a feature.
        x = h.shift(1)
        f = pd.DataFrame(index=h.index)
        c, v, q = x.close, x.volume, x.quote_volume
        r = c.pct_change(fill_method=None)
        for n in (4, 12, 24, 72, 168, 336, 720):
            add(
                f,
                f"mom_{n}",
                c.pct_change(n, fill_method=None),
                f"{n}期动量",
                "trend",
                f"close[t-1]/close[t-{n + 1}]-1",
            )
        for n in (12, 48, 168):
            add(
                f,
                f"ema_gap_{n}",
                c / c.ewm(span=n, min_periods=n, adjust=False).mean() - 1,
                f"EMA{n}偏离",
                "trend",
                "completed close / EMA -1",
            )
        for n in (24, 72, 168):
            add(
                f,
                f"efficiency_{n}",
                (c - c.shift(n)) / c.diff().abs().rolling(n).sum(),
                f"{n}期路径效率",
                "trend",
                "signed displacement / absolute path",
            )
            add(
                f,
                f"channel_{n}",
                (c - x.low.rolling(n).min()) / (x.high.rolling(n).max() - x.low.rolling(n).min())
                - 0.5,
                f"{n}期通道位置",
                "trend",
                "position within completed high/low range -0.5",
            )
        add(
            f,
            "accel_4_24",
            r.rolling(4).mean() - r.rolling(24).mean(),
            "短期动量加速度",
            "trend",
            "4h mean return -24h mean return",
        )
        add(
            f,
            "accel_24_168",
            r.rolling(24).mean() - r.rolling(168).mean(),
            "中期动量加速度",
            "trend",
            "24h mean return -168h mean return",
        )
        for n in (24, 72, 168):
            add(
                f,
                f"vol_{n}",
                r.rolling(n).std(),
                f"{n}期波动率",
                "risk",
                "sample standard deviation of hourly returns",
            )
        for n in (24, 168):
            add(
                f,
                f"downvol_{n}",
                np.sqrt(r.clip(upper=0).pow(2).rolling(n).mean()),
                f"{n}期下行波动",
                "risk",
                "root mean squared negative hourly return",
            )
        add(
            f,
            "vol_shock",
            r.rolling(12).std() / r.rolling(168).std(),
            "波动率冲击",
            "risk",
            "12h volatility /168h volatility",
        )
        add(f, "skew_72", r.rolling(72).skew(), "收益偏度", "risk", "72h return skewness")
        span = x.high - x.low
        add(
            f,
            "close_location",
            (c - x.low) / span - 0.5,
            "收盘位置",
            "price",
            "close location within completed hour",
        )
        add(
            f,
            "upper_wick",
            (x.high - pd.concat([x.open, c], axis=1).max(axis=1)) / span,
            "上影线比例",
            "price",
            "upper wick / full range",
        )
        add(
            f,
            "lower_wick",
            (pd.concat([x.open, c], axis=1).min(axis=1) - x.low) / span,
            "下影线比例",
            "price",
            "lower wick / full range",
        )
        # Conventional indicators use the same lagged completed candles, via the existing TA-Lib runtime.
        hi, lo, cl, vo = (x[k].to_numpy(dtype=float) for k in ("high", "low", "close", "volume"))
        add(
            f,
            "rsi14",
            pd.Series(talib.RSI(cl, 14), index=f.index) / 100 - 0.5,
            "RSI(14)",
            "technical",
            "TA-Lib RSI14 /100 -0.5",
        )
        macd, signal, hist = talib.MACD(cl, 12, 26, 9)
        add(f, "macd_hist", hist / c, "MACD柱", "technical", "MACD12/26/9 histogram / close")
        add(f, "macd_line", macd / c, "MACD线", "technical", "MACD12/26 line /close")
        add(
            f,
            "atr14",
            talib.ATR(hi, lo, cl, 14) / c,
            "ATR(14)比例",
            "technical",
            "ATR14 /completed close",
        )
        add(
            f,
            "adx14",
            talib.ADX(hi, lo, cl, 14) / 100,
            "ADX(14)",
            "technical",
            "directional strength, not directional prediction",
        )
        add(
            f, "cci20", talib.CCI(hi, lo, cl, 20) / 100, "CCI(20)", "technical", "TA-Lib CCI20 /100"
        )
        add(
            f,
            "mfi14",
            talib.MFI(hi, lo, cl, vo, 14) / 100 - 0.5,
            "MFI(14)",
            "technical",
            "money-flow index from completed OHLCV",
        )
        add(
            f,
            "boll_z20",
            (c - c.rolling(20).mean()) / c.rolling(20).std(ddof=0),
            "布林标准分",
            "technical",
            "(close -20h SMA)/population std",
        )
        add(
            f,
            "boll_width20",
            4 * c.rolling(20).std(ddof=0) / c.rolling(20).mean(),
            "布林带宽",
            "technical",
            "4*20h std /SMA",
        )
        add(
            f,
            "willr14",
            talib.WILLR(hi, lo, cl, 14) / 100,
            "Williams %R",
            "technical",
            "TA-Lib WILLR14 /100",
        )
        obv = pd.Series(talib.OBV(cl, vo), index=f.index)
        add(
            f,
            "obv_change24",
            obv.diff(24) / v.rolling(24).sum(),
            "OBV变化",
            "technical",
            "24h OBV change /24h base volume",
        )
        for n in (24, 168):
            add(
                f,
                f"rel_volume_{n}",
                q / q.rolling(n).mean() - 1,
                f"{n}期相对成交额",
                "liquidity",
                "current completed quote turnover /trailing mean -1",
            )
            add(
                f,
                f"amihud_{n}",
                (r.abs() / q.replace(0, np.nan) * 1e6).rolling(n).mean(),
                f"{n}期非流动性",
                "liquidity",
                "mean absolute return /quote turnover, scaled1e6",
            )
        for n in (24, 72):
            add(
                f,
                f"price_signed_flow_{n}",
                (np.sign(r) * q).rolling(n).sum() / q.rolling(n).sum(),
                f"{n}期价格签名成交额",
                "liquidity",
                "price-signed quote turnover proxy; not actual taker flow",
            )
        add(
            f,
            "log_quote",
            np.log1p(q),
            "成交额对数",
            "liquidity",
            "log(1+completed hourly quote turnover)",
        )
        add(
            f,
            "turnover_shock",
            q.rolling(4).mean() / q.rolling(168).mean() - 1,
            "成交额冲击",
            "liquidity",
            "4h /168h mean quote turnover -1",
        )
        add(
            f,
            "price_volume_corr",
            r.rolling(72).corr(np.log1p(q)),
            "量价相关",
            "liquidity",
            "72h correlation of returns and log quote turnover",
        )
        rate = (
            funding[s]
            if funding is not None and s in funding
            else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        )
        last = rate.reindex(h.index).ffill().shift(1)
        add(
            f,
            "funding_last",
            last,
            "最近已结算资金费",
            "carry",
            "last settled rate strictly before decision hour; not next funding forecast",
        )
        add(
            f,
            "funding_change",
            last - last.shift(8),
            "资金费变化",
            "carry",
            "last settled rate minus rate known8h earlier",
        )
        add(
            f,
            "funding_z",
            (last - last.rolling(168).mean()) / last.rolling(168).std().replace(0, np.nan),
            "资金费标准分",
            "carry",
            "168h completed settled funding zscore",
        )
        add(
            f,
            "funding_pressure",
            last * np.sign(f["mom_24"]),
            "方向融资压力",
            "carry",
            "settled rate times24h momentum sign",
        )
        raw[s] = h
        frames[s] = f
    closes = pd.DataFrame({s: h.close.shift(1) for s, h in raw.items()})
    returns = closes.pct_change(fill_method=None)
    for s, f in frames.items():
        for n in (24, 168):
            m = closes.pct_change(n, fill_method=None)
            add(
                f,
                f"xs_rank_{n}",
                m.rank(axis=1, pct=True)[s] - 0.5,
                f"{n}期横截面排名",
                "market",
                "rank within fixed six-name pool",
            )
            add(
                f,
                f"xs_relative_{n}",
                m[s] - m.mean(axis=1),
                f"{n}期相对动量",
                "market",
                "asset momentum minus same-time pool mean",
            )
            add(
                f,
                f"benchmark_mom_{n}",
                m[benchmark_symbol],
                f"基准 {n}期动量",
                "market",
                "completed benchmark momentum, common market state",
            )
        beta = (
            returns[s].rolling(168).cov(returns[benchmark_symbol])
            / returns[benchmark_symbol].rolling(168).var()
        )
        add(
            f,
            "beta_benchmark168",
            beta,
            "基准滚动Beta",
            "market",
            "168h covariance /benchmark variance",
        )
        add(
            f,
            "residual24",
            (returns[s] - beta.shift(1) * returns[benchmark_symbol]).rolling(24).sum(),
            "基准残差动量",
            "market",
            "sum of returns residualized with preceding-hour beta",
        )
        m = closes.pct_change(24, fill_method=None)
        add(
            f,
            "breadth24",
            m.gt(0).mean(axis=1),
            "市场上涨广度",
            "market",
            "fraction of six contracts with positive24h returns",
        )
        add(
            f,
            "dispersion24",
            m.std(axis=1),
            "市场收益分散度",
            "market",
            "cross-sectional24h return dispersion",
        )
        news = event_features(f.index, events)
        labels = {
            "news_attention24": "FOMC事件衰减",
            "news_age_days": "距官方新闻天数",
            "news_hawkish": "声明鹰鸽词典分数",
            "news_uncertainty": "声明不确定性词频",
            "news_inflation": "声明通胀关注度",
            "policy_rate": "政策利率中点",
            "policy_rate_change": "已公布政策利率变化",
            "news_count7d": "七日官方事件数",
        }
        for key in news:
            add(
                f,
                key,
                news[key],
                labels[key],
                "macro_news",
                "official statement available after declared publication latency; no pre-release content",
            )
        for key, value, label in [
            ("news_trend_interaction", f.news_attention24 * f.mom_24, "事件×动量"),
            (
                "policy_volume_interaction",
                f.policy_rate_change * f.rel_volume_24,
                "政策变化×成交额",
            ),
            ("funding_macd_interaction", f.funding_last * f.macd_hist, "资金费×MACD"),
            ("uncertainty_vol_interaction", f.news_uncertainty * f.vol_shock, "新闻不确定性×波动"),
        ]:
            add(
                f,
                key,
                value,
                label,
                "interaction",
                "predeclared product of two already available inputs",
            )
        # Published factors may be missing; no forward information is used to impute.
        if asset_class == "equity":
            f = f.drop(
                columns=[
                    "funding_last",
                    "funding_change",
                    "funding_z",
                    "funding_pressure",
                    "funding_macd_interaction",
                ]
            )
        frames[s] = f.replace([np.inf, -np.inf], np.nan)
    catalog = [c for c in _CATALOG.get() if asset_class in c["asset_classes"]]
    _CATALOG.reset(catalog_token)
    return frames, raw, catalog


def build(data_root, events, symbols):
    """Crypto source adapter for the shared calendar-neutral calculator."""
    market, rates = {}, {}
    for symbol in symbols:
        b = pd.read_parquet(data_root / f"{symbol}_5m.parquet")
        market[symbol] = b.resample("h").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
            }
        )
        rates[symbol] = pd.read_parquet(data_root / f"{symbol}_funding.parquet").funding_rate
    return calculate_market(market, events, rates, "crypto", "BTCUSDT")
