"""Normalize existing accounts while retaining incompatible scopes and raw metrics."""

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from research_tools.nfi_reproduction.audit_original import cash_from_orders


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(x):
    if isinstance(x, bool):
        return None
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def pick(d, *keys):
    return next((d[k] for k in keys if d.get(k) is not None), None)


def curve_stats(frame, capital, opening_date=None, annualization=365):
    name = next((k for k in ["date", "time", "open_ms", "timestamp", "ts"] if k in frame), None)
    if name is None or "equity" not in frame or len(frame) < 2:
        return None
    f = frame.copy()
    f["date"] = pd.to_datetime(f[name], unit="ms" if name == "open_ms" else None, utc=True)
    f = f.sort_values("date", kind="stable")
    seeded = False
    if opening_date and capital and pd.Timestamp(opening_date, tz="UTC") < f.date.iloc[0]:
        start = pd.Timestamp(opening_date, tz="UTC")
        if f.date.iloc[0] - start <= pd.Timedelta(hours=1):
            seed = f.iloc[0].copy()
            seed["date"], seed["equity"] = start, capital
            f = pd.concat([pd.DataFrame([seed]), f], ignore_index=True)
            seeded = True
    # Retain same-clock observations before and after native end settlement for DD.
    raw_equity = f.equity.astype(float)
    raw_peaks = raw_equity.cummax().clip(lower=capital) if capital else raw_equity.cummax()
    raw_drawdown = float((1 - raw_equity / raw_peaks).max())
    f = f.drop_duplicates("date", keep="last").set_index("date")
    equity = f.equity.astype(float)
    if not (equity > 0).all():
        return {"invalid": "nonpositive_equity"}
    daily_levels = equity.resample("1D").first()
    missing_days = int(daily_levels.isna().sum())
    if annualization == 252:
        daily_levels = daily_levels.dropna()
    daily = daily_levels.pct_change().dropna()
    out = {
        "start": str(f.index[0]),
        "end": str(f.index[-1]),
        "rows": len(f),
        "last_equity": float(equity.iloc[-1]),
        "sampled_full_drawdown": raw_drawdown,
        "recomputed_daily_sharpe": float(
            daily.mean() / daily.std(ddof=1) * math.sqrt(annualization)
        )
        if len(daily) > 1 and daily.std(ddof=1) > 0 and (annualization == 252 or not missing_days)
        else None,
        "daily_return_count": len(daily),
        "annualization_periods": annualization,
        "missing_calendar_day_observations": missing_days,
        "known_initial_cash_seed_added": seeded,
        "valuation": "existing saved equity, not certified executable bid/ask",
        "month_returns": {},
    }
    for period in f.index.strftime("%Y-%m").unique():
        lo = pd.Timestamp(str(period) + "-01", tz="UTC")
        hi = lo + pd.offsets.MonthBegin(1)
        # Require both month endpoints; never extrapolate a partial month.
        if lo in f.index and hi in f.index:
            out["month_returns"][str(period)] = float(f.loc[hi, "equity"] / f.loc[lo, "equity"] - 1)
    return out


def near_plan(path):
    for p in list(path.parents)[:5]:
        f = p / "plan.json"
        if f.is_file() and f.stat().st_size < 1000000:
            try:
                return json.loads(f.read_text())
            except (OSError, ValueError):
                pass
    return {}


def observation(path, capital, opening_date=None, annualization=365):
    for name in [
        "reference_equity.parquet",
        "equity_5m.parquet",
        "hourly_equity.parquet",
        "selected_equity.csv",
        "equity.csv",
    ]:
        p = path.parent / name
        if p.is_file():
            f = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            result = curve_stats(f, capital, opening_date, annualization)
            if result is not None:
                return {"path": str(p), "sha256": sha(p), **result}
    return None


def native_records(path, data, owner):
    result = []
    if not isinstance(data, dict) or not isinstance(data.get("strategy"), dict):
        return result
    for name, s in data["strategy"].items():
        if (
            not isinstance(s, dict)
            or not isinstance(s.get("trades"), list)
            or number(s.get("starting_balance")) is None
        ):
            continue
        capital = float(s["starting_balance"])
        final = float(s["final_balance"])
        reported = float(s["profit_total"])
        actual = final / capital - 1
        trades = s["trades"]
        fees = sorted({t.get("fee_open") for t in trades if t.get("fee_open") is not None})
        leverage = sorted({t.get("leverage", 1) for t in trades})
        errors = []
        commission = 0.0
        cash = 0.0
        quantity_residual = 0.0
        order_audit = None
        if trades and all(t.get("orders") for t in trades):
            for trade in trades:
                row = cash_from_orders(trade)
                errors.append(float(row["absolute_error"]))
                commission += float(row["commission"])
                cash += float(row["reconstructed_net_pnl"])
                quantity_residual = max(quantity_residual, abs(float(row["quantity_residual"])))
            order_audit = {
                "max_trade_cash_error_USDT": max(errors),
                "total_cash_error_USDT": abs(cash - float(s["profit_total_abs"])),
                "max_quantity_residual": quantity_residual,
                "passed": max(errors) < 1e-5 and abs(cash - float(s["profit_total_abs"])) < 1e-4,
            }
        return_error = abs(reported - actual)
        obs = observation(path, capital, s.get("backtest_start"))
        pairlist = s.get("pairlist", [])
        signature = hashlib.sha256(
            json.dumps(
                {
                    "strategy": name,
                    "capital": capital,
                    "period": [s.get("backtest_start"), s.get("backtest_end")],
                    "fee": fees,
                    "trades": trades,
                    "pairlist": pairlist,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        result.append(
            {
                "id": signature[:24],
                "identity_sha256": signature,
                "owner": owner,
                "source_path": str(path),
                "aliases": [],
                "strategy": name,
                "study": next(
                    (
                        p.name
                        for p in path.parents
                        if p.name.endswith(("v001", "v002", "v003", "v004"))
                    ),
                    path.parent.name,
                ),
                "kind": "native_account",
                "market": "crypto",
                "currency": s.get("stake_currency", "USDT"),
                "trading_mode": s.get("trading_mode", "unknown"),
                "timeframe": s.get("timeframe"),
                "engine_version": s.get("freqtrade_version") or data.get("freqtrade_version"),
                "start": s.get("backtest_start"),
                "end": s.get("backtest_end"),
                "capital": capital,
                "final_equity": final,
                "net_return": actual,
                "reported_return": reported,
                "return_identity_error": return_error,
                "reported_closed_trade_sharpe": s.get("sharpe"),
                "reported_closed_trade_drawdown": s.get("max_drawdown_account"),
                "reported_profit_factor": s.get("profit_factor"),
                "win_rate": s.get("wins", 0) / len(trades) if trades else None,
                "closed_trades": len(trades),
                "filled_orders": sum(len(t.get("orders", [])) for t in trades),
                "fees_per_side": fees,
                "leverage_values": leverage,
                "funding_model_cash": math.fsum(t.get("funding_fees", 0) or 0 for t in trades),
                "commission_and_cost_proxy": commission if errors else None,
                "open_trades_at_end": sum(bool(t.get("is_open")) for t in trades),
                "order_cash_audit": order_audit,
                "saved_curve": obs,
                "pairlist_count": len(pairlist),
                "pairlist_sha256": hashlib.sha256(
                    json.dumps(pairlist, sort_keys=True).encode()
                ).hexdigest(),
                "stake_amount": s.get("stake_amount"),
                "max_open_trades": s.get("max_open_trades"),
                "source_sha256": sha(path),
                "new_replay": False,
                "production_eligible": False,
            }
        )
    return result


def flat_account_record(path, data, owner):
    """Capacity ledgers and explicit sums of native wallets have flat result schemas."""
    is_sum = data.get("native_account_count", 1) > 1
    is_capacity = data.get("schema") == "crypto-trend-account.v1"
    if not (is_sum or is_capacity) or number(data.get("net_return")) is None:
        return None
    capital = float(data["initial_equity_usdt"])
    final = number(data.get("ending_equity"))
    if final is None:
        final = capital + float(data["net_profit_usdt"])
    obs = observation(path, capital, data.get("start"))
    order_audit = None
    if is_sum and (path.parent / "trade_ledger.json").is_file():
        trades = json.loads((path.parent / "trade_ledger.json").read_text())
        audit = [cash_from_orders(t) for t in trades if t.get("orders")]
        if len(audit) == len(trades) and audit:
            maximum = max(float(x["absolute_error"]) for x in audit)
            total = abs(
                math.fsum(float(x["reconstructed_net_pnl"]) for x in audit)
                - float(data["net_profit_usdt"])
            )
            order_audit = {
                "max_trade_cash_error_USDT": maximum,
                "total_cash_error_USDT": total,
                "passed": maximum < 1e-5 and total < 1e-4,
            }
    row = {
        "owner": owner,
        "source_path": str(path),
        "aliases": [],
        "strategy": data["candidate"],
        "study": next(
            (p.name for p in path.parents if p.name.endswith(("v001", "v002", "v003", "v004"))),
            path.parents[2].name,
        ),
        "kind": "native_wallet_sum" if is_sum else "custom_account_with_curve",
        "market": "crypto",
        "currency": "USDT",
        "trading_mode": "independent_native_wallets" if is_sum else "custom_crypto_account",
        "start": data["start"],
        "end": data["end"],
        "capital": capital,
        "final_equity": final,
        "net_return": float(data["net_return"]),
        "return_identity_error": abs(final / capital - 1 - float(data["net_return"])),
        "reported_full_drawdown": pick(data, "hourly_observed_drawdown", "max_drawdown"),
        "reported_daily_sharpe": data.get("daily_sharpe"),
        "win_rate": data.get("win_rate"),
        "closed_trades": data.get("closed_trades"),
        "fees_per_side": [0.0005 + float(data["extra_cost_bps_per_side"]) / 10000]
        if is_sum
        else [0.0005],
        "slippage_bps": None if is_sum else data["extra_cost_bps_per_side"],
        "leverage_values": sorted(
            {z for c in data.get("components", []) for z in c.get("actual_leverage_values", [])}
        )
        if is_sum
        else [],
        "gross_budget": None,
        "observed_max_gross": data.get("maximum_gross_to_equity"),
        "saved_curve": obs,
        "phase": data.get("phase"),
        "risk_halted": data.get("account_halted"),
        "order_cash_audit": order_audit,
        "funding_model_cash": data.get("funding_net_usdt"),
        "native_account_count": data.get("native_account_count", 1),
        "open_positions_at_end": data.get("open_positions"),
        "source_sha256": sha(path),
        "new_replay": False,
        "production_eligible": False,
    }
    identity = {
        k: v
        for k, v in row.items()
        if k not in ["owner", "source_path", "aliases", "source_sha256"]
    }
    if obs:
        identity["saved_curve"] = {k: v for k, v in obs.items() if k != "path"}
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    row.update(id=signature[:24], identity_sha256=signature)
    return row


def custom_record(path, data, owner):
    if not isinstance(data, dict):
        return None
    flat = flat_account_record(path, data, owner)
    if flat:
        return flat
    m = data.get("metrics")
    config = data.get("config", {})
    window = data.get("window", {})
    equity_schema = str(data.get("schema", "")).startswith("ta-equity-account.")
    if equity_schema:
        m = data.get("account", {}).get("metrics", {})
    if (
        not isinstance(m, dict)
        or number(m.get("net_return")) is None
        or not isinstance(config, dict)
        or not isinstance(window, dict)
    ):
        return None
    if not config and not equity_schema:
        return None
    capital = number(pick(config, "capital", "initial_equity_usdt", "initial_cash")) or number(
        data.get("initial_cash")
    )
    if (
        capital is None
        and number(m.get("ending_equity")) is not None
        and number(m.get("net_pnl")) is not None
    ):
        capital = float(m["ending_equity"]) - float(m["net_pnl"])
    final = number(pick(m, "ending_equity", "final_equity"))
    obs = observation(path, capital, None, 252 if equity_schema else 365)
    phase = data.get("phase") or window.get("phase")
    record = {
        "owner": owner,
        "source_path": str(path),
        "aliases": [],
        "strategy": data.get("candidate") or data.get("mode") or path.parent.name,
        "study": next(
            (
                p.name
                for p in path.parents
                if p.name.startswith(
                    (
                        "factor_lab_",
                        "factor_discovery_",
                        "github_factor_",
                        "forecast_",
                        "regime_",
                        "sparse_",
                        "equities_",
                        "max_",
                        "monthly_",
                    )
                )
            ),
            path.parents[1].name,
        ),
        "kind": "equity_account"
        if equity_schema
        else "custom_account_with_curve"
        if obs
        else "custom_report_only",
        "market": data.get("market", "crypto"),
        "currency": data.get("currency", "USDT"),
        "trading_mode": "cash_equity" if equity_schema else "custom_crypto_account",
        "timeframe": "1d" if equity_schema else "5m_reference",
        "start": pick(window, "start", "start_at"),
        "end": pick(window, "end", "end_at"),
        "capital": capital,
        "final_equity": final,
        "net_return": float(m["net_return"]),
        "return_identity_error": abs(final / capital - 1 - float(m["net_return"]))
        if final is not None and capital
        else None,
        "reported_full_drawdown": pick(
            m, "max_drawdown", "maximum_close_drawdown", "max_continuous_close_drawdown"
        ),
        "reported_daily_sharpe": pick(m, "sharpe", "sharpe_zero_rf_252"),
        "win_rate": m.get("win_rate"),
        "closed_trades": pick(m, "closed_trades", "complete_cycles", "complete_lifecycles"),
        "fees_per_side": [float(config["fee_bps"]) / 10000]
        if number(config.get("fee_bps")) is not None
        else [],
        "slippage_bps": pick(config, "slippage_bps", "entry_slippage_buffer_bps")
        if not equity_schema
        else data.get("slippage_bps_per_side"),
        "leverage_values": [],
        "gross_budget": config.get("gross"),
        "observed_max_gross": m.get("max_gross_exposure"),
        "saved_curve": obs,
        "phase": phase,
        "risk_halted": data.get("risk_halted"),
        "economic_account_evaluated": data.get("economic_account_evaluated"),
        "sample_scope": m.get("sampled_return_frequency"),
        "source_sha256": sha(path),
        "new_replay": False,
        "production_eligible": False,
    }
    identity_record = {
        k: v
        for k, v in record.items()
        if k not in ["owner", "source_path", "aliases", "source_sha256", "study"]
    }
    if obs:
        identity_record["saved_curve"] = {k: v for k, v in obs.items() if k != "path"}
    signature = hashlib.sha256(
        json.dumps(
            identity_record,
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    record["id"] = signature[:24]
    record["identity_sha256"] = signature
    return record


def financial_nodes(value, pointer=""):
    if isinstance(value, dict):
        if any(
            number(value.get(k)) is not None for k in ["net_return", "net_return_pct", "return_pct"]
        ):
            yield pointer, value
        for key, item in value.items():
            if key not in [
                "curve",
                "equity_curve",
                "trades",
                "orders",
                "events",
                "positions",
                "targets",
                "config",
                "protocol",
                "source_hashes",
            ] and isinstance(item, (dict, list)):
                yield from financial_nodes(item, pointer + "/" + key)
    elif isinstance(value, list) and len(value) < 500:
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                yield from financial_nodes(item, pointer + "/" + str(i))


def normalize(inventory_path, output, additional_root=None):
    output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(inventory_path.read_text())
    accounts = {}
    summaries = []
    issues = []
    native_parent = set()
    unrecognized = []
    candidates = [
        x for x in inventory["files"] if x["type"] == "native_zip" or x.get("has_account_markers")
    ]
    if additional_root:
        candidates.extend(
            {"type": "native_zip", "path": str(p.resolve()), "owner": "Freqtrade"}
            for p in sorted(additional_root.rglob("backtest-result-*.zip"))
        )
    for item in candidates:
        path = Path(item["path"])
        try:
            if item["type"] == "native_zip":
                with zipfile.ZipFile(path) as archive:
                    names = [
                        i
                        for i in archive.infolist()
                        if i.filename.endswith(".json")
                        and not i.filename.endswith("_config.json")
                        and i.file_size < 128 * 1024 * 1024
                    ]
                    records = []
                    for info in names:
                        records.extend(
                            native_records(path, json.loads(archive.read(info)), item["owner"])
                        )
            else:
                data = json.loads(path.read_text())
                records = native_records(path, data, item["owner"])
                if not records:
                    r = custom_record(path, data, item["owner"])
                    records = [r] if r else []
                    for pointer, node in financial_nodes(data):
                        raw = number(node.get("net_return"))
                        raw = (
                            raw
                            if raw is not None
                            else number(pick(node, "net_return_pct", "return_pct")) / 100
                        )
                        summaries.append(
                            {
                                "path": str(path),
                                "pointer": pointer,
                                "owner": item["owner"],
                                "reported_net_return": raw,
                                "classified_as_account": bool(r),
                                "source_sha256": item.get("sha256"),
                            }
                        )
            for record in records:
                if additional_root and path.resolve().is_relative_to(additional_root.resolve()):
                    record["new_replay"] = True
                if record["id"] in accounts:
                    accounts[record["id"]]["aliases"].append(str(path))
                    continue
                accounts[record["id"]] = record
                if record["kind"] == "native_account":
                    native_parent.add(str(path.parent))
            if not records:
                unrecognized.append(str(path))
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            issues.append({"path": str(path), "error": str(exc)[:220]})
    value = {
        "schema": "existing-strategy-normalized-accounts.v1",
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha(inventory_path),
        "accounts": list(accounts.values()),
        "reported_summary_nodes": summaries,
        "unrecognized_account_marker_files": unrecognized,
        "issues": issues,
        "new_strategies": 0,
        "new_replays": 0,
        "orders_enabled": False,
    }
    (output / "accounts.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    rows = list(accounts.values())
    print(
        json.dumps(
            {
                "accounts": len(rows),
                "kinds": Counter(r["kind"] for r in rows),
                "market": Counter(r["market"] for r in rows),
                "native_strategy_classes": len(
                    {r["strategy"] for r in rows if r["kind"] == "native_account"}
                ),
                "summary_nodes": len(summaries),
                "issues": issues[:8],
                "top_native": [
                    {
                        k: r[k]
                        for k in [
                            "strategy",
                            "start",
                            "end",
                            "net_return",
                            "capital",
                            "leverage_values",
                            "fees_per_side",
                        ]
                    }
                    for r in sorted(
                        (r for r in rows if r["kind"] == "native_account"),
                        key=lambda r: r["net_return"],
                        reverse=True,
                    )[:7]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--additional-root", type=Path)
    a = p.parse_args()
    normalize(a.inventory, a.output, a.additional_root)
