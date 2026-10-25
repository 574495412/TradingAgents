"""Immutable resized-account publication including resumable state."""

from research_tools.factor_lab.io import save, sha


def write_account(folder, a, config, window):
    folder.mkdir(parents=True, exist_ok=False)
    a["curve"].to_parquet(folder / "equity_5m.parquet", index=False)
    for name in ("events", "trades", "requests", "positions", "pending_exits", "resume_state"):
        save(folder / (name + ".json"), a[name])
    nav = a["curve"].set_index("date").equity
    dd = 1 - nav / nav.cummax()
    samples = nav.loc[nav.index.minute == 0]
    result = {
        "metrics": a["metrics"],
        "capacity": a["capacity"],
        "config": config,
        "window": window,
        "curve": [
            {"time": t.isoformat(), "equity": float(v), "drawdown": float(dd.loc[t])}
            for t, v in samples.items()
        ],
        "trades": a["trades"],
        "open_positions": a["positions"],
        "risk_halted": a["risk_halted"],
        "engine": a["engine"],
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "source_hashes": {p.name: sha(p) for p in folder.iterdir()},
    }
    save(folder / "result.json", result)
