"""One source-bound dated-carry opportunity and one frozen exit-policy contrast."""

import pandas as pd

from factors.common.dated_carry import basis_reference


def schedule(spot, future, plan, profile):
    if profile not in plan["profiles"]:
        raise ValueError("Unknown registered dated-carry policy")
    begin = pd.Timestamp(plan["window"]["start"])
    previous = begin - pd.Timedelta(minutes=5)
    fee = plan["account"]
    threshold = (
        1.25
        * 2
        * (fee["spot_fee_bps"] + fee["perpetual_fee_bps"] + 2 * max(plan["costs_slippage_bps"]))
        / 10000
    )
    entries, audit = {}, []
    for s, c in sorted(plan["contracts"].items()):
        if begin < pd.Timestamp(c["known_at"]):
            raise ValueError("Before declared contract information availability")
        ref = basis_reference(
            spot[s].loc[previous, "close"],
            future[s].loc[previous, "close"],
            spot_asset=c["base"],
            future_asset=c["base"],
            spot_quote="USDT",
            future_quote=c["quote"],
            contract_style=c["style"],
        )
        active = ref["simple_basis_total"] > threshold
        entries[s] = {"entry": active, "exit": False}
        audit.append(
            {
                "asset": s,
                "decision": begin.isoformat(),
                "source_opened_at": previous.isoformat(),
                "reference_basis_total": ref["simple_basis_total"],
                "cost_coverage_threshold": threshold,
                "eligible_before_liquidity_or_execution": active,
                "annualized_profit_or_cash_forecast": False,
            }
        )
    signals = [{"time": begin.isoformat(), "assets": entries}]
    if profile == "early_exit_24h":
        clock = pd.Timestamp(next(iter(plan["contracts"].values()))["expiry"]) - pd.Timedelta(
            days=1
        )
        signals.append(
            {
                "time": clock.isoformat(),
                "assets": {
                    s: {"entry": False, "exit": True, "exit_reason": "fixed_24h_pre_delivery_exit"}
                    for s in entries
                },
            }
        )
    return signals, audit
