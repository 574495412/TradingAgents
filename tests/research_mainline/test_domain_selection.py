"""Freeze conditional research followups using only registered screen evidence."""

import pytest

from research_tools.regime_program.run_domains import select_followups


def cases(name, n, expectancy, stress_return=0.01):
    return [
        {
            "candidate": name,
            "kind": "module",
            "phase": "screen2025",
            "slippage_bps": c,
            "metrics": {
                "net_return": 0.02 if c == 3 else stress_return,
                "closed_net_pnl": n * expectancy,
                "closed_trades": n,
                "expectancy": expectancy,
            },
        }
        for c in (3, 10)
    ]


def test_future_return_and_reference_accounts_do_not_change_frozen_followups():
    rows = cases("A", 40, 1) + cases("B", 30, 100) + cases("C", 40, 2) + cases("D", 100, 1, -0.01)
    expected = select_followups(rows, ["A", "B", "C", "D"])
    assert expected == ["C", "A", "B"]
    rows += [
        {**r, "phase": "diagnostic2026", "metrics": {**r["metrics"], "net_return": 1000}}
        for r in cases("D", 1000, 1000)
    ]
    rows += [{**r, "kind": "control"} for r in cases("A", 1000, 1000)]
    assert select_followups(rows, ["A", "B", "C", "D"]) == expected


def test_no_positive_pair_of_cost_cases_means_no_forced_winner():
    assert select_followups(cases("A", 10, 1, -0.01), ["A"]) == []
    with pytest.raises(ValueError, match="Incomplete"):
        select_followups(cases("A", 10, 1)[:1], ["A"])
