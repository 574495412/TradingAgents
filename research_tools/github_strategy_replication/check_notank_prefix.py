"""Test a reviewed source expression on one synthetic prefix; no strategy replay."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from scipy.signal import argrelextrema


def branch(close):
    # Same three expressions as current source lines332,354-355,378-383.
    # Whole module, callbacks, network and financial execution are not loaded.
    minima = np.zeros(len(close))
    minima[argrelextrema(close, np.less, order=5)] = 1
    check = pd.Series(minima).rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
    return minima, check, (check == 0) & (talib.RSI(close) < 30)


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(
            {
                "kind": "one_synthetic_counterexample_not_economic_account",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "prefix": "250 strictly descending closes,1000 to751",
                "future": "five closes752..756",
                "copied_reviewed_expressions": [
                    "argrelextrema(close,np.less,order=5)",
                    "minima rolling4 all-not-one",
                    "minima_check==0 and volume>0 and RSI<30",
                ],
                "whole_source_execution": False,
                "optimization_allowed": False,
            },
            indent=2,
        )
        + "\n"
    )
    prefix = 1000.0 - np.arange(250)
    extended = np.r_[prefix, np.arange(752.0, 757.0)]
    assert np.array_equal(prefix, extended[: len(prefix)])
    a, b, c = branch(prefix)
    x, y, z = branch(extended)
    at = len(prefix) - 1
    result = {
        "schema": "notank-current-source-prefix-counterexample.v1",
        "decision_index": at,
        "decision_close": float(prefix[at]),
        "unchanged_information_through_decision": True,
        "prefix_minima": float(a[at]),
        "with_future_minima": float(x[at]),
        "prefix_minima_check": float(b.iloc[at]),
        "with_future_minima_check": float(y.iloc[at]),
        "prefix_minima_full_send_branch": bool(c.iloc[at]),
        "with_future_minima_full_send_branch": bool(z.iloc[at]),
        "causal_invariance_violated": bool(c.iloc[at]) != bool(z.iloc[at]),
        "current_source_only_not_original_report_identity": True,
        "historical_PnL_evaluated": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    if not result["causal_invariance_violated"]:
        raise ValueError("Constructed counterexample did not reproduce")
    (output / "counterexample.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run(a.source, a.output)
