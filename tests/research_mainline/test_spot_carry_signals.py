import pandas as pd

from factors.crypto.spot_perpetual_carry import schedule


def test_settlement_delay_and_future_edits_do_not_change_past_decision():
    clocks = pd.to_datetime(["2025-01-01T00:00Z", "2025-01-06T00:00Z"], utc=True)
    index = pd.date_range("2024-12-01", "2025-01-07", freq="8h", tz="UTC")
    funding = {"X": pd.DataFrame({"funding_rate": 0.0001}, index=index)}
    c = {
        "availability_delay_minutes": 5,
        "lookback_days": 28,
        "exit_negative_funding_days": 7,
        "forecast_cost_recovery_days": 60,
        "cost_coverage_buffer": 1.25,
    }
    original = schedule(funding, clocks, c, "2024-12-01T00:00Z", "cost_guarded")
    funding["X"].loc[index >= clocks[0], "funding_rate"] = -0.5
    changed = schedule(funding, clocks, c, "2024-12-01T00:00Z", "cost_guarded")
    assert original[0] == changed[0]
    assert (
        pd.Timestamp(original[0]["assets"]["X"]["last_observable_event"]) + pd.Timedelta(minutes=5)
        <= clocks[0]
    )
    assert changed[1]["assets"]["X"]["entry"] is False
    assert changed[1]["assets"]["X"]["exit"] is True
