import io
import zipfile

import pandas as pd
import pytest

from research_tools.carry_feasibility.prepare import cost_arithmetic, expected_clock, parse_spot


def archive(month, unit, omit_last=False):
    start = pd.Timestamp(month + "-01", tz="UTC")
    index = expected_clock(start, start + pd.offsets.MonthBegin())
    if omit_last:
        index = index[:-1]
    opening = index.as_unit(unit).asi8
    closing = (index + pd.Timedelta(minutes=5) - pd.Timedelta(1, unit=unit)).as_unit(unit).asi8
    frame = pd.DataFrame(
        {
            0: opening,
            1: 100,
            2: 102,
            3: 98,
            4: 101,
            5: 2,
            6: closing,
            7: 201,
            8: 2,
            9: 1,
            10: 100,
            11: 0,
        }
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as stream:
        stream.writestr("fixture.csv", frame.to_csv(header=False, index=False))
    return output.getvalue()


@pytest.mark.parametrize("month,unit", [("2024-12", "ms"), ("2025-01", "us")])
def test_archive_time_transition_preserves_actual_month_and_rejects_gaps(month, unit):
    frame = parse_spot(archive(month, unit), month)
    assert len(frame) == 31 * 288
    assert frame.index[0] == pd.Timestamp(month + "-01", tz="UTC")
    with pytest.raises(ValueError, match="missing/duplicate"):
        parse_spot(archive(month, unit, omit_last=True), month)


def test_api_millisecond_sample_cannot_be_parsed_as_2025_spot_archive():
    with pytest.raises(ValueError, match="wrong-unit"):
        parse_spot(archive("2025-01", "ms"), "2025-01")


def test_round_trip_includes_both_markets_and_both_entry_and_exit():
    scenario = {
        "id": "stress",
        "spot_commission_bps": 10,
        "perpetual_commission_bps": 5,
        "slippage_bps_each_leg_each_side": 10,
    }
    result = cost_arithmetic(
        scenario, {"spot_leg_nav_fraction": 0.4, "perpetual_leg_nav_fraction": 0.4}
    )
    # Spot20bps and perp15bps each way, both entry and exit on40%NAV legs.
    assert result["round_trip_cost_on_initial_nav"] == pytest.approx(0.0028)
    assert result["required_sum_funding_rates_on_short_notional"] == pytest.approx(0.007)
    assert result["executed_account"] is False
