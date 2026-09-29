"""Weekly and monthly change logic across holidays and missing days."""

import numpy as np
import pandas as pd
import pytest

from euro_rates_monitor.changes import WINDOWS, change_over, change_table


def _series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), name="x")


def test_plain_week():
    s = _series(["2026-03-02", "2026-03-06", "2026-03-09"], [1.0, 1.5, 2.0])
    ch = change_over(s, WINDOWS["1w"])
    assert ch.base_date == pd.Timestamp("2026-03-02")
    assert ch.change == pytest.approx(1.0)


def test_base_day_is_a_holiday_uses_previous_business_day():
    # Latest Mon 13 Apr 2026. One week earlier is Easter Monday 6 Apr (TARGET2 closed);
    # Good Friday 3 Apr is also closed, so the base is Thu 2 Apr.
    s = _series(["2026-04-01", "2026-04-02", "2026-04-07", "2026-04-13"], [3.0, 3.1, 3.2, 3.5])
    ch = change_over(s, WINDOWS["1w"])
    assert ch.base_date == pd.Timestamp("2026-04-02")
    assert ch.change == pytest.approx(0.4)


def test_latest_day_missing_value_is_skipped():
    s = _series(["2026-03-02", "2026-03-06", "2026-03-09"], [1.0, 1.5, np.nan])
    ch = change_over(s, WINDOWS["1w"])
    assert ch.latest_date == pd.Timestamp("2026-03-06")
    assert ch.base_date is None or ch.base_date <= pd.Timestamp("2026-02-27")


def test_long_gap_returns_missing_not_a_longer_change():
    s = _series(["2026-01-05", "2026-03-09"], [1.0, 2.0])
    ch = change_over(s, WINDOWS["1w"], max_stale_days=5)
    assert ch.change is None


def test_month_uses_calendar_month_end():
    # 31 Mar minus one month is 28 Feb in 2026 (not a leap year).
    s = _series(["2026-02-27", "2026-03-31"], [2.0, 2.3])
    ch = change_over(s, WINDOWS["1m"])
    assert ch.base_date == pd.Timestamp("2026-02-27")   # 28 Feb is a Saturday
    assert ch.change == pytest.approx(0.3)


def test_mixed_holiday_calendars_in_one_table():
    # EUR open on US Thanksgiving (26 Nov 2026), US open on no EUR-only holiday here.
    idx = pd.to_datetime(["2026-11-19", "2026-11-20", "2026-11-26", "2026-11-27"])
    frame = pd.DataFrame({"eur": [1.00, 1.01, 1.05, 1.06],
                          "usd": [4.00, 4.02, np.nan, 4.10]}, index=idx)
    table = change_table(frame, scale={"eur": 100, "usd": 100})
    assert table.loc["eur", "chg_1w"] == pytest.approx(5.0)    # 27 Nov vs 20 Nov
    assert table.loc["usd", "chg_1w"] == pytest.approx(8.0)    # 27 Nov vs 20 Nov


def test_end_parameter_evaluates_in_the_past():
    s = _series(["2026-03-02", "2026-03-09", "2026-03-16"], [1.0, 2.0, 5.0])
    ch = change_over(s, WINDOWS["1w"], end=pd.Timestamp("2026-03-10"))
    assert ch.latest_date == pd.Timestamp("2026-03-09")
    assert ch.change == pytest.approx(1.0)
