"""Weekly and monthly changes that are robust to holidays and missing days.

Rule: the change over a window is latest value minus the last available value on or
before (latest date - window). Calendar offsets, not observation counts, so a week is
a week even when TARGET2 or US holidays remove days, and the euro and US sides compare
like with like despite different holiday calendars.

If the base value is older than `max_stale_days` before the target date (a long data
gap), the change is reported as missing rather than silently spanning a longer period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WINDOWS: dict[str, pd.DateOffset] = {
    "1w": pd.DateOffset(weeks=1),
    "1m": pd.DateOffset(months=1),
}


@dataclass(frozen=True)
class Change:
    """Latest value, the base value it is compared to, and the difference."""

    latest_date: pd.Timestamp
    latest: float
    base_date: pd.Timestamp | None
    base: float | None
    change: float | None


def value_asof(s: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Last non-missing observation on or before `when`."""
    s = s.dropna()
    s = s[s.index <= when]
    if s.empty:
        return None
    return s.index[-1], float(s.iloc[-1])


def change_over(s: pd.Series, window: pd.DateOffset, max_stale_days: int = 5,
                end: pd.Timestamp | None = None) -> Change:
    """Change of `s` over a calendar window ending at its latest observation.

    end: evaluate as of this date instead of the latest observation (the latest value
    is then the last observation on or before `end`).
    """
    clean = s.dropna()
    if end is not None:
        clean = clean[clean.index <= end]
    if clean.empty:
        raise ValueError(f"series {s.name!r} has no observations")
    latest_date, latest = clean.index[-1], float(clean.iloc[-1])
    target = latest_date - window
    found = value_asof(clean, target)
    if found is None or (target - found[0]).days > max_stale_days:
        return Change(latest_date, latest, None, None, None)
    base_date, base = found
    return Change(latest_date, latest, base_date, base, latest - base)


def change_table(frame: pd.DataFrame, scale: dict[str, float] | None = None,
                 max_stale_days: int = 5) -> pd.DataFrame:
    """1w and 1m changes for every column of `frame`.

    scale: optional multiplier per column for the change (e.g. 100 to express a
    change in a percent-valued yield in bp). Levels are left in original units.
    """
    scale = scale or {}
    rows = []
    for col in frame.columns:
        row: dict[str, object] = {"measure": col}
        for label, window in WINDOWS.items():
            ch = change_over(frame[col], window, max_stale_days=max_stale_days)
            row["latest_date"], row["latest"] = ch.latest_date, ch.latest
            row[f"chg_{label}"] = None if ch.change is None else ch.change * scale.get(col, 1.0)
            row[f"base_date_{label}"] = ch.base_date
        rows.append(row)
    return pd.DataFrame(rows).set_index("measure")
