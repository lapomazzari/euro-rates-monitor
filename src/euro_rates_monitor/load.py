"""Parse the latest cached raw file for each series into tidy pandas objects.

Conventions: every daily/monthly series comes back as a float Series indexed by a
DatetimeIndex, values in the provider's unit (percent for rates). Curves come back as
DataFrames with one column per maturity in years.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from . import paths
from .sources import BY_NAME, PAR_TENORS, SPOT_TENORS


class MissingDataError(RuntimeError):
    """Raised when a series has no cached file. Run `erm fetch` first."""


def latest_file(name: str) -> Path:
    """Most recent cached file for a catalogue entry (file names sort by date)."""
    series = BY_NAME[name]
    files = sorted((paths.RAW / series.source.lower()).glob(f"{name}_*.csv"))
    if not files:
        raise MissingDataError(f"no cached data for {name}; run `erm fetch`")
    return files[-1]


def retrieval_date(name: str) -> str:
    """Retrieval date encoded in the latest cache file name (YYYY-MM-DD)."""
    return latest_file(name).stem.rsplit("_", 1)[1]


def _ecb_long(name: str) -> pd.DataFrame:
    df = pd.read_csv(latest_file(name))
    return df[["KEY", "TIME_PERIOD", "OBS_VALUE"]]


def _ecb_period_index(periods: pd.Series) -> pd.DatetimeIndex:
    # Daily 'YYYY-MM-DD', monthly 'YYYY-MM', quarterly 'YYYY-Qn'. Monthly and quarterly
    # observations are stamped at the *start* of the period here; callers decide when
    # the value became known (see realrates.py for publication lags).
    return pd.DatetimeIndex(pd.PeriodIndex(periods, freq=_freq(periods.iloc[0])).start_time)


def _freq(sample: str) -> str:
    if "Q" in sample:
        return "Q"
    return "D" if len(sample) == 10 else "M"


def ecb_series(name: str) -> pd.Series:
    """A single ECB series as a float Series."""
    df = _ecb_long(name)
    out = pd.Series(df["OBS_VALUE"].astype(float).to_numpy(),
                    index=_ecb_period_index(df["TIME_PERIOD"]), name=name)
    return out.sort_index().dropna()


def ecb_wide(name: str) -> pd.DataFrame:
    """A multi-series ECB file pivoted to one column per final key item (e.g. SR_10Y)."""
    df = _ecb_long(name)
    df = df.assign(item=df["KEY"].str.rsplit(".", n=1).str[1],
                   date=pd.to_datetime(df["TIME_PERIOD"]))
    wide = df.pivot(index="date", columns="item", values="OBS_VALUE").astype(float)
    wide.index.name = None
    wide.columns.name = None
    return wide.sort_index()


def ecb_curve(name: str) -> pd.DataFrame:
    """ECB spot or par curve with columns renamed to maturities in years, sorted."""
    wide = ecb_wide(name)
    mapping = SPOT_TENORS if name.startswith("ecb_spot") else PAR_TENORS
    wide = wide.rename(columns=mapping)[sorted(mapping.values())]
    return wide.dropna(how="all")


def fred_series(name: str) -> pd.Series:
    """FRED series; the API and CSV paths are cached in the same two-column layout."""
    df = pd.read_csv(latest_file(name), na_values=["."])
    out = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                    index=pd.to_datetime(df.iloc[:, 0]), name=name)
    return out.sort_index().dropna()


def gsw() -> pd.DataFrame:
    """Fed Board GSW file: skip the free-text preamble, keep zero yields and parameters."""
    text = latest_file("fed_gsw").read_text()
    start = text.index("\nDate,") + 1
    df = pd.read_csv(io.StringIO(text[start:]), na_values=["NA"])
    df.index = pd.to_datetime(df.pop("Date"))
    keep = [c for c in df.columns if c.startswith("SVENY")]
    keep += ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    return df[keep].sort_index()
