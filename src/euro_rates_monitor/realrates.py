"""Simple real rates: nominal yield minus an inflation measure.

Finance notes
-------------
A real rate is the nominal yield minus *expected* inflation over the bond's life. The
market measure is an inflation swap or a linker breakeven; neither is free for the
euro area. So this module offers two clearly labelled proxies:

* Ex-post (realised): 10Y AAA yield minus the latest published HICP y/y. Realised
  inflation is backward-looking and noisy (energy, base effects); it is not an
  expectation and can make the "real rate" swing for reasons unrelated to rates.
* Survey-based: 10Y AAA yield minus the ECB Survey of Professional Forecasters'
  longer-term inflation expectation. Forward-looking, but a survey of economists
  published quarterly, not a market price.

The US side has a market measure (TIPS real yield, DFII10, and breakevens) which we
show for comparison; the asymmetry is a data limitation, not a finding.

Publication timing: a month's HICP is usable only after release (flash estimate at
the start of the following month). We treat a month's value as known from the first
day of the next month, and an SPF round as known from the first day of its survey
quarter (the survey runs in the first weeks of the quarter). Both are approximations.
"""

from __future__ import annotations

import pandas as pd


def available_from(monthly: pd.Series, lag: pd.DateOffset) -> pd.Series:
    """Re-stamp a period-start-indexed series at the date its value became known."""
    out = monthly.copy()
    out.index = out.index + lag
    return out


def real_rate(nominal: pd.Series, inflation_known: pd.Series) -> pd.Series:
    """Nominal (daily) minus the most recent inflation value known on each day."""
    infl = inflation_known.reindex(nominal.index.union(inflation_known.index)).ffill()
    return (nominal - infl.reindex(nominal.index)).dropna()


def us_cpi_yoy(cpi_index: pd.Series) -> pd.Series:
    """US CPI year-on-year % from the monthly index."""
    return (cpi_index / cpi_index.shift(12) - 1).dropna() * 100
