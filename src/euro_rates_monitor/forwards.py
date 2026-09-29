"""Implied forward rates and the policy path "priced" by the curve.

Finance notes
-------------
With continuously compounded zero rates, investing to t2 must earn the same as
investing to t1 and rolling at the forward rate f(t1, t2) (no arbitrage):

    exp(r2*t2) = exp(r1*t1) * exp(f*(t2 - t1))   =>   f = (r2*t2 - r1*t1) / (t2 - t1)

A string of 3-month forwards (0-3M, 3-6M, ..., 21-24M) is the curve's implied path for
the 3-month rate. Desks read that as "what the market is pricing" for policy. Two
reasons it is not a clean expectation of the ECB deposit rate:

1. Term premium. Forwards = expected short rate + compensation for bearing duration
   risk. The premium is unobservable, varies over time, is not linear in horizon and
   has been estimated as negative in some periods. We show forwards minus k*h for
   k = 0, 10, 20 bp per year of horizon h purely as a sensitivity illustration.
2. Instrument basis. These forwards come from the AAA *government bond* curve, not
   from EUR STR swaps (OIS), which is what desks use and which is not freely available.
   Government bills/bonds can trade away from EUR STR (collateral scarcity, supply).
   We report the gap between the curve's 3M rate and EUR STR, and the path *relative
   to the curve's own 3M rate*, which removes a constant basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .curve import spot_from_params

TP_SCENARIOS_BP_PER_YEAR = (0, 10, 20)


def forward_rate(r1: float | np.ndarray, t1: float | np.ndarray,
                 r2: float | np.ndarray, t2: float | np.ndarray) -> np.ndarray:
    """Continuously compounded forward rate between t1 and t2 (years), same units as r.

    t1 = 0 returns r2 itself (the spot rate is the forward from today).
    """
    t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
    if np.any(t2 <= t1):
        raise ValueError("t2 must be greater than t1")
    return (np.asarray(r2) * t2 - np.asarray(r1) * t1) / (t2 - t1)


def forward_path(params: pd.Series, horizon_years: float = 2.0,
                 tenor: float = 0.25) -> pd.DataFrame:
    """3-month forward rates starting every quarter out to `horizon_years`.

    Row h gives the rate for the period [h, h + tenor]. The first row (h = 0) is the
    3M spot rate. Spot rates come from the NSS parameters, so no interpolation.
    """
    starts = np.round(np.arange(0.0, horizon_years + 1e-9, tenor), 10)
    ends = starts + tenor
    r_end = spot_from_params(params, ends)
    r_start = np.where(starts > 0, spot_from_params(params, np.maximum(starts, 1e-6)), 0.0)
    fwd = forward_rate(r_start, starts, r_end, ends)
    return pd.DataFrame({"start_years": starts, "end_years": ends, "forward": fwd})


def term_premium_scenarios(path: pd.DataFrame,
                           ks: tuple[int, ...] = TP_SCENARIOS_BP_PER_YEAR) -> pd.DataFrame:
    """Forward minus an assumed term premium of k bp per year of horizon.

    Illustration only: the true premium is neither linear nor constant and can be
    negative. The k = 0 column is the raw forward and remains the headline.
    """
    out = path.copy()
    for k in ks:
        out[f"expected_k{k}"] = path["forward"] - k / 100 * path["start_years"]
    return out


def priced_change(path: pd.DataFrame, anchor: float) -> pd.Series:
    """Forward path minus an anchor rate, in bp (e.g. anchor = current deposit rate)."""
    return (path["forward"] - anchor) * 100


def ecb_pricing_summary(path: pd.DataFrame, dfr: float, estr: float) -> dict[str, float]:
    """Headline numbers for the note: level and change priced at 6M, 1Y, 2Y horizons.

    * fwd_{h}: 3M forward starting in h, % (raw, no term-premium adjustment).
    * vs_dfr_{h}_bp: that forward minus today's deposit rate. Includes the basis.
    * vs_3m_{h}_bp: that forward minus today's 3M rate on the same curve. Removes a
      constant government-vs-EUR STR basis, keeps the term premium.
    * basis_3m_estr_bp: today's AAA 3M spot minus EUR STR.
    """
    idx = path.set_index("start_years")["forward"]
    spot3m = float(idx.loc[0.0])
    out: dict[str, float] = {"spot_3m": spot3m, "basis_3m_estr_bp": (spot3m - estr) * 100,
                             "estr_minus_dfr_bp": (estr - dfr) * 100}
    for label, h in [("6m", 0.5), ("1y", 1.0), ("2y", 2.0)]:
        f = float(idx.loc[h])
        out[f"fwd_{label}"] = f
        out[f"vs_dfr_{label}_bp"] = (f - dfr) * 100
        out[f"vs_3m_{label}_bp"] = (f - spot3m) * 100
    return out
