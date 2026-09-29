"""Curve construction and the standard summary measures.

Finance notes
-------------
* The ECB and the Fed Board (GSW) both publish Nelson-Siegel-Svensson (NSS) fits. The
  parameters define the zero-coupon ("spot") rate at *any* maturity, so we can
  evaluate the curve on a fine grid instead of interpolating between published points.
* Both publish continuously compounded zero rates: a zero-coupon bond paying 1 at T
  costs exp(-z(T) * T). This is what makes the forward maths in forwards.py exact.
* Level, slope and curvature are the three words a rates desk uses to describe a
  curve. Conventions used here (all from the AAA spot curve):
    level      = 10Y spot, in %
    2s10s      = 10Y - 2Y, in bp (positive = upward sloping, "steep")
    10s30s     = 30Y - 10Y, in bp
    2s5s10s    = 2*5Y - 2Y - 10Y, in bp: a butterfly. Positive = the 5Y "belly"
                 yields more than the average of the wings, i.e. the belly is cheap
                 relative to the wings / the curve is humped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def svensson_spot(t: np.ndarray | float, b0: float, b1: float, b2: float, b3: float,
                  tau1: float, tau2: float) -> np.ndarray:
    """NSS zero rate z(t) in the units of the betas (percent here), t in years.

    z(t) = b0 + b1*L(t/tau1) + b2*[L(t/tau1) - exp(-t/tau1)] + b3*[L(t/tau2) - exp(-t/tau2)]
    with L(x) = (1 - exp(-x)) / x. b0 is the long-run level, b1 sets the short end
    (z(0) = b0 + b1), b2 and b3 add humps at horizons governed by tau1 and tau2.
    """
    t = np.asarray(t, dtype=float)
    x1, x2 = t / tau1, t / tau2
    load1 = (1 - np.exp(-x1)) / x1
    load2 = (1 - np.exp(-x2)) / x2
    return (b0 + b1 * load1 + b2 * (load1 - np.exp(-x1))
            + b3 * (load2 - np.exp(-x2)))


def spot_from_params(params: pd.Series, t: np.ndarray) -> np.ndarray:
    """Evaluate NSS spot rates for a parameter row with BETA0..3, TAU1, TAU2."""
    return svensson_spot(t, params["BETA0"], params["BETA1"], params["BETA2"],
                         params["BETA3"], params["TAU1"], params["TAU2"])


def curve_measures(spot: pd.DataFrame) -> pd.DataFrame:
    """Daily level/slope/curvature from a spot curve with maturity-in-years columns."""
    return pd.DataFrame({
        "level_10y": spot[10.0],
        "slope_2s10s": (spot[10.0] - spot[2.0]) * 100,
        "slope_10s30s": (spot[30.0] - spot[10.0]) * 100,
        "fly_2s5s10s": (2 * spot[5.0] - spot[2.0] - spot[10.0]) * 100,
    }).dropna()


def range_position(s: pd.Series, window: str = "365D") -> dict[str, float]:
    """Latest value against its own trailing range (default one calendar year).

    Returns min, max and the latest value's position in the range from 0 (at the low)
    to 100 (at the high). Useful shorthand: "2s10s is at the 90th percent of its
    1-year range" tells a trader it is near the steep end of recent history.
    """
    s = s.dropna()
    recent = s[s.index > s.index[-1] - pd.Timedelta(window)]
    lo, hi, last = float(recent.min()), float(recent.max()), float(s.iloc[-1])
    pos = 100 * (last - lo) / (hi - lo) if hi > lo else 50.0
    return {"min": lo, "max": hi, "latest": last, "pct_of_range": pos}
