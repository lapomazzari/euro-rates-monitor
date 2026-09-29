"""Euro versus US rates, EUR/USD against the rate differential, and the AAA-vs-all spread.

Finance notes
-------------
* Like with like. FRED's Treasury constant-maturity (CMT) yields are *par* yields, so
  the maturity-by-maturity spread uses the ECB's AAA *par* curve. The implied-path
  comparison uses zero curves on both sides (ECB NSS and Fed GSW NSS).
* Timing. The ECB curve, US CMT yields and the ECB EUR/USD reference rate (14:15 CET)
  are fixed at different times of day, so daily changes carry some noise. Weekly and
  monthly changes are less affected.
* EUR/USD and rate differentials. Higher euro rates relative to US rates tend to
  support the euro (carry and expected-return channel). The 2Y differential captures
  relative policy expectations; the 10Y adds term premium and growth expectations.
  The link is unstable, so we show rolling correlations and one simple regression over
  the last year, not a fair-value model.
* AAA vs all-issuer spread. The all-issuer curve includes lower-rated sovereigns
  (e.g. Italy, Spain). The gap to the AAA curve is a compact indicator of euro area
  sovereign credit/fragmentation risk. It also moves when the rating composition of
  the AAA basket changes (ratings come from Fitch), which is not a market move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forwards import forward_path


def par_spreads(eur_par: pd.DataFrame, us: dict[float, pd.Series]) -> pd.DataFrame:
    """EUR AAA par minus US CMT par yield by maturity, in bp, on common dates."""
    cols = {}
    for tenor, s in us.items():
        joined = pd.concat([eur_par[tenor], s], axis=1, join="inner").dropna()
        cols[tenor] = (joined.iloc[:, 0] - joined.iloc[:, 1]) * 100
    return pd.DataFrame(cols).sort_index()


def path_differential(eur_params: pd.Series, us_params: pd.Series) -> pd.DataFrame:
    """EUR and US 3M forward paths side by side and their difference (bp)."""
    eur = forward_path(eur_params).set_index("start_years")["forward"]
    usd = forward_path(us_params).set_index("start_years")["forward"]
    return pd.DataFrame({"eur": eur, "usd": usd, "diff_bp": (eur - usd) * 100})


def fx_vs_differential(eurusd: pd.Series, diff_bp: pd.Series,
                       windows: tuple[int, ...] = (63, 252)) -> dict[str, object]:
    """Rolling correlation of daily changes, and the level versus a 1-year regression.

    correlation: of daily log changes in EUR/USD with daily changes in the differential,
                 over the last 63 and 252 common observations (about 3 and 12 months).
    regression:  EUR/USD level on the differential level over the last 252 common
                 observations. `residual_pct` is today's EUR/USD versus the fitted value,
                 in percent. The relationship is descriptive, not a fair value.
    """
    df = pd.concat([eurusd.rename("fx"), diff_bp.rename("diff")], axis=1,
                   join="inner").dropna()
    chg = pd.DataFrame({"fx": np.log(df["fx"]).diff() * 100, "diff": df["diff"].diff()})
    chg = chg.dropna()
    out: dict[str, object] = {"latest_date": df.index[-1], "fx": float(df["fx"].iloc[-1]),
                              "diff_bp": float(df["diff"].iloc[-1])}
    for w in windows:
        out[f"corr_{w}d"] = float(chg.tail(w).corr().iloc[0, 1])
    out["rolling_corr_63d"] = chg["fx"].rolling(63).corr(chg["diff"])
    last = df.tail(252)
    slope, intercept = np.polyfit(last["diff"], last["fx"], 1)
    fitted = intercept + slope * out["diff_bp"]
    out.update({"beta_per_100bp": float(slope * 100), "fitted": float(fitted),
                "residual_pct": float((out["fx"] / fitted - 1) * 100),
                "r2": float(np.corrcoef(last["diff"], last["fx"])[0, 1] ** 2)})
    return out


def credit_spread(spot_all: pd.DataFrame, spot_aaa: pd.DataFrame,
                  tenors: tuple[float, ...] = (5.0, 10.0)) -> pd.DataFrame:
    """All-issuer minus AAA spot yield, in bp, at the given maturities."""
    return pd.DataFrame({
        f"all_minus_aaa_{int(t)}y": (spot_all[t] - spot_aaa[t]) * 100 for t in tenors
    }).dropna()
