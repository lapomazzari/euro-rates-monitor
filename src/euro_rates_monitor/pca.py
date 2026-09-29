"""Principal component analysis of daily curve changes.

Finance notes
-------------
Daily moves across maturities are highly correlated: when the 10Y sells off the 5Y
and 30Y usually do too. PCA finds the orthogonal combinations of maturities that
explain the most variance. In every developed bond market the first three are, almost
always:

  PC1 "level":     all maturities move the same way (loadings share one sign)
  PC2 "slope":     short and long ends move in opposite directions (one sign change)
  PC3 "curvature": the belly moves against both wings (two sign changes)

Together they typically explain well over 95% of daily variance. We run PCA on
changes (stationary) rather than levels (trending).

Curve-shape anomaly
-------------------
Projecting today's curve (in deviation from its window mean) on the first three
components and taking the residual shows where the curve's shape departs from its
usual three-factor structure. The ECB curve is itself a fitted Svensson curve, not
traded bond yields, so this residual is NOT a tradable rich/cheap signal on a bond.
It is labelled "curve-shape anomaly" throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PCA_TENORS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
FACTOR_NAMES = ["level", "slope", "curvature"]


@dataclass
class PCAResult:
    """Output of `fit_pca`.

    loadings:  maturities x components (unit-length eigenvectors, sign-normalised).
    explained: share of variance explained by each component.
    mean_level: window mean of the curve (used to centre levels for the residual).
    window:    first and last date of the estimation window.
    """

    loadings: pd.DataFrame
    explained: pd.Series
    mean_level: pd.Series
    window: tuple[pd.Timestamp, pd.Timestamp]


def _normalise_signs(vecs: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Make signs interpretable (eigenvectors are only defined up to sign).

    PC1 positive on average (a rise in yields); PC2 positive at the long end (a
    steepening); PC3 positive at the belly, near 5Y (belly up vs wings).
    """
    out = vecs.copy()
    belly = int(np.argmin(np.abs(tenors - 5.0)))
    rules = [lambda v: v.mean(), lambda v: v[-1] - v[0], lambda v: v[belly]]
    for i, rule in enumerate(rules[: out.shape[1]]):
        if rule(out[:, i]) < 0:
            out[:, i] *= -1
    return out


def fit_pca(spot: pd.DataFrame, tenors: list[float] = PCA_TENORS, years: float = 3.0,
            n: int = 3) -> PCAResult:
    """PCA on daily changes (bp) of the spot curve over the trailing `years`."""
    curve = spot[tenors].dropna()
    curve = curve[curve.index > curve.index[-1] - pd.Timedelta(days=round(365.25 * years))]
    dchg = curve.diff().dropna() * 100
    cov = np.cov(dchg.to_numpy(), rowvar=False)
    vals, vecs = np.linalg.eigh(cov)  # ascending order for symmetric matrices
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vecs = _normalise_signs(vecs[:, :n], np.asarray(tenors))
    names = FACTOR_NAMES[:n] if n <= 3 else [f"pc{i + 1}" for i in range(n)]
    return PCAResult(
        loadings=pd.DataFrame(vecs, index=tenors, columns=names),
        explained=pd.Series(vals[:n] / vals.sum(), index=names),
        mean_level=curve.mean(),
        window=(curve.index[0], curve.index[-1]),
    )


def sign_changes(v: np.ndarray, tol: float = 0.05) -> int:
    """Number of sign changes in a loading vector, ignoring near-zero entries.

    Level has 0, slope 1, curvature 2: the check that the statistical factors are the
    economic ones.
    """
    v = np.asarray(v)
    signs = np.sign(v[np.abs(v) > tol * np.abs(v).max()])
    return int(np.sum(signs[1:] != signs[:-1]))


def shape_residuals(spot: pd.DataFrame, res: PCAResult) -> pd.DataFrame:
    """Residual (bp) of each day's curve after removing the first three factors.

    Positive residual: the yield at that maturity is higher than the three-factor
    structure implies (in bond terms it "looks cheap" on the fitted curve).
    """
    tenors = list(res.loadings.index)
    dev = (spot[tenors].dropna() - res.mean_level) * 100
    load = res.loadings.to_numpy()
    fitted = dev.to_numpy() @ load @ load.T  # orthonormal projection onto 3 factors
    return pd.DataFrame(dev.to_numpy() - fitted, index=dev.index, columns=tenors)


def anomaly_table(spot: pd.DataFrame, res: PCAResult, z_flag: float = 2.0) -> pd.DataFrame:
    """Latest residual per maturity with a z-score against the window's residuals."""
    resid = shape_residuals(spot, res)
    window = resid[resid.index >= res.window[0]]
    latest = resid.iloc[-1]
    z = latest / window.std()
    flag = np.where(z >= z_flag, "yield above 3-factor fit",
                    np.where(z <= -z_flag, "yield below 3-factor fit", ""))
    return pd.DataFrame({"residual_bp": latest, "z": z, "flag": flag})
