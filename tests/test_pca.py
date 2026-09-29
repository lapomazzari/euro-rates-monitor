"""PCA on a synthetic curve whose level/slope/curvature structure is known."""

import numpy as np
import pandas as pd
import pytest

from euro_rates_monitor.pca import PCA_TENORS, anomaly_table, fit_pca, sign_changes

TENORS = np.array(PCA_TENORS)


def _true_factors() -> np.ndarray:
    """Orthonormal level, slope and curvature shapes over the PCA maturities."""
    x = (np.log(TENORS) - np.log(TENORS).mean()) / np.log(TENORS).std()
    raw = np.column_stack([np.ones_like(x), x, x**2 - 1])
    q, _ = np.linalg.qr(raw)  # Gram-Schmidt keeps the level/slope/curvature ordering
    return q


def _synthetic_curve(days: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = _true_factors()
    sd_bp = np.array([6.0, 3.0, 1.2])  # factor volatilities, bp per day
    shocks = rng.normal(size=(days, 3)) * sd_bp
    noise = rng.normal(scale=0.2, size=(days, len(TENORS)))
    changes_bp = shocks @ q.T + noise
    levels = 2.5 + np.cumsum(changes_bp, axis=0) / 100
    idx = pd.bdate_range("2023-01-02", periods=days)
    return pd.DataFrame(levels, index=idx, columns=TENORS)


def test_recovers_explained_variance():
    res = fit_pca(_synthetic_curve())
    total = 6.0**2 + 3.0**2 + 1.2**2 + len(TENORS) * 0.2**2
    expected = np.array([6.0**2, 3.0**2, 1.2**2]) / total
    assert res.explained.to_numpy() == pytest.approx(expected, abs=0.02)
    assert res.explained.sum() > 0.95


def test_loadings_match_level_slope_curvature():
    res = fit_pca(_synthetic_curve())
    q = _true_factors()
    for i, name in enumerate(["level", "slope", "curvature"]):
        # Both are unit vectors, so |dot product| is the cosine similarity.
        cosine = abs(float(res.loadings[name].to_numpy() @ q[:, i]))
        assert cosine > 0.99, name
    assert sign_changes(res.loadings["level"].to_numpy()) == 0
    assert sign_changes(res.loadings["slope"].to_numpy()) == 1
    assert sign_changes(res.loadings["curvature"].to_numpy()) == 2


def test_sign_convention():
    lo = fit_pca(_synthetic_curve()).loadings
    assert lo["level"].mean() > 0
    assert lo["slope"].iloc[-1] > lo["slope"].iloc[0]   # positive = steepening
    assert lo.loc[5.0, "curvature"] > 0                  # positive = belly up


def test_shape_anomaly_is_flagged_at_the_kink():
    curve = _synthetic_curve()
    res = fit_pca(curve)
    kinked = curve.copy()
    kinked.iloc[-1, list(TENORS).index(7.0)] += 0.15   # +15bp only at 7Y
    table = anomaly_table(kinked, res)
    assert table["flag"].loc[7.0] == "yield above 3-factor fit"
    assert table["residual_bp"].abs().idxmax() == 7.0
