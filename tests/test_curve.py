"""Curve measures, and our NSS evaluation against the ECB's published spot rates."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euro_rates_monitor.curve import curve_measures, range_position, svensson_spot
from euro_rates_monitor.sources import SPOT_TENORS

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


def test_measures_hand_computed():
    spot = pd.DataFrame({2.0: [3.0], 5.0: [3.2], 10.0: [3.6], 30.0: [3.9]},
                        index=pd.to_datetime(["2026-09-28"]))
    m = curve_measures(spot).iloc[0]
    assert m["level_10y"] == pytest.approx(3.6)
    assert m["slope_2s10s"] == pytest.approx(60)       # (3.6 - 3.0) * 100
    assert m["slope_10s30s"] == pytest.approx(30)
    assert m["fly_2s5s10s"] == pytest.approx(-20)      # (2*3.2 - 3.0 - 3.6) * 100


def test_range_position():
    idx = pd.date_range("2025-01-01", periods=5, freq="MS")
    r = range_position(pd.Series([1.0, 3.0, 2.0, 5.0, 4.0], index=idx))
    assert (r["min"], r["max"], r["pct_of_range"]) == (1.0, 5.0, 75.0)


def test_svensson_short_end_limit():
    # As t -> 0 the NSS spot rate tends to b0 + b1.
    assert float(svensson_spot(1e-8, 3.0, -1.0, 2.0, 1.0, 1.0, 5.0)) == pytest.approx(2.0)


@pytest.mark.skipif(not list(SAMPLE.glob("ecb_params_aaa_*.csv")), reason="no sample data")
def test_nss_reproduces_published_ecb_spot():
    """Evaluating the published parameters must give the published spot rates."""
    def wide(pattern: str) -> pd.DataFrame:
        df = pd.read_csv(sorted(SAMPLE.glob(pattern))[-1])
        df["item"] = df["KEY"].str.rsplit(".", n=1).str[1]
        return df.pivot(index="TIME_PERIOD", columns="item", values="OBS_VALUE")

    params, spot = wide("ecb_params_aaa_*.csv"), wide("ecb_spot_aaa_*.csv")
    day = sorted(set(params.index) & set(spot.index))[-1]
    p = params.loc[day]
    for code, t in SPOT_TENORS.items():
        ours = svensson_spot(t, p["BETA0"], p["BETA1"], p["BETA2"], p["BETA3"],
                             p["TAU1"], p["TAU2"])
        assert np.isclose(ours, spot.loc[day, code], atol=1e-6), code
