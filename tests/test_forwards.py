"""Forward-rate maths against hand-computed examples."""

import math

import numpy as np
import pandas as pd
import pytest

from euro_rates_monitor.curve import svensson_spot
from euro_rates_monitor.forwards import (
    forward_path,
    forward_rate,
    priced_change,
    term_premium_scenarios,
)


def test_forward_hand_computed():
    # 1Y zero at 2%, 2Y zero at 3% (continuous). The 1Y rate in 1Y must satisfy
    # exp(0.03 * 2) = exp(0.02 * 1) * exp(f * 1)  =>  f = 0.06 - 0.02 = 4%.
    assert forward_rate(2.0, 1.0, 3.0, 2.0) == pytest.approx(4.0)
    # 6M at 1%, 18M at 2%: f = (2 * 1.5 - 1 * 0.5) / 1.0 = 2.5%.
    assert forward_rate(1.0, 0.5, 2.0, 1.5) == pytest.approx(2.5)


def test_forward_is_no_arbitrage():
    r1, t1, r2, t2 = 0.031, 0.75, 0.0345, 2.25
    f = float(forward_rate(r1, t1, r2, t2))
    lhs = math.exp(-r2 * t2)
    rhs = math.exp(-r1 * t1) * math.exp(-f * (t2 - t1))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_forward_from_today_is_spot():
    assert forward_rate(0.0, 0.0, 2.7, 0.25) == pytest.approx(2.7)


def test_forward_rejects_inverted_dates():
    with pytest.raises(ValueError):
        forward_rate(2.0, 2.0, 3.0, 1.0)


def test_flat_curve_has_flat_forwards():
    params = pd.Series({"BETA0": 2.5, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0,
                        "TAU1": 1.5, "TAU2": 10.0})
    path = forward_path(params)
    assert np.allclose(path["forward"], 2.5)
    assert path["start_years"].iloc[-1] == pytest.approx(2.0)


def test_upward_curve_prices_higher_forwards():
    # b1 < 0 gives a curve rising from b0 + b1 at the short end towards b0.
    params = pd.Series({"BETA0": 3.0, "BETA1": -1.0, "BETA2": 0.0, "BETA3": 0.0,
                        "TAU1": 1.0, "TAU2": 10.0})
    path = forward_path(params)
    assert path["forward"].is_monotonic_increasing
    # Forwards sit above spot on a rising curve.
    spot_2y = float(svensson_spot(2.25, 3.0, -1.0, 0.0, 0.0, 1.0, 10.0))
    assert path["forward"].iloc[-1] > spot_2y


def test_term_premium_scenario_is_linear_in_horizon():
    params = pd.Series({"BETA0": 2.5, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0,
                        "TAU1": 1.5, "TAU2": 10.0})
    sc = term_premium_scenarios(forward_path(params))
    two_years = sc.set_index("start_years").loc[2.0]
    assert two_years["expected_k0"] == pytest.approx(2.5)
    assert two_years["expected_k10"] == pytest.approx(2.3)   # 2.5 - 0.10 * 2
    assert two_years["expected_k20"] == pytest.approx(2.1)
    assert priced_change(sc, 2.0).iloc[0] == pytest.approx(50.0)
