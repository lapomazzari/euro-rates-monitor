"""Backtest: matching, recovery of a known premium, benchmark alignment, inference rules."""

import numpy as np
import pandas as pd
import pytest

from euro_rates_monitor import backtest as bt


def _series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates))


# ------------------------------------------------------------------------- matching

def test_realised_on_a_holiday_uses_previous_business_day():
    # Origin 2 Jan 2026 + 3 months = Thu 2 Apr 2026: present.
    # Origin 3 Jan 2026 (Sat) + 3 months = Fri 3 Apr 2026, Good Friday (TARGET2 closed),
    # so the realisation is Thu 2 Apr.
    spot = _series(["2026-01-02", "2026-04-01", "2026-04-02", "2026-04-07"],
                   [2.0, 2.1, 2.2, 2.3])
    got = bt.realised_at(spot, pd.to_datetime(["2026-01-02", "2026-01-03"]), 3)
    assert got.tolist() == [2.2, 2.2]


def test_realised_too_stale_or_not_yet_known_is_missing():
    spot = _series(["2026-01-02", "2026-03-20", "2026-06-30"], [2.0, 2.1, 2.2])
    # Target 2 Apr: last observation 20 Mar is 13 days stale -> missing.
    # Target 30 Jul: after the last observation -> not known yet -> missing,
    # even though 30 Jun would be an "as of" value.
    got = bt.realised_at(spot, pd.to_datetime(["2026-01-02", "2026-04-30"]), 3)
    assert got.isna().all()


def test_month_end_origins_use_last_available_day():
    idx = pd.to_datetime(["2026-04-28", "2026-04-29", "2026-05-01", "2026-05-29"])
    assert bt.month_end_origins(idx).tolist() == list(
        pd.to_datetime(["2026-04-29", "2026-05-29"]))


# ------------------------------------------------------------------------ recovery

def _regimes() -> bt.RegimeDates:
    return bt.RegimeDates(pd.Timestamp("2100-01-01"), pd.Timestamp("2100-06-01"))


def test_exact_forecasts_recover_a_constant_premium():
    """Forwards = realised rate + 25bp exactly: mean error must be 25bp, zero spread."""
    days = pd.bdate_range("2000-01-03", "2020-12-31")
    rate = pd.Series(3 + np.sin(np.arange(len(days)) / 300), index=days)
    fc = pd.DataFrame(index=days)
    for h in bt.HORIZONS_MONTHS:
        fc[h] = bt.realised_at(rate, days, h).to_numpy() + 0.25
    errors = bt.error_table(rate, fc.dropna(), _regimes())
    summary = bt.summary_table(errors).set_index(["sample", "horizon_m"])
    for h in (3, 6, 12):
        r = summary.loc[("full sample", h)]
        assert r["mean_err_fwd_bp"] == pytest.approx(25.0, abs=1e-9)
        assert r["sd_err_fwd_bp"] == pytest.approx(0.0, abs=1e-9)
        # No noise: RMSE is the bias itself. (A biased forecast can still lose to the
        # random walk when rates barely move, which is why bias and RMSE are both shown.)
        assert r["rmse_fwd_bp"] == pytest.approx(25.0, abs=1e-9)


def test_premium_recovered_through_the_live_forward_code():
    """Flat NSS curves at x_t + k with x a random walk: E[error] = k."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("1990-01-01", "2020-12-31")
    x = pd.Series(3 + np.cumsum(rng.normal(0, 0.02, len(days))), index=days)
    k = 0.30
    params = pd.DataFrame({"BETA0": x + k, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0,
                           "TAU1": 1.5, "TAU2": 10.0}, index=days)
    month_ends = bt.month_end_origins(days)
    errors = bt.error_table(x, bt.forward_forecasts(params.loc[month_ends]), _regimes())
    summary = bt.summary_table(errors).set_index(["sample", "horizon_m"])
    for h in (3, 6, 12):
        r = summary.loc[("full sample", h)]
        assert r["ci95_lo_fwd_bp"] <= 30 <= r["ci95_hi_fwd_bp"], h
        assert abs(r["mean_err_fwd_bp"] - 30) < abs(r["mean_err_rw_bp"] - 30)


# ------------------------------------------------------------------------ benchmark

def test_random_walk_is_scored_on_exactly_the_same_dates():
    days = pd.bdate_range("2010-01-01", "2012-12-31")
    rate = pd.Series(np.linspace(1, 2, len(days)), index=days)
    fc = pd.DataFrame({h: rate.to_numpy() + 0.1 for h in bt.HORIZONS_MONTHS}, index=days)
    fc.iloc[::7, :] = np.nan              # forecasts missing on some origin dates
    errors = bt.error_table(rate, fc, _regimes())
    assert errors[["err_fwd_bp", "err_rw_bp"]].notna().all().all()
    missing = set(fc.index[fc[3].isna()])
    assert missing.isdisjoint(set(errors.loc[errors["horizon_m"] == 3, "origin"]))
    row = errors.iloc[0]
    assert row["err_rw_bp"] == pytest.approx((rate[row["origin"]] - row["realised"]) * 100)


# ------------------------------------------------------------------------ inference

def test_withheld_cells_report_counts_only():
    errors = pd.DataFrame({"origin": pd.date_range("2020-01-31", periods=100, freq="ME"),
                           "err_fwd_bp": 1.0, "err_rw_bp": 2.0, "forward": 1.0,
                           "spot_now": 1.0, "realised_change_bp": 3.0})
    out = bt.summarise(errors, 24)          # 100 // 24 = 4 windows < MIN_WINDOWS
    assert out == {"horizon_m": 24, "n_months": 100, "windows": 4, "withheld": True}
    assert bt.summarise(errors, 3)["windows"] == 33
    assert "mean_err_fwd_bp" in bt.summarise(errors, 3)


def test_newey_west_matches_iid_at_lag_zero_and_widens_with_overlap():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=5000)
    assert bt.newey_west_se(iid, 0) == pytest.approx(iid.std() / np.sqrt(len(iid)), rel=1e-6)
    overlapping = np.convolve(rng.normal(size=5012), np.ones(12), "valid")  # MA(11)
    naive = overlapping.std() / np.sqrt(len(overlapping))
    assert bt.newey_west_se(overlapping, 12) > 2 * naive


def test_regime_dates_from_the_deposit_rate():
    dfr = _series(["2011-01-03", "2012-07-11", "2014-06-11", "2022-07-27", "2022-09-14"],
                  [1.0, 0.0, -0.1, 0.0, 0.75])
    rd = bt.regime_dates(dfr)
    assert rd.lower_bound_start == pd.Timestamp("2012-07-11")
    assert rd.hiking_start == pd.Timestamp("2022-07-27")
    labels = bt.regime_of(pd.to_datetime(["2010-01-01", "2020-01-01", "2023-01-01"]), rd)
    assert list(labels) == list(bt.REGIMES)
