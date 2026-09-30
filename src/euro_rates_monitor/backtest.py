"""Backtest: how well have AAA-curve forwards predicted the 3M rate that materialised?

Design
------
* Forecast: at origin date t, the 3M forward starting in h months, taken from
  `forwards.forward_path` (the same code the live tool uses).
* Realisation: the published AAA 3M spot rate at t + h calendar months (last
  observation on or before that date, at most 5 days stale; pairs whose target date
  lies beyond the end of the data are dropped).
* Benchmark: a random walk that predicts today's 3M rate, scored on exactly the same
  origin dates.
* Forecast error = forecast - realised, in bp. Its mean is the empirical average term
  premium estimate. Because forecast and realisation come from the same government
  curve, a constant bill-vs-EUR STR gap cancels; the error still contains the term
  premium, genuine surprises in the rate path, and changes in that gap.

Sampling and inference
----------------------
Errors are computed for every day (for the chart), but statistics use month-end
origins only. Consecutive daily forecasts share almost all of their h-month window, so
daily origins add little information and would need Newey-West lags of hundreds of days.

With monthly origins and an h-month horizon, consecutive errors overlap by h-1 months
and are autocorrelated, so the usual i.i.d. standard error is far too small. We use
Newey-West (Bartlett kernel, lag = h months) and report a moving-block bootstrap
(block = h months) as a cross-check, because Newey-West understates uncertainty when
there are few independent windows.

Regimes are derived from the deposit rate: pre-lower-bound (to the first day the
deposit rate reached zero, 11 Jul 2012), zero rates and asset purchases (to the first
hike of the next cycle, 27 Jul 2022), and hiking and after. Forecasts are assigned by
origin date.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import load, paths
from .forwards import forward_path

log = logging.getLogger(__name__)

HORIZONS_MONTHS = (3, 6, 12, 24)
# Judgement call: below this many non-overlapping h-month windows, a sample holds too
# few independent observations for Newey-West (or any HAC) inference to be credible,
# so no mean, interval or test is reported. The window count is printed regardless.
MIN_WINDOWS = 12
MAX_STALE_DAYS = 5
DIRECTION_DEADBAND_BP = 5.0
BOOTSTRAP_DRAWS = 2000
SEED = 20260929

REGIMES = ("pre-lower-bound", "zero rates and asset purchases", "hiking and after")
SAMPLES = ("full sample", *REGIMES)


# ------------------------------------------------------------------------ matching

def realised_at(spot: pd.Series, origins: pd.DatetimeIndex, months: int,
                max_stale_days: int = MAX_STALE_DAYS) -> pd.Series:
    """Value of `spot` at origin + `months`, indexed by origin.

    Uses the last observation on or before the target date (holidays, missing days).
    NaN if that observation is more than `max_stale_days` older than the target, or if
    the target lies after the last observation (the outcome is not known yet).
    """
    spot = spot.dropna().sort_index()
    targets = origins + pd.DateOffset(months=months)
    pos = spot.index.searchsorted(targets, side="right") - 1
    out = np.full(len(origins), np.nan)
    for i, (target, p) in enumerate(zip(targets, pos, strict=True)):
        if p < 0 or target > spot.index[-1]:
            continue
        if (target - spot.index[p]).days <= max_stale_days:
            out[i] = spot.iloc[p]
    return pd.Series(out, index=origins, name=f"realised_{months}m")


def month_end_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last available date in each calendar month."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def forward_forecasts(params: pd.DataFrame,
                      horizons: tuple[int, ...] = HORIZONS_MONTHS) -> pd.DataFrame:
    """3M forward starting in h months for every date, via forwards.forward_path."""
    rows = {}
    for day, p in params.iterrows():
        path = forward_path(p, horizon_years=max(horizons) / 12).set_index("start_years")
        rows[day] = [path.loc[round(h / 12, 10), "forward"] for h in horizons]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(horizons)).sort_index()


# -------------------------------------------------------------------------- regimes

@dataclass(frozen=True)
class RegimeDates:
    """Boundaries derived from the deposit rate series."""

    lower_bound_start: pd.Timestamp   # first day the deposit rate was at or below zero
    hiking_start: pd.Timestamp        # first hike after that


def regime_dates(dfr: pd.Series) -> RegimeDates:
    """Find the regime boundaries in the deposit facility rate."""
    at_zero = dfr[dfr <= 0]
    lb = at_zero.index[0]
    step = dfr.diff()
    hike = step[(step.index > lb) & (step > 0)].index[0]
    return RegimeDates(lb, hike)


def regime_of(dates: pd.DatetimeIndex, rd: RegimeDates) -> np.ndarray:
    """Regime label for each origin date."""
    return np.where(dates < rd.lower_bound_start, REGIMES[0],
                    np.where(dates < rd.hiking_start, REGIMES[1], REGIMES[2]))


# --------------------------------------------------------------------------- errors

def error_table(spot3m: pd.Series, forecasts: pd.DataFrame, rd: RegimeDates) -> pd.DataFrame:
    """Long table: one row per origin date and horizon with both forecasts' errors (bp)."""
    frames = []
    for h in forecasts.columns:
        origins = forecasts.index
        now = spot3m.reindex(origins)
        realised = realised_at(spot3m, origins, int(h))
        df = pd.DataFrame({
            "origin": origins, "horizon_m": int(h), "forward": forecasts[h].to_numpy(),
            "spot_now": now.to_numpy(), "realised": realised.to_numpy(),
        })
        df["err_fwd_bp"] = (df["forward"] - df["realised"]) * 100
        df["err_rw_bp"] = (df["spot_now"] - df["realised"]) * 100
        df["realised_change_bp"] = (df["realised"] - df["spot_now"]) * 100
        df["regime"] = regime_of(pd.DatetimeIndex(df["origin"]), rd)
        frames.append(df.dropna(subset=["err_fwd_bp", "err_rw_bp"]))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------ inference

def newey_west_se(x: np.ndarray, lag: int) -> float:
    """HAC standard error of the mean of x (Bartlett weights 1 - j/(lag+1))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    u = x - x.mean()
    var = u @ u / n
    for j in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - j / (lag + 1)) * (u[j:] @ u[:-j]) / n
    return math.sqrt(max(var, 0.0) / n)


def block_bootstrap_ci(x: np.ndarray, block: int, draws: int = BOOTSTRAP_DRAWS,
                       seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval for the mean from a moving-block bootstrap."""
    x = np.asarray(x, dtype=float)
    n, block = len(x), max(1, min(block, len(x)))
    rng = np.random.default_rng(seed)
    starts_max = n - block + 1
    n_blocks = math.ceil(n / block)
    means = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, starts_max, n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _p_two_sided(z: float) -> float:
    """Two-sided normal p-value (no scipy needed: 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt 2))."""
    return math.erfc(abs(z) / math.sqrt(2)) if math.isfinite(z) else float("nan")


def summarise(errors: pd.DataFrame, horizon: int) -> dict[str, object]:
    """Statistics for one sample and horizon, on month-end origins.

    Withheld (fewer than MIN_WINDOWS non-overlapping windows): only counts reported.
    """
    df = errors.sort_values("origin")
    n = len(df)
    windows = n // horizon
    out: dict[str, object] = {"horizon_m": horizon, "n_months": n, "windows": windows,
                              "withheld": windows < MIN_WINDOWS}
    if out["withheld"] or n == 0:
        return out
    e_f, e_r = df["err_fwd_bp"].to_numpy(), df["err_rw_bp"].to_numpy()
    chg = df["realised_change_bp"].to_numpy()
    mean_abs_chg = float(np.abs(chg).mean())
    se_f, se_r = newey_west_se(e_f, horizon), newey_west_se(e_r, horizon)
    boot = block_bootstrap_ci(e_f, horizon)
    d = e_f**2 - e_r**2                       # Diebold-Mariano loss differential
    se_d = newey_west_se(d, horizon)
    dm = float(d.mean() / se_d) if se_d > 0 else float("nan")
    # Direction: only where the rate actually moved by at least the dead band.
    moved = np.abs(chg) >= DIRECTION_DEADBAND_BP
    pred = np.sign(df["forward"].to_numpy() - df["spot_now"].to_numpy())[moved]
    real = np.sign(chg[moved])
    out.update({
        "mean_err_fwd_bp": float(e_f.mean()),
        "sd_err_fwd_bp": float(e_f.std(ddof=1)),
        "rmse_fwd_bp": float(np.sqrt((e_f**2).mean())),
        "nw_se_fwd_bp": se_f,
        "ci95_lo_fwd_bp": float(e_f.mean() - 1.96 * se_f),
        "ci95_hi_fwd_bp": float(e_f.mean() + 1.96 * se_f),
        "boot_ci95_lo_fwd_bp": boot[0], "boot_ci95_hi_fwd_bp": boot[1],
        "mean_err_rw_bp": float(e_r.mean()),
        "sd_err_rw_bp": float(e_r.std(ddof=1)),
        "rmse_rw_bp": float(np.sqrt((e_r**2).mean())),
        "nw_se_rw_bp": se_r,
        "ci95_lo_rw_bp": float(e_r.mean() - 1.96 * se_r),
        "ci95_hi_rw_bp": float(e_r.mean() + 1.96 * se_r),
        "mean_abs_realised_change_bp": mean_abs_chg,
        "mean_err_fwd_share_of_change_pct":
            float(100 * e_f.mean() / mean_abs_chg) if mean_abs_chg > 0 else float("nan"),
        "mean_err_rw_share_of_change_pct":
            float(100 * e_r.mean() / mean_abs_chg) if mean_abs_chg > 0 else float("nan"),
        "rmse_ratio_fwd_to_rw": float(np.sqrt((e_f**2).mean() / (e_r**2).mean())),
        "dm_stat": dm,
        "dm_p_value": _p_two_sided(dm),
        "hit_rate_fwd_pct": float(100 * (pred == real).mean()) if moved.any() else float("nan"),
        "hit_base_rate_pct":
            float(100 * max((real > 0).mean(), (real < 0).mean())) if moved.any()
            else float("nan"),
        "n_direction_obs": int(moved.sum()),
        "n_direction_excluded": int((~moved).sum()),
    })
    return out


def summary_table(errors: pd.DataFrame) -> pd.DataFrame:
    """Statistics for every sample x horizon, on month-end origins."""
    origins = month_end_origins(pd.DatetimeIndex(errors["origin"].unique()).sort_values())
    monthly = errors[errors["origin"].isin(origins)]
    rows = []
    for sample in SAMPLES:
        sub = monthly if sample == "full sample" else monthly[monthly["regime"] == sample]
        for h in HORIZONS_MONTHS:
            rows.append({"sample": sample, **summarise(sub[sub["horizon_m"] == h], h)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- headline

def headline(summary: pd.DataFrame) -> dict[str, object]:
    """Rounded figures for the README (full sample, horizons with inference)."""
    full = summary[(summary["sample"] == "full sample") & ~summary["withheld"]]
    out: dict[str, object] = {"backtest_min_windows": MIN_WINDOWS}
    beats, loses = [], []
    for _, r in full.iterrows():
        h = int(r["horizon_m"])
        out[f"bt_{h}m_rmse_ratio"] = round(r["rmse_ratio_fwd_to_rw"], 2)
        out[f"bt_{h}m_dm_p"] = round(r["dm_p_value"], 2)
        out[f"bt_{h}m_mean_err_bp"] = int(round(r["mean_err_fwd_bp"]))
        out[f"bt_{h}m_ci_lo_bp"] = int(round(r["ci95_lo_fwd_bp"]))
        out[f"bt_{h}m_ci_hi_bp"] = int(round(r["ci95_hi_fwd_bp"]))
        out[f"bt_{h}m_share_pct"] = int(round(r["mean_err_fwd_share_of_change_pct"]))
        out[f"bt_{h}m_windows"] = int(r["windows"])
        significant = r["dm_p_value"] < 0.05
        if r["rmse_ratio_fwd_to_rw"] < 1 and significant:
            beats.append(h)
        elif r["rmse_ratio_fwd_to_rw"] > 1 and significant:
            loses.append(h)
    out["bt_horizons_tested"] = [int(h) for h in full["horizon_m"]]
    out["bt_horizons_forwards_significantly_better"] = beats
    out["bt_horizons_forwards_significantly_worse"] = loses
    wh = summary[(summary["sample"] == "full sample") & summary["withheld"]]
    out["bt_horizons_withheld_full_sample"] = [int(h) for h in wh["horizon_m"]]
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame, RegimeDates]:
    """Compute daily errors and the summary from the cached data."""
    spot = load.ecb_curve("ecb_spot_aaa")[0.25]
    params = load.ecb_wide("ecb_params_aaa").dropna()
    rd = regime_dates(load.ecb_series("ecb_dfr"))
    log.info("computing forwards for %d dates", len(params))
    errors = error_table(spot, forward_forecasts(params), rd)
    return errors, summary_table(errors), rd


def write(errors: pd.DataFrame, summary: pd.DataFrame, rd: RegimeDates) -> None:
    """Persist results to data/processed/."""
    out = paths.PROCESSED
    errors.round({c: 4 for c in errors.select_dtypes("number").columns}).to_csv(
        out / "backtest_errors.csv", index=False)
    summary.round(4).to_csv(out / "backtest_summary.csv", index=False)
    head = headline(summary)
    head["bt_lower_bound_start"] = rd.lower_bound_start.date().isoformat()
    head["bt_hiking_start"] = rd.hiking_start.date().isoformat()
    head["bt_first_origin"] = pd.Timestamp(errors["origin"].min()).date().isoformat()
    (out / "backtest_headline.json").write_text(json.dumps(head, indent=2))
    log.info("wrote data/processed/backtest_*.csv and backtest_headline.json")
