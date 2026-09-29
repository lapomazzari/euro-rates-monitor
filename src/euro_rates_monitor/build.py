"""Run every analytic on the cached data and write data/processed/.

The output that matters most is `metrics_latest.json`: a flat dictionary of every
number the weekly note is allowed to use, already rounded to the precision shown in
the note. The note generator reads only this file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import crossmarket, curve, forwards, load, paths, pca, realrates
from .changes import WINDOWS, change_over

log = logging.getLogger(__name__)


@dataclass
class Results:
    """Everything the charts and the note need, computed once."""

    as_of: pd.Timestamp
    spot_aaa: pd.DataFrame
    spot_all: pd.DataFrame
    params_aaa: pd.DataFrame
    measures: pd.DataFrame
    dfr: pd.Series
    estr: pd.Series
    path: pd.DataFrame
    path_1w: pd.DataFrame
    pca: pca.PCAResult
    anomalies: pd.DataFrame
    par_spreads: pd.DataFrame
    path_diff: pd.DataFrame
    credit: pd.DataFrame
    real: pd.DataFrame
    fx: dict[str, object]
    turning_points: pd.DataFrame
    metrics: dict[str, object] = field(default_factory=dict)


def policy_turning_points(dfr: pd.Series) -> pd.DataFrame:
    """Dates where the ECB changed direction: first hike after cuts, first cut after hikes.

    Derived from the deposit facility rate itself, so no hand-typed dates. A change
    that reverses the previous change within 7 days is skipped: that is the ECB
    adjusting the width of its rate corridor (e.g. 9 Oct 2008, the day after a
    coordinated cut), not a change in policy direction.
    """
    step = dfr.diff().fillna(0)
    moves = step[step != 0]
    rows, prev = [], 0.0
    last_when, last_step = None, 0.0
    for when, s in moves.items():
        corridor = (last_when is not None and (when - last_when).days <= 7
                    and np.isclose(s, -last_step))
        last_when, last_step = when, s
        if corridor:
            continue
        d = np.sign(s)
        if d != prev:
            rows.append({"date": when, "kind": "first hike" if d > 0 else "first cut",
                         "dfr": float(dfr.loc[when])})
            prev = d
    return pd.DataFrame(rows)


def _weekly_z(s: pd.Series, chg_1w: float | None) -> float:
    """1-week change relative to the typical 1-week change over the past year."""
    if chg_1w is None:
        return 0.0
    daily = s.asfreq("D").ffill()
    weekly = (daily - daily.shift(7)).dropna()
    sd = float(weekly[weekly.index > weekly.index[-1] - pd.Timedelta(days=365)].std())
    return chg_1w / sd if sd > 0 else 0.0


def _fwd_history(params: pd.DataFrame, start: float, tenor: float = 0.25) -> pd.Series:
    """Daily history of the 3M forward starting in `start` years, from NSS parameters."""
    args = [params[c].to_numpy() for c in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")]
    r1 = curve.svensson_spot(start, *args)
    r2 = curve.svensson_spot(start + tenor, *args)
    return pd.Series(forwards.forward_rate(r1, start, r2, start + tenor), index=params.index)


def _r(x: float | None, nd: int) -> float | None:
    """Round for display; nd=0 returns an int so the note shows '12', not '12.0'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(round(x)) if nd == 0 else round(float(x), nd)


def _chg(s: pd.Series, scale: float = 1.0, nd: int = 0, stale: int = 5) -> dict[str, object]:
    out = {}
    for label, window in WINDOWS.items():
        ch = change_over(s, window, max_stale_days=stale)
        out[f"chg_{label}"] = _r(None if ch.change is None else ch.change * scale, nd)
    return out


def run() -> Results:
    """Load the cache, compute every analytic, and return them in one object."""
    spot_aaa = load.ecb_curve("ecb_spot_aaa")
    spot_all = load.ecb_curve("ecb_spot_all")
    params = load.ecb_wide("ecb_params_aaa").dropna()
    par_aaa = load.ecb_curve("ecb_par_aaa")
    as_of = spot_aaa.index[-1]
    dfr_full, estr = load.ecb_series("ecb_dfr"), load.ecb_series("ecb_estr")
    dfr = dfr_full[dfr_full.index <= as_of]

    measures = curve.curve_measures(spot_aaa)
    path = forwards.term_premium_scenarios(forwards.forward_path(params.loc[as_of]))
    wk_date = change_over(spot_aaa[10.0], WINDOWS["1w"]).base_date
    path_1w = forwards.forward_path(params.loc[wk_date])

    fit = pca.fit_pca(spot_aaa)
    anomalies = pca.anomaly_table(spot_aaa, fit)

    us_cmt = {t: load.fred_series(f"fred_dgs{t}") for t in (2, 5, 10, 30)}
    spreads = crossmarket.par_spreads(par_aaa, {float(t): s for t, s in us_cmt.items()})
    gsw = load.gsw().dropna(subset=["BETA0", "TAU1", "TAU2"])
    path_diff = crossmarket.path_differential(params.loc[as_of], gsw.iloc[-1])
    credit = crossmarket.credit_spread(spot_all, spot_aaa)

    eurusd = load.ecb_series("ecb_eurusd")
    fx2 = crossmarket.fx_vs_differential(eurusd, spreads[2.0])
    fx10 = crossmarket.fx_vs_differential(eurusd, spreads[10.0])

    hicp = realrates.available_from(load.ecb_series("ecb_hicp"), pd.DateOffset(months=1))
    spf_lt_raw = load.ecb_series("ecb_spf_lt")
    cpi_yoy = realrates.us_cpi_yoy(load.fred_series("fred_cpiaucsl"))
    real = pd.DataFrame({
        "ea_10y_minus_hicp": realrates.real_rate(spot_aaa[10.0], hicp),
        "ea_10y_minus_spf_lt": realrates.real_rate(spot_aaa[10.0], spf_lt_raw),
        "us_10y_minus_cpi": realrates.real_rate(
            us_cmt[10], realrates.available_from(cpi_yoy, pd.DateOffset(months=1, days=14))),
        "us_tips_10y": load.fred_series("fred_dfii10"),
    })

    res = Results(as_of, spot_aaa, spot_all, params, measures, dfr, estr, path, path_1w,
                  fit, anomalies, spreads, path_diff, credit, real,
                  {"2y": fx2, "10y": fx10}, policy_turning_points(dfr_full))
    res.metrics = _metrics(res, load.ecb_series("ecb_hicp"), spf_lt_raw,
                           load.ecb_series("ecb_spf_1y"), us_cmt, gsw, eurusd)
    return res


def _metrics(r: Results, hicp: pd.Series, spf_lt: pd.Series, spf_1y: pd.Series,
             us_cmt: dict[int, pd.Series], gsw: pd.DataFrame,
             eurusd: pd.Series) -> dict[str, object]:
    """Flat dictionary of display-rounded figures. Units are in the key names."""
    m: dict[str, object] = {"as_of": r.as_of.date().isoformat()}
    dfr = float(r.dfr.iloc[-1])
    estr_now = float(r.estr[r.estr.index <= r.as_of].iloc[-1])

    # 1. Curve state -------------------------------------------------------------
    for t in (2.0, 5.0, 10.0, 30.0):
        key = f"aaa_{int(t)}y"
        m[f"{key}_pct"] = _r(r.spot_aaa[t].iloc[-1], 2)
        for k, v in _chg(r.spot_aaa[t], 100).items():
            m[f"{key}_{k}_bp"] = v
    for col, unit, nd in [("slope_2s10s", "bp", 0), ("slope_10s30s", "bp", 0),
                          ("fly_2s5s10s", "bp", 0)]:
        s = r.measures[col]
        rng = curve.range_position(s)
        m[f"{col}_{unit}"] = _r(s.iloc[-1], nd)
        for k, v in _chg(s).items():
            m[f"{col}_{k}_bp"] = v
        m[f"{col}_1y_low_bp"], m[f"{col}_1y_high_bp"] = _r(rng["min"], 0), _r(rng["max"], 0)
        m[f"{col}_pct_of_1y_range"] = _r(rng["pct_of_range"], 0)
    rng = curve.range_position(r.measures["level_10y"])
    m["aaa_10y_1y_low_pct"], m["aaa_10y_1y_high_pct"] = _r(rng["min"], 2), _r(rng["max"], 2)
    m["aaa_10y_pct_of_1y_range"] = _r(rng["pct_of_range"], 0)

    # 2. ECB pricing -------------------------------------------------------------
    summ = forwards.ecb_pricing_summary(r.path, dfr, estr_now)
    summ_1w = forwards.ecb_pricing_summary(r.path_1w, dfr, estr_now)
    m["ecb_dfr_pct"], m["estr_pct"] = _r(dfr, 2), _r(estr_now, 2)
    m["estr_minus_dfr_bp"] = _r(summ["estr_minus_dfr_bp"], 0)
    m["aaa_3m_pct"] = _r(summ["spot_3m"], 2)
    m["basis_aaa3m_minus_estr_bp"] = _r(summ["basis_3m_estr_bp"], 0)
    for h in ("6m", "1y", "2y"):
        m[f"fwd3m_in_{h}_pct"] = _r(summ[f"fwd_{h}"], 2)
        m[f"fwd3m_in_{h}_minus_dfr_bp"] = _r(summ[f"vs_dfr_{h}_bp"], 0)
        m[f"fwd3m_in_{h}_minus_aaa3m_bp"] = _r(summ[f"vs_3m_{h}_bp"], 0)
        m[f"fwd3m_in_{h}_chg_1w_bp"] = _r((summ[f"fwd_{h}"] - summ_1w[f"fwd_{h}"]) * 100, 0)
    idx = r.path.set_index("start_years")
    for k in forwards.TP_SCENARIOS_BP_PER_YEAR[1:]:
        m[f"scenario_tp_{k}bp_per_year_fwd3m_in_2y_pct"] = _r(idx.loc[2.0, f"expected_k{k}"], 2)
    m["scenario_tp_bp_per_year_values"] = list(forwards.TP_SCENARIOS_BP_PER_YEAR)

    # 3. PCA ---------------------------------------------------------------------
    exp = r.pca.explained * 100
    for name in exp.index:
        m[f"pca_{name}_explained_pct"] = _r(exp[name], 1)
        m[f"pca_{name}_sign_changes"] = pca.sign_changes(r.pca.loadings[name].to_numpy())
    m["pca_top3_explained_pct"] = _r(exp.sum(), 1)
    m["pca_window_start"] = r.pca.window[0].date().isoformat()
    flagged = r.anomalies[r.anomalies["flag"] != ""]
    m["shape_anomalies"] = [
        {"maturity_years": _r(t, 0), "residual_bp": _r(row.residual_bp, 0), "z": _r(row.z, 1),
         "flag": row.flag} for t, row in flagged.iterrows()
    ]

    # 4. Cross-market ------------------------------------------------------------
    for t in (2.0, 10.0):
        s = r.par_spreads[t].dropna()
        m[f"eur_minus_us_{int(t)}y_par_bp"] = _r(s.iloc[-1], 0)
        for k, v in _chg(s).items():
            m[f"eur_minus_us_{int(t)}y_par_{k}_bp"] = v
    m["eur_us_spread_date"] = r.par_spreads.dropna().index[-1].date().isoformat()
    m["ust_2y_pct"], m["ust_10y_pct"] = _r(us_cmt[2].iloc[-1], 2), _r(us_cmt[10].iloc[-1], 2)
    fu, fl = load.fred_series("fred_dfedtaru"), load.fred_series("fred_dfedtarl")
    m["fed_funds_range_low_pct"], m["fed_funds_range_high_pct"] = (_r(fl.iloc[-1], 2),
                                                                   _r(fu.iloc[-1], 2))
    m["us_zero_curve_date"] = gsw.index[-1].date().isoformat()
    m["us_fwd3m_in_1y_pct"] = _r(r.path_diff.loc[1.0, "usd"], 2)
    m["eur_minus_us_fwd3m_in_1y_bp"] = _r(r.path_diff.loc[1.0, "diff_bp"], 0)
    m["eur_minus_us_3m_zero_bp"] = _r(r.path_diff.loc[0.0, "diff_bp"], 0)

    # FX -------------------------------------------------------------------------
    fx2, fx10 = r.fx["2y"], r.fx["10y"]
    m["eurusd"] = _r(eurusd.iloc[-1], 4)
    m["eurusd_date"] = eurusd.index[-1].date().isoformat()
    # The regression uses dates common to EUR/USD and the US yields, so its "today"
    # can lag the latest ECB fixing by a US data day.
    m["eurusd_fit_date"] = fx2["latest_date"].date().isoformat()
    m["eurusd_on_fit_date"] = _r(fx2["fx"], 4)
    for k, v in _chg(eurusd, 1.0, nd=4).items():
        m[f"eurusd_{k}"] = v
    for k, v in _chg(np.log(eurusd) * 100, 1.0, nd=1).items():
        m[f"eurusd_{k}_pct"] = v
    for tag, fx in (("2y", fx2), ("10y", fx10)):
        m[f"eurusd_corr_{tag}_diff_3m"] = _r(fx["corr_63d"], 2)
        m[f"eurusd_corr_{tag}_diff_1y"] = _r(fx["corr_252d"], 2)
    m["eurusd_fit_on_2y_diff_1y"] = _r(fx2["fitted"], 4)
    m["eurusd_minus_fit_pct"] = _r(fx2["residual_pct"], 1)
    m["eurusd_fit_r2"] = _r(fx2["r2"], 2)

    # Credit -------------------------------------------------------------------
    for col in r.credit.columns:
        s = r.credit[col]
        m[f"{col}_bp"] = _r(s.iloc[-1], 0)
        for k, v in _chg(s).items():
            m[f"{col}_{k}_bp"] = v
        rng = curve.range_position(s)
        m[f"{col}_1y_low_bp"], m[f"{col}_1y_high_bp"] = _r(rng["min"], 0), _r(rng["max"], 0)

    # 5. Real rates --------------------------------------------------------------
    m["hicp_yoy_pct"] = _r(hicp.iloc[-1], 1)
    m["hicp_month"] = hicp.index[-1].strftime("%Y-%m")
    m["spf_longer_term_pct"] = _r(spf_lt.iloc[-1], 1)
    m["spf_survey_quarter"] = f"{spf_lt.index[-1].year}-Q{spf_lt.index[-1].quarter}"
    m["spf_1y_ahead_pct"] = _r(spf_1y.iloc[-1], 1)
    for col in r.real.columns:
        s = r.real[col].dropna()
        m[f"real_{col}_pct"] = _r(s.iloc[-1], 2)
    m["us_breakeven_10y_pct"] = _r(load.fred_series("fred_t10yie").iloc[-1], 2)

    # Hedging context ----------------------------------------------------------
    m["aaa_2y_minus_estr_bp"] = _r((r.spot_aaa[2.0].iloc[-1] - estr_now) * 100, 0)

    # What moved: rank the week's changes by size relative to a normal week -----
    cands = {
        "AAA 2Y yield": (r.spot_aaa[2.0], 100),
        "AAA 10Y yield": (r.spot_aaa[10.0], 100),
        "AAA 30Y yield": (r.spot_aaa[30.0], 100),
        "3M forward in 1Y": (_fwd_history(r.params_aaa, 1.0), 100),
        "2s10s slope": (r.measures["slope_2s10s"], 1),
        "10s30s slope": (r.measures["slope_10s30s"], 1),
        "2s5s10s butterfly": (r.measures["fly_2s5s10s"], 1),
        "EUR-US 2Y spread": (r.par_spreads[2.0].dropna(), 1),
        "EUR-US 10Y spread": (r.par_spreads[10.0].dropna(), 1),
    }
    ranked = []
    for label, (s, scale) in cands.items():
        ch = change_over(s, WINDOWS["1w"]).change
        ch = None if ch is None else ch * scale
        is_yield = scale != 1  # yields are stored in %, spreads/slopes already in bp
        ranked.append({"measure": label, "chg_1w_bp": _r(ch, 0),
                       "latest": _r(s.iloc[-1], 2 if is_yield else 0),
                       "latest_unit": "%" if is_yield else "bp",
                       "size_vs_typical_week": _r(abs(_weekly_z(s * scale, ch)), 1)})
    ranked.sort(key=lambda d: d["size_vs_typical_week"] or 0, reverse=True)
    m["top_moves_1w"] = ranked[:3]
    return m


def write(res: Results) -> None:
    """Persist processed tables and the metrics dictionary."""
    out = paths.PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    res.spot_aaa.round(4).to_csv(out / "spot_aaa.csv")
    res.measures.round(2).to_csv(out / "curve_measures.csv")
    res.path.round(4).to_csv(out / "forward_path.csv", index=False)
    res.pca.loadings.round(4).to_csv(out / "pca_loadings.csv")
    res.anomalies.round(2).to_csv(out / "shape_anomalies.csv")
    res.par_spreads.round(1).to_csv(out / "eur_minus_us_par_bp.csv")
    res.path_diff.round(4).to_csv(out / "path_differential.csv")
    res.credit.round(1).to_csv(out / "credit_all_minus_aaa_bp.csv")
    res.real.round(3).to_csv(out / "real_rates.csv")
    res.turning_points.to_csv(out / "ecb_turning_points.csv", index=False)
    text = json.dumps(res.metrics, indent=2, default=str)
    (out / "metrics_latest.json").write_text(text)
    (out / f"metrics_{res.metrics['as_of']}.json").write_text(text)
    log.info("wrote data/processed/ (as of %s)", res.metrics["as_of"])
