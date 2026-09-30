"""Run every analytic on the cached data and write data/processed/.

The output that matters most is `metrics_latest.json`: a flat dictionary of every
number the weekly note is allowed to use, already rounded to the precision shown in
the note. The note generator reads only this file.

The core euro analysis always runs (its inputs are required by `erm fetch`). Each
optional section runs only if all of its inputs are cached; otherwise it is listed in
`sections_unavailable` and none of its keys are written, so a missing section can
never be filled with last week's numbers. `freshness` records, per series, when the
file was retrieved and its last observation (see freshness.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import crossmarket, curve, forwards, freshness, load, paths, pca, realrates
from .changes import WINDOWS, change_over
from .sources import SECTION_LABELS

log = logging.getLogger(__name__)

_CROSS = ["ecb_par_aaa", "fred_dgs2", "fred_dgs5", "fred_dgs10", "fred_dgs30",
          "fred_dfedtaru", "fred_dfedtarl", "fed_gsw"]
SECTION_INPUTS: dict[str, list[str]] = {
    "credit": ["ecb_spot_all"],
    "cross_market": _CROSS,
    "fx": ["ecb_eurusd", *_CROSS],                 # FX is read against the EUR-US spread
    "real_rates_ea": ["ecb_hicp", "ecb_spf_lt", "ecb_spf_1y"],
    "real_rates_us": ["fred_dgs10", "fred_cpiaucsl", "fred_t10yie", "fred_dfii10"],
}


def available_sections() -> tuple[list[str], list[str]]:
    """Split sections into (available, unavailable) by whether their inputs are cached."""
    ok = [s for s, names in SECTION_INPUTS.items() if all(load.has(n) for n in names)]
    return ok, [s for s in SECTION_INPUTS if s not in ok]


@dataclass
class Results:
    """Everything the charts and the note need, computed once."""

    as_of: pd.Timestamp
    spot_aaa: pd.DataFrame
    params_aaa: pd.DataFrame
    measures: pd.DataFrame
    dfr: pd.Series
    estr: pd.Series
    path: pd.DataFrame
    path_1w: pd.DataFrame
    pca: pca.PCAResult
    anomalies: pd.DataFrame
    turning_points: pd.DataFrame
    unavailable: list[str] = field(default_factory=list)
    # Optional sections: None when their inputs are unavailable.
    par_spreads: pd.DataFrame | None = None
    path_diff: pd.DataFrame | None = None
    credit: pd.DataFrame | None = None
    real: pd.DataFrame | None = None
    fx: dict[str, object] | None = None
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


def run(run_date: date | None = None) -> Results:
    """Load the cache, compute every available analytic, and return them in one object."""
    ok, unavailable = available_sections()
    for sec in unavailable:
        log.warning("section unavailable (inputs missing): %s", SECTION_LABELS[sec])

    # Core euro analysis: always computed.
    spot_aaa = load.ecb_curve("ecb_spot_aaa")
    params = load.ecb_wide("ecb_params_aaa").dropna()
    as_of = spot_aaa.index[-1]
    dfr_full, estr = load.ecb_series("ecb_dfr"), load.ecb_series("ecb_estr")
    dfr = dfr_full[dfr_full.index <= as_of]
    measures = curve.curve_measures(spot_aaa)
    path = forwards.term_premium_scenarios(forwards.forward_path(params.loc[as_of]))
    wk_date = change_over(spot_aaa[10.0], WINDOWS["1w"]).base_date
    path_1w = forwards.forward_path(params.loc[wk_date])
    fit = pca.fit_pca(spot_aaa)
    res = Results(as_of, spot_aaa, params, measures, dfr, estr, path, path_1w, fit,
                  pca.anomaly_table(spot_aaa, fit), policy_turning_points(dfr_full),
                  unavailable=unavailable)
    inputs: dict[str, object] = {}

    if "credit" in ok:
        res.credit = crossmarket.credit_spread(load.ecb_curve("ecb_spot_all"), spot_aaa)
    if "cross_market" in ok:
        us_cmt = {t: load.fred_series(f"fred_dgs{t}") for t in (2, 5, 10, 30)}
        res.par_spreads = crossmarket.par_spreads(
            load.ecb_curve("ecb_par_aaa"), {float(t): s for t, s in us_cmt.items()})
        gsw = load.gsw().dropna(subset=["BETA0", "TAU1", "TAU2"])
        res.path_diff = crossmarket.path_differential(params.loc[as_of], gsw.iloc[-1])
        inputs.update(us_cmt=us_cmt, gsw=gsw)
    if "fx" in ok:
        eurusd = load.ecb_series("ecb_eurusd")
        res.fx = {"2y": crossmarket.fx_vs_differential(eurusd, res.par_spreads[2.0]),
                  "10y": crossmarket.fx_vs_differential(eurusd, res.par_spreads[10.0])}
        inputs["eurusd"] = eurusd
    real = {}
    if "real_rates_ea" in ok:
        hicp_raw, spf_lt = load.ecb_series("ecb_hicp"), load.ecb_series("ecb_spf_lt")
        hicp = realrates.available_from(hicp_raw, pd.DateOffset(months=1))
        real["ea_10y_minus_hicp"] = realrates.real_rate(spot_aaa[10.0], hicp)
        real["ea_10y_minus_spf_lt"] = realrates.real_rate(spot_aaa[10.0], spf_lt)
        inputs.update(hicp=hicp_raw, spf_lt=spf_lt, spf_1y=load.ecb_series("ecb_spf_1y"))
    if "real_rates_us" in ok:
        cpi_yoy = realrates.us_cpi_yoy(load.fred_series("fred_cpiaucsl"))
        real["us_10y_minus_cpi"] = realrates.real_rate(
            load.fred_series("fred_dgs10"),
            realrates.available_from(cpi_yoy, pd.DateOffset(months=1, days=14)))
        real["us_tips_10y"] = load.fred_series("fred_dfii10")
    if real:
        res.real = pd.DataFrame(real)

    res.metrics = _metrics(res, inputs)
    per = freshness.series_freshness(run_date)
    res.metrics["sections_unavailable"] = unavailable
    res.metrics["freshness"] = {
        "run_date": (run_date or date.today()).isoformat(),
        "stale_after_days": freshness.STALE_AFTER_DAYS,
        "series": per,
        "sources": freshness.source_summary(per),
    }
    return res


def _metrics(r: Results, inputs: dict[str, object]) -> dict[str, object]:
    """Flat dictionary of display-rounded figures. Units are in the key names.

    Keys of an unavailable section are omitted entirely.
    """
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
    if r.par_spreads is not None:
        _cross_metrics(m, r, inputs)
    if r.fx is not None:
        _fx_metrics(m, r, inputs["eurusd"])
    if r.credit is not None:
        _credit_metrics(m, r)
    if r.real is not None:
        _real_metrics(m, r, inputs)

    # Hedging context ----------------------------------------------------------
    m["aaa_2y_minus_estr_bp"] = _r((r.spot_aaa[2.0].iloc[-1] - estr_now) * 100, 0)

    # What moved: rank the week's changes by size relative to a normal week -----
    m["top_moves_1w"] = _top_moves(r)
    return m


def _cross_metrics(m: dict[str, object], r: Results, inputs: dict[str, object]) -> None:
    us_cmt, gsw = inputs["us_cmt"], inputs["gsw"]
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


def _fx_metrics(m: dict[str, object], r: Results, eurusd: pd.Series) -> None:
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


def _credit_metrics(m: dict[str, object], r: Results) -> None:
    for col in r.credit.columns:
        s = r.credit[col]
        m[f"{col}_bp"] = _r(s.iloc[-1], 0)
        for k, v in _chg(s).items():
            m[f"{col}_{k}_bp"] = v
        rng = curve.range_position(s)
        m[f"{col}_1y_low_bp"], m[f"{col}_1y_high_bp"] = _r(rng["min"], 0), _r(rng["max"], 0)


def _real_metrics(m: dict[str, object], r: Results, inputs: dict[str, object]) -> None:
    if "hicp" in inputs:
        hicp, spf_lt = inputs["hicp"], inputs["spf_lt"]
        m["hicp_yoy_pct"] = _r(hicp.iloc[-1], 1)
        m["hicp_month"] = hicp.index[-1].strftime("%Y-%m")
        m["spf_longer_term_pct"] = _r(spf_lt.iloc[-1], 1)
        m["spf_survey_quarter"] = f"{spf_lt.index[-1].year}-Q{spf_lt.index[-1].quarter}"
        m["spf_1y_ahead_pct"] = _r(inputs["spf_1y"].iloc[-1], 1)
    for col in r.real.columns:
        s = r.real[col].dropna()
        m[f"real_{col}_pct"] = _r(s.iloc[-1], 2)
    if "us_tips_10y" in r.real.columns:
        m["us_breakeven_10y_pct"] = _r(load.fred_series("fred_t10yie").iloc[-1], 2)


def _top_moves(r: Results) -> list[dict[str, object]]:
    """The week's largest moves relative to a typical week over the past year."""
    cands = {
        "AAA 2Y yield": (r.spot_aaa[2.0], 100),
        "AAA 10Y yield": (r.spot_aaa[10.0], 100),
        "AAA 30Y yield": (r.spot_aaa[30.0], 100),
        "3M forward in 1Y": (_fwd_history(r.params_aaa, 1.0), 100),
        "2s10s slope": (r.measures["slope_2s10s"], 1),
        "10s30s slope": (r.measures["slope_10s30s"], 1),
        "2s5s10s butterfly": (r.measures["fly_2s5s10s"], 1),
    }
    if r.par_spreads is not None:
        cands["EUR-US 2Y spread"] = (r.par_spreads[2.0].dropna(), 1)
        cands["EUR-US 10Y spread"] = (r.par_spreads[10.0].dropna(), 1)
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
    return ranked[:3]


def write(res: Results) -> None:
    """Persist processed tables and the metrics dictionary."""
    out = paths.PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    res.spot_aaa.round(4).to_csv(out / "spot_aaa.csv")
    res.measures.round(2).to_csv(out / "curve_measures.csv")
    res.path.round(4).to_csv(out / "forward_path.csv", index=False)
    res.pca.loadings.round(4).to_csv(out / "pca_loadings.csv")
    res.anomalies.round(2).to_csv(out / "shape_anomalies.csv")
    optional = {"eur_minus_us_par_bp.csv": (res.par_spreads, 1),
                "path_differential.csv": (res.path_diff, 4),
                "credit_all_minus_aaa_bp.csv": (res.credit, 1),
                "real_rates.csv": (res.real, 3)}
    for name, (frame, nd) in optional.items():
        if frame is None:
            # Remove last week's table rather than leave it looking current.
            (out / name).unlink(missing_ok=True)
        else:
            frame.round(nd).to_csv(out / name)
    res.turning_points.to_csv(out / "ecb_turning_points.csv", index=False)
    write_metrics(res)


def write_metrics(res: Results) -> None:
    """Write metrics_latest.json and the dated figures file (frozen once noted)."""
    out = paths.PROCESSED
    text = json.dumps(res.metrics, indent=2, default=str)
    (out / "metrics_latest.json").write_text(text)
    # A note is checked against the figures file for its date. Once a note exists,
    # that file is frozen: a later rebuild (more US days, a data revision, a new
    # retrieval date) must not change the figures the note was written from.
    as_of = res.metrics["as_of"]
    dated = out / f"metrics_{as_of}.json"
    if dated.exists() and (paths.NOTES / f"{as_of}.md").exists():
        log.info("kept %s unchanged: notes/%s.md was written against it", dated.name, as_of)
    else:
        dated.write_text(text)
    log.info("wrote data/processed/ (as of %s)", as_of)
