"""The five charts. One y-axis each, units and source on every chart.

Palette (validated for colour-vision deficiency, light surface): blue = the headline
series, orange = comparison, aqua = third series (direct-labelled, as its contrast
against the background is below 3:1). Grey = context, never data that matters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import paths  # noqa: E402
from .build import Results  # noqa: E402
from .changes import WINDOWS, value_asof  # noqa: E402
from .curve import spot_from_params  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
SOURCE_ECB = "Source: ECB statistics; own calculations."
SOURCE_ALL = "Source: ECB statistics; FRED (St. Louis Fed); own calculations."


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 24,
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
        "legend.labelcolor": INK2, "lines.linewidth": 2, "font.size": 10,
    })


def _finish(fig: plt.Figure, ax: plt.Axes, subtitle: str, source: str, path: Path) -> None:
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.text(0.01, 0.01, source, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curve_snapshot(r: Results, out: Path) -> Path:
    """AAA spot curve today vs one week and one year ago (smooth NSS, dots = published)."""
    grid = np.linspace(0.25, 30, 240)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    specs = [("today", r.as_of, BLUE, 2.2), ("1 week ago", r.as_of - WINDOWS["1w"], ORANGE, 1.6),
             ("1 year ago", r.as_of - pd.DateOffset(years=1), GREY, 1.6)]
    for label, when, color, lw in specs:
        found = value_asof(r.params_aaa["BETA0"], when)
        if found is None:
            continue
        day = found[0]
        ax.plot(grid, spot_from_params(r.params_aaa.loc[day], grid), color=color, lw=lw,
                label=f"{label} ({day:%d %b %Y})")
        pts = r.spot_aaa.loc[day]
        ax.plot(pts.index, pts.values, "o", ms=4.5, color=color, mec=SURFACE, mew=1)
        ax.annotate(f"{pts[30.0]:.2f}%", (30, pts[30.0]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 33)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Zero-coupon yield (%)")
    ax.set_title("Euro area AAA government curve")
    ax.legend(loc="lower right")
    _finish(fig, ax, "Spot (zero-coupon) rates, continuously compounded. Dots: published "
            "maturities; lines: ECB Svensson fit.", SOURCE_ECB, out / "curve_snapshot.png")
    return out / "curve_snapshot.png"


def forward_path(r: Results, out: Path) -> Path:
    """3M forward path vs the deposit rate, with term-premium sensitivity lines."""
    p = r.path
    start = r.as_of + pd.to_timedelta(p["start_years"] * 365.25, unit="D")
    dfr = float(r.dfr.iloc[-1])
    estr = float(r.estr[r.estr.index <= r.as_of].iloc[-1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, ls in ((10, (0, (4, 3))), (20, (0, (1, 2)))):
        ax.plot(start, p[f"expected_k{k}"], color=GREY, lw=1.3, ls=ls,
                label=f"sensitivity: minus {k} bp per year of horizon")
    ax.plot(start, p["forward"], color=BLUE, lw=2.2, marker="o", ms=5, mec=SURFACE,
            label="3M forward rate from AAA curve (headline)")
    ax.axhline(dfr, color=ORANGE, lw=1.6, label=f"ECB deposit rate today ({dfr:.2f}%)")
    ax.plot([r.as_of], [estr], marker="D", ms=6, color=INK2, ls="none",
            label=f"EUR STR today ({estr:.2f}%)")
    last = p.iloc[-1]
    ax.annotate(f"{last['forward']:.2f}%", (start.iloc[-1], last["forward"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Start of 3-month forward period")
    ax.set_title(f"What the AAA curve prices for euro short rates ({r.as_of:%d %b %Y})")
    ax.legend(loc="lower right", fontsize=8.5)
    _finish(fig, ax, "Forwards include a term premium and a bond-vs-EUR STR basis; grey "
            "lines are illustrations, not estimates.", SOURCE_ECB, out / "forward_path.png")
    return out / "forward_path.png"


def slope_history(r: Results, out: Path) -> Path:
    """2s10s since 2004 with ECB policy turning points marked."""
    s = r.measures["slope_2s10s"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(s.index, s.values, color=BLUE, lw=1.4)
    tps = r.turning_points[r.turning_points["date"] >= s.index[0]]
    lo, hi = s.min(), s.max()
    span = hi - lo
    ax.set_ylim(lo - 0.05 * span, hi + 0.42 * span)  # headroom for the labels
    for i, (_, tp) in enumerate(tps.iterrows()):
        color = ORANGE if tp["kind"] == "first hike" else AQUA
        ax.axvline(tp["date"], color=color, lw=1.2, ls=(0, (3, 2)))
        # Two label rows above the data so turning points close in time do not collide.
        y = hi + (0.40 if i % 2 == 0 else 0.20) * span
        ax.text(tp["date"], y, f" {tp['kind']}\n {tp['date']:%b %Y}", fontsize=7.5,
                color=INK2, va="top")
    ax.annotate(f"{s.iloc[-1]:.0f} bp", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_ylabel("10Y minus 2Y (bp)")
    ax.set_title("Euro area AAA 2s10s slope")
    _finish(fig, ax, "Dashed lines: ECB policy turning points from deposit rate changes "
            "(orange = first hike, teal = first cut).", SOURCE_ECB, out / "slope_2s10s.png")
    return out / "slope_2s10s.png"


def pca_loadings(r: Results, out: Path) -> Path:
    """Loadings of the first three principal components across maturities."""
    lo = r.pca.loadings
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=INK2, lw=0.8)
    for name, color in zip(lo.columns, (BLUE, ORANGE, AQUA), strict=True):
        share = r.pca.explained[name] * 100
        ax.plot(lo.index, lo[name], color=color, marker="o", ms=5, mec=SURFACE,
                label=f"{name} ({share:.1f}% of variance)")
        ax.annotate(name, (lo.index[-1], lo[name].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 33)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Loading")
    ax.set_title("Principal components of daily AAA curve changes")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 0.0))
    start, end = r.pca.window
    _finish(fig, ax, f"Daily changes in bp, {start:%b %Y} to {end:%b %Y}. Level, slope and "
            "curvature, in that order.", SOURCE_ECB, out / "pca_loadings.png")
    return out / "pca_loadings.png"


def eur_us_10y(r: Results, out: Path) -> Path:
    """EUR AAA 10Y par minus US 10Y CMT, since 2015."""
    s = r.par_spreads[10.0].dropna()
    s = s[s.index >= "2015-01-01"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s.index, s.values, color=BLUE, lw=1.4)
    ax.annotate(f"{s.iloc[-1]:.0f} bp", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_ylabel("EUR minus US (bp)")
    ax.set_title("Euro AAA vs US Treasury 10-year yield spread")
    _finish(fig, ax, "Par yields on both sides: ECB AAA par curve minus Treasury "
            "constant-maturity yield.", SOURCE_ALL, out / "eur_us_10y.png")
    return out / "eur_us_10y.png"


def make_all(r: Results, out: Path | None = None) -> list[Path]:
    """Render every chart into figures/."""
    out = out or paths.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    _style()
    return [f(r, out) for f in (curve_snapshot, forward_path, slope_history, pca_loadings,
                                eur_us_10y)]
