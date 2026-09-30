"""How fresh is each input? Two separate questions, never mixed up.

1. Retrieval age: how long ago did we download the file we are using? If a download
   failed and an older cached file was used instead, this is where it shows. A series
   is flagged STALE only on this measure.
2. Last observation: the most recent date inside the data. Some sources lag by design
   (the Fed Board zero curve by 1-2 weeks, HICP is monthly, the SPF quarterly), so a
   late last observation is reported but never flagged: "the Fed curve is two weeks
   behind by design" is not "we failed to download it".
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from . import load
from .sources import CATALOGUE, Series

# Judgement call for a weekly job: a file retrieved more than 3 days before the build
# means this week's download failed and an older cache was used. Short enough to catch
# a missed weekly refresh, long enough not to flag a build run the day after a fetch.
STALE_AFTER_DAYS = 3

SOURCE_LABELS = {"ECB": "ECB Data Portal", "FRED": "FRED", "FEDBOARD": "Fed Board"}


def last_observation(series: Series) -> str:
    """Date of the most recent observation in the cached file (YYYY-MM-DD)."""
    if series.source == "ECB":
        periods = pd.read_csv(load.latest_file(series.name), usecols=["TIME_PERIOD"])
        last = periods["TIME_PERIOD"].max()
        return str(last)
    if series.source == "FRED":
        return load.fred_series(series.name).index[-1].date().isoformat()
    return load.gsw().index[-1].date().isoformat()


def series_freshness(run_date: date | None = None) -> dict[str, dict[str, Any]]:
    """Per-series retrieval date, retrieval age, stale flag and last observation."""
    run_date = run_date or date.today()
    out: dict[str, dict[str, Any]] = {}
    for s in CATALOGUE:
        if not load.has(s.name):
            out[s.name] = {"source": s.source, "status": "missing", "stale": False,
                           "required": s.required}
            continue
        retrieved = date.fromisoformat(load.retrieval_date(s.name))
        age = (run_date - retrieved).days
        out[s.name] = {
            "source": s.source,
            "status": "stale" if age > STALE_AFTER_DAYS else "ok",
            "retrieved": retrieved.isoformat(),
            "retrieval_age_days": age,
            "stale": age > STALE_AFTER_DAYS,
            "last_observation": last_observation(s),
            "required": s.required,
        }
    return out


def source_summary(per_series: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One line per source: oldest retrieval, whether anything is stale or missing."""
    out: dict[str, dict[str, Any]] = {}
    for key, label in SOURCE_LABELS.items():
        rows = {n: r for n, r in per_series.items() if r["source"] == key}
        present = [r for r in rows.values() if r["status"] != "missing"]
        stale = [n for n, r in rows.items() if r["stale"]]
        missing = [n for n, r in rows.items() if r["status"] == "missing"]
        out[label] = {
            "oldest_retrieved": min((r["retrieved"] for r in present), default=None),
            "stale": stale, "n_stale": len(stale),
            "missing": missing, "n_missing": len(missing),
        }
    return out


def data_line(fresh: dict[str, Any]) -> str:
    """Human-readable 'Data retrieved' line for the note footer and the README headline.

    Every date and number in it comes from the freshness dictionary, so it passes the
    note's number check.
    """
    parts = []
    for label, s in fresh["sources"].items():
        if s["oldest_retrieved"] is None:
            parts.append(f"{label} unavailable")
            continue
        text = f"{label} {s['oldest_retrieved']}"
        if s["stale"]:
            text += f" (**stale**: cached copy, {s['n_stale']} series)"
        if s["missing"]:
            text += f" ({s['n_missing']} series unavailable)"
        parts.append(text)
    return ("; ".join(parts) + f". Stale = retrieved more than "
            f"{fresh['stale_after_days']} days before this build (run {fresh['run_date']}).")
