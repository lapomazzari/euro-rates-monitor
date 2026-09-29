"""Download every catalogued series and cache it, stamped with the retrieval date.

Each file lands in data/raw/<source>/<name>_<YYYY-MM-DD>.csv exactly as the provider
returned it, so a result can always be traced back to the bytes it came from. Nothing
downstream reads from the network: `load.py` parses the latest cached file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from . import paths
from .sources import CATALOGUE, ECB_API, FRED_API, FRED_CSV, Series

log = logging.getLogger(__name__)
TIMEOUT = 120
HEADERS = {"User-Agent": "euro-rates-monitor/0.1 (research project)"}


def cache_path(series: Series, day: date) -> Path:
    """Where the raw file for `series` retrieved on `day` lives."""
    return paths.RAW / series.source.lower() / f"{series.name}_{day.isoformat()}.csv"


def _get(url: str, params: dict[str, str] | None = None, attempts: int = 3) -> bytes:
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except (requests.Timeout, requests.ConnectionError):
            if i == attempts - 1:
                raise
            log.warning("retrying %s", url)
    raise AssertionError("unreachable")


def _fetch_ecb(series: Series) -> bytes:
    """Fetch an ECB key; a key selecting several series ('A+B+C') is split into one
    request per series (the combined query times out on full history) and the CSV
    bodies are concatenated under a single header.

    csvdata + dataonly returns one row per observation: KEY, dimensions, TIME_PERIOD,
    OBS_VALUE.
    """
    flow, key = series.key.split("/", 1)
    head, last = key.rsplit(".", 1)
    parts = []
    for item in last.split("+"):
        body = _get(f"{ECB_API}/{flow}/{head}.{item}", {"format": "csvdata", "detail": "dataonly"})
        lines = body.decode("utf-8-sig").splitlines()
        parts.append(lines if not parts else lines[1:])
    return ("\n".join(line for p in parts for line in p) + "\n").encode()


def _fetch_fred(series: Series) -> bytes:
    """Official FRED API when FRED_API_KEY is set, public CSV download otherwise.

    Both paths are written in the same two-column layout (DATE, VALUE) so the parser
    does not care which one ran. FRED marks missing days with '.'.
    """
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return _get(FRED_CSV, {"id": series.key})
    raw = _get(FRED_API, {"series_id": series.key, "api_key": key, "file_type": "json"})
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["observation_date", series.key])
    for obs in json.loads(raw)["observations"]:
        writer.writerow([obs["date"], obs["value"]])
    return out.getvalue().encode()


def fetch_one(series: Series, day: date | None = None, force: bool = False) -> Path:
    """Download one series unless today's cache file already exists."""
    day = day or date.today()
    target = cache_path(series, day)
    if target.exists() and not force:
        log.info("cached  %s", target.relative_to(paths.ROOT))
        return target
    if series.source == "ECB":
        body = _fetch_ecb(series)
    elif series.source == "FRED":
        body = _fetch_fred(series)
    else:
        body = _get(series.key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    _record(series, target, body)
    log.info("fetched %s (%d kB)", target.relative_to(paths.ROOT), len(body) // 1024)
    return target


def _record(series: Series, target: Path, body: bytes) -> None:
    """Append retrieval metadata (time, URL, hash) to data/raw/manifest.jsonl."""
    entry = {
        "name": series.name,
        "file": str(target.relative_to(paths.ROOT)),
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_url": series.url,
        "fred_path": ("api" if os.environ.get("FRED_API_KEY") else "csv")
        if series.source == "FRED" else None,
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    with (paths.RAW / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def fetch_all(force: bool = False) -> list[Path]:
    """Download (or reuse today's cache of) every catalogued series."""
    return [fetch_one(s, force=force) for s in CATALOGUE]
