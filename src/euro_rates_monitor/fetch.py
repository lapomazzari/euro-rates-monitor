"""Download every catalogued series and cache it, stamped with the retrieval date.

Each file lands in data/raw/<source>/<name>_<YYYY-MM-DD>.csv exactly as the provider
returned it, so a result can always be traced back to the bytes it came from. Nothing
downstream reads from the network: `load.py` parses the latest cached file.

Failure handling (one flaky provider must not stop the run):
* Each series is fetched independently; a failure is recorded, not raised.
* A failed series falls back to its most recent cached file, if one exists. The
  retrieval date in the file name then shows how old it is (see freshness.py).
* Retries: only timeouts, connection errors, HTTP 429 and 5xx, with exponential
  backoff and jitter. A 403/404 (or any other 4xx) will not fix itself: no retry.
* Circuit breaker: after two series in a row fail on the same host, the rest of that
  host's series go straight to the cache instead of burning time on the same outage.
* A total time budget (ERM_FETCH_BUDGET_S, default 300s) caps the whole step.
* Every run writes data/raw/fetch_report.json: fresh / cache / missing per series.
* `erm fetch` exits non-zero only if a required series is missing entirely.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from . import load, paths
from .sources import CATALOGUE, ECB_API, FRED_API, FRED_CSV, Series

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "euro-rates-monitor/0.1 (research project)"}
TIMEOUT = (30, 30)                # seconds: (connect, read between bytes)
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0              # waits of about 2s, then 4s, plus up to 2s of jitter
MAX_RETRY_AFTER_S = 30            # cap on a server's Retry-After request
HOST_FAILURES_TO_TRIP = 2         # consecutive failed series before skipping a host
DEFAULT_BUDGET_S = 300            # whole fetch step; override with ERM_FETCH_BUDGET_S
REPORT = "fetch_report.json"


class FetchError(Exception):
    """A download that did not succeed. `retryable` says whether retrying could help."""

    def __init__(self, reason: str, retryable: bool) -> None:
        super().__init__(reason)
        self.retryable = retryable


def redact(text: str) -> str:
    """Remove API keys from anything that may be logged or written to the report.

    `requests` puts the full URL, query string included, into its error messages.
    """
    return re.sub(r"(api_key=)[^&\s'\"]+", r"\1***", text)


def _retryable_status(code: int) -> bool:
    return code == 429 or 500 <= code <= 599


class Fetcher:
    """HTTP GET with retries, backoff, a per-host circuit breaker and a time budget.

    session, sleep, clock and jitter are injectable so tests run instantly and
    deterministically.
    """

    def __init__(self, session: Any | None = None, budget_s: float | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 jitter: Callable[[], float] = random.random) -> None:
        if budget_s is None:
            budget_s = float(os.environ.get("ERM_FETCH_BUDGET_S", DEFAULT_BUDGET_S))
        self.session = session or requests.Session()
        self.sleep, self.clock, self.jitter = sleep, clock, jitter
        self.budget_s = budget_s
        self.deadline = clock() + budget_s
        self.host_failures: Counter[str] = Counter()

    def get(self, url: str, params: dict[str, str] | None = None) -> bytes:
        """Return the response body or raise FetchError."""
        host = urlparse(url).netloc
        if self.host_failures[host] >= HOST_FAILURES_TO_TRIP:
            raise FetchError(f"skipped: {host} failed for {HOST_FAILURES_TO_TRIP} series "
                               "in a row this run", retryable=True)
        reason = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.clock() >= self.deadline:
                raise FetchError(f"fetch time budget of {self.budget_s:.0f}s exhausted",
                                   retryable=True)
            retry_after = None
            try:
                resp = self.session.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            except (requests.Timeout, requests.ConnectionError) as err:
                reason = f"{type(err).__name__}: {redact(str(err))}"
            else:
                if 200 <= resp.status_code < 300:
                    self.host_failures[host] = 0
                    return resp.content
                if not _retryable_status(resp.status_code):
                    raise FetchError(f"HTTP {resp.status_code} (not retried)",
                                       retryable=False)
                reason = f"HTTP {resp.status_code}"
                ra = resp.headers.get("Retry-After", "")
                retry_after = float(ra) if ra.isdigit() else None
            if attempt == MAX_ATTEMPTS:
                break
            delay = BACKOFF_BASE_S * 2 ** (attempt - 1) + self.jitter() * BACKOFF_BASE_S
            if retry_after is not None:
                delay = max(delay, min(retry_after, MAX_RETRY_AFTER_S))
            if self.clock() + delay >= self.deadline:
                reason += "; fetch time budget would be exceeded by the next retry"
                break
            log.warning("retry %d/%d in %.1fs for %s: %s", attempt, MAX_ATTEMPTS - 1, delay,
                        host, reason)
            self.sleep(delay)
        self.host_failures[host] += 1
        raise FetchError(f"{reason} after {attempt} attempt(s)", retryable=True)


# --------------------------------------------------------------------- per-source fetch

def cache_path(series: Series, day: date) -> Path:
    """Where the raw file for `series` retrieved on `day` lives."""
    return paths.RAW / series.source.lower() / f"{series.name}_{day.isoformat()}.csv"


def _fetch_ecb(series: Series, f: Fetcher) -> bytes:
    """Fetch an ECB key; a key selecting several series ('A+B+C') is split into one
    request per series (the combined query times out on full history) and the CSV
    bodies are concatenated under a single header. Every part must succeed.

    csvdata + dataonly returns one row per observation: KEY, dimensions, TIME_PERIOD,
    OBS_VALUE.
    """
    flow, key = series.key.split("/", 1)
    head, last = key.rsplit(".", 1)
    parts: list[list[str]] = []
    for item in last.split("+"):
        body = f.get(f"{ECB_API}/{flow}/{head}.{item}",
                     {"format": "csvdata", "detail": "dataonly"})
        lines = body.decode("utf-8-sig").splitlines()
        parts.append(lines if not parts else lines[1:])
    return ("\n".join(line for p in parts for line in p) + "\n").encode()


def fred_path() -> str:
    """'api' when FRED_API_KEY is set (non-empty), else 'csv'."""
    return "api" if os.environ.get("FRED_API_KEY", "").strip() else "csv"


def _fetch_fred(series: Series, f: Fetcher) -> bytes:
    """Official FRED API when FRED_API_KEY is set, public CSV download otherwise.

    Both paths are written in the same two-column layout (DATE, VALUE) so the parser
    does not care which one ran. FRED marks missing days with '.'. An unset secret in
    GitHub Actions arrives as an empty string, which counts as unset.
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        return f.get(FRED_CSV, {"id": series.key})
    raw = f.get(FRED_API, {"series_id": series.key, "api_key": key, "file_type": "json"})
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["observation_date", series.key])
    for obs in json.loads(raw)["observations"]:
        writer.writerow([obs["date"], obs["value"]])
    return out.getvalue().encode()


def _download(series: Series, f: Fetcher) -> bytes:
    if series.source == "ECB":
        return _fetch_ecb(series, f)
    if series.source == "FRED":
        return _fetch_fred(series, f)
    return f.get(series.key)


# ---------------------------------------------------------------------------- results

@dataclass
class FetchResult:
    """Outcome for one series. status: 'fresh', 'cache' or 'missing'."""

    name: str
    source: str
    required: bool
    status: str
    retrieved: str | None = None
    file: str | None = None
    error: str | None = None


def fetch_one(series: Series, fetcher: Fetcher, day: date | None = None,
              force: bool = False) -> FetchResult:
    """Download one series; on failure fall back to its latest cached file."""
    day = day or date.today()
    target = cache_path(series, day)
    base = {"name": series.name, "source": series.source, "required": series.required}
    if target.exists() and not force:
        log.info("cached  %s (already downloaded today)", target.relative_to(paths.ROOT))
        return FetchResult(**base, status="fresh", retrieved=day.isoformat(),
                           file=str(target.relative_to(paths.ROOT)))
    try:
        body = _download(series, fetcher)
    except Exception as err:  # noqa: BLE001 - one series must never stop the others
        reason = redact(str(err)) or type(err).__name__
        if load.has(series.name):
            cached = load.latest_file(series.name)
            log.warning("FAILED  %s: %s -> using cache %s", series.name, reason, cached.name)
            return FetchResult(**base, status="cache", retrieved=load.retrieval_date(
                series.name), file=str(cached.relative_to(paths.ROOT)), error=reason)
        log.error("FAILED  %s: %s -> no cache available", series.name, reason)
        return FetchResult(**base, status="missing", error=reason)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    _record(series, target, body)
    log.info("fetched %s (%d kB)", target.relative_to(paths.ROOT), len(body) // 1024)
    return FetchResult(**base, status="fresh", retrieved=day.isoformat(),
                       file=str(target.relative_to(paths.ROOT)))


def _record(series: Series, target: Path, body: bytes) -> None:
    """Append retrieval metadata (time, URL, hash) to data/raw/manifest.jsonl."""
    entry = {
        "name": series.name,
        "file": str(target.relative_to(paths.ROOT)),
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_url": series.url,
        "fred_path": fred_path() if series.source == "FRED" else None,
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    with (paths.RAW / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def fetch_all(force: bool = False, fetcher: Fetcher | None = None,
              day: date | None = None) -> list[FetchResult]:
    """Fetch every series, never stopping on a failure; write the fetch report."""
    fetcher = fetcher or Fetcher()
    started = time.monotonic()
    results = [fetch_one(s, fetcher, day=day, force=force) for s in CATALOGUE]
    counts = Counter(r.status for r in results)
    log.info("fetch summary: %d fresh, %d from cache, %d missing (%.0fs)",
             counts["fresh"], counts["cache"], counts["missing"], time.monotonic() - started)
    paths.RAW.mkdir(parents=True, exist_ok=True)
    report = {
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget_s": fetcher.budget_s,
        "fred_path": fred_path(),
        "results": [asdict(r) for r in results],
    }
    (paths.RAW / REPORT).write_text(json.dumps(report, indent=2))
    return results


def missing_required(results: list[FetchResult]) -> list[str]:
    """Required series with neither a fresh download nor a cached copy."""
    return [r.name for r in results if r.required and r.status == "missing"]


def summary_markdown(report: dict[str, Any]) -> str:
    """The fetch report as a markdown table (for the Actions run summary page)."""
    rows = report["results"]
    counts = Counter(r["status"] for r in rows)
    icon = {"fresh": "fresh", "cache": "**cache**", "missing": "**MISSING**"}
    lines = [
        "## Data fetch",
        "",
        f"Run {report['run_utc']} UTC; FRED via {report['fred_path']}; budget "
        f"{report['budget_s']:.0f}s. {counts['fresh']} fresh, {counts['cache']} from cache, "
        f"{counts['missing']} missing.",
        "",
        "| Series | Source | Status | Retrieved | Required | Error |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = (r["error"] or "").replace("|", "/")[:120]
        lines.append(f"| `{r['name']}` | {r['source']} | {icon[r['status']]} | "
                     f"{r['retrieved'] or ''} | {'yes' if r['required'] else ''} | {err} |")
    return "\n".join(lines) + "\n"
