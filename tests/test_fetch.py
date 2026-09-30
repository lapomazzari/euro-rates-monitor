"""Fetch resilience: retries, circuit breaker, budget, cache fallback, redaction, and a
build that completes without the US data."""

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest
import requests

from euro_rates_monitor import build, charts, cli, fetch, freshness, paths
from euro_rates_monitor.note import render_scaffold, unknown_tokens
from euro_rates_monitor.sources import BY_NAME, Series

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


class Resp:
    def __init__(self, status: int, body: bytes = b"ok", headers: dict | None = None):
        self.status_code, self.content, self.headers = status, body, headers or {}


class Session:
    """Returns (or raises) the scripted outcomes in order and counts calls per host."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def get(self, url, **_):
        self.calls.append(url)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _fetcher(session, budget=300.0):
    clock, sleeps = Clock(), []

    def sleep(s):
        sleeps.append(s)
        clock.t += s

    f = fetch.Fetcher(session=session, budget_s=budget, sleep=sleep, clock=clock,
                      jitter=lambda: 0.0)
    return f, sleeps, clock


URL = "https://example.org/data"


# ---------------------------------------------------------------------------- retries

@pytest.mark.parametrize("first", [Resp(503), Resp(502), requests.Timeout("slow"),
                                   requests.ConnectionError("reset")])
def test_retryable_failures_are_retried(first):
    f, sleeps, _ = _fetcher(Session(first, Resp(200, b"data")))
    assert f.get(URL) == b"data"
    assert sleeps == [fetch.BACKOFF_BASE_S]


def test_429_honours_retry_after_within_the_cap():
    f, sleeps, _ = _fetcher(Session(Resp(429, headers={"Retry-After": "10"}), Resp(200)))
    f.get(URL)
    assert sleeps == [10.0]
    f, sleeps, _ = _fetcher(Session(Resp(429, headers={"Retry-After": "999"}), Resp(200)))
    f.get(URL)
    assert sleeps == [fetch.MAX_RETRY_AFTER_S]


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_never_retried(code):
    session = Session(Resp(code))
    f, sleeps, _ = _fetcher(session)
    with pytest.raises(fetch.FetchError) as err:
        f.get(URL)
    assert len(session.calls) == 1 and sleeps == []
    assert err.value.retryable is False


def test_gives_up_after_max_attempts_with_exponential_backoff():
    session = Session(*[Resp(500)] * fetch.MAX_ATTEMPTS)
    f, sleeps, _ = _fetcher(session)
    with pytest.raises(fetch.FetchError, match=f"after {fetch.MAX_ATTEMPTS}"):
        f.get(URL)
    assert sleeps == [fetch.BACKOFF_BASE_S * 2**i for i in range(fetch.MAX_ATTEMPTS - 1)]


def test_circuit_breaker_skips_a_failing_host_but_not_others():
    session = Session(*[requests.Timeout()] * (2 * fetch.MAX_ATTEMPTS), Resp(200))
    f, _, _ = _fetcher(session)
    for _ in range(fetch.HOST_FAILURES_TO_TRIP):
        with pytest.raises(fetch.FetchError):
            f.get(URL)
    calls = len(session.calls)
    with pytest.raises(fetch.FetchError, match="skipped"):
        f.get(URL + "/other-series")
    assert len(session.calls) == calls                      # no request made
    assert f.get("https://another-host.org/x") == b"ok"     # other hosts unaffected


def test_budget_caps_the_time_spent():
    f, _, clock = _fetcher(Session(), budget=60)
    clock.t = 61
    with pytest.raises(fetch.FetchError, match="budget"):
        f.get(URL)
    # A retry that would overshoot the budget is not attempted.
    session = Session(Resp(503), Resp(200))
    f, sleeps, _ = _fetcher(session, budget=1)
    with pytest.raises(fetch.FetchError, match="budget"):
        f.get(URL)
    assert len(session.calls) == 1 and sleeps == []


def test_api_key_is_redacted():
    assert fetch.redact("for url: https://x/obs?series_id=A&api_key=abc123&x=1") == (
        "for url: https://x/obs?series_id=A&api_key=***&x=1")


# --------------------------------------------------------------- fallback, non-fatal

@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    (tmp_path / "raw").mkdir()
    return tmp_path / "raw"


def _cached(raw: Path, series: Series, day: date, text: str = "a,b\n2026-01-01,1\n") -> None:
    d = raw / series.source.lower()
    d.mkdir(exist_ok=True)
    (d / f"{series.name}_{day.isoformat()}.csv").write_text(text)


def test_failed_series_is_non_fatal_and_reported(raw, monkeypatch):
    good, bad = BY_NAME["ecb_dfr"], BY_NAME["fred_dgs10"]
    monkeypatch.setattr(fetch, "CATALOGUE", [bad, good])
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class Mixed:
        def get(self, url, **_):
            if "stlouisfed" in url:
                return Resp(403)
            return Resp(200, b"KEY,TIME_PERIOD,OBS_VALUE\nx,2026-09-29,2.5\n")

    results = fetch.fetch_all(fetcher=fetch.Fetcher(session=Mixed(), sleep=lambda s: None))
    status = {r.name: r.status for r in results}
    assert status == {"fred_dgs10": "missing", "ecb_dfr": "fresh"}
    report = json.loads((raw / fetch.REPORT).read_text())
    assert [r["status"] for r in report["results"]] == ["missing", "fresh"]
    assert "HTTP 403" in report["results"][0]["error"]
    assert fetch.missing_required(results) == []            # dgs10 is optional


def test_cache_fallback_records_the_old_retrieval_date(raw):
    s = BY_NAME["fred_dgs10"]
    old = date.today() - timedelta(days=5)
    _cached(raw, s, old)
    f, _, _ = _fetcher(Session(*[requests.Timeout()] * fetch.MAX_ATTEMPTS))
    r = fetch.fetch_one(s, f)
    assert (r.status, r.retrieved) == ("cache", old.isoformat())
    assert "Timeout" in r.error


def test_missing_required_series_fails_the_command(raw, monkeypatch):
    missing = fetch.FetchResult("ecb_spot_aaa", "ECB", True, "missing", error="HTTP 404")
    monkeypatch.setattr(fetch, "fetch_all", lambda **_: [missing])
    with pytest.raises(SystemExit, match="ecb_spot_aaa"):
        cli.main(["fetch"])
    optional = fetch.FetchResult("fred_dgs10", "FRED", False, "missing", error="HTTP 404")
    monkeypatch.setattr(fetch, "fetch_all", lambda **_: [optional])
    assert cli.main(["fetch"]) == 0


def test_key_never_reaches_the_report(raw, monkeypatch):
    monkeypatch.setattr(fetch, "CATALOGUE", [BY_NAME["fred_dgs10"]])
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    err = requests.ConnectionError(f"{fetch.FRED_API}?series_id=DGS10&api_key=SECRETKEY123")
    session = Session(*[err] * fetch.MAX_ATTEMPTS)
    fetch.fetch_all(fetcher=fetch.Fetcher(session=session, sleep=lambda s: None))
    assert "SECRETKEY123" not in (raw / fetch.REPORT).read_text()


def test_summary_markdown_reads_the_report():
    report = {"run_utc": "2026-10-03T06:01:00+00:00", "budget_s": 300, "fred_path": "csv",
              "results": [
                  {"name": "ecb_dfr", "source": "ECB", "required": True, "status": "fresh",
                   "retrieved": "2026-10-03", "error": None},
                  {"name": "fred_dgs10", "source": "FRED", "required": False,
                   "status": "cache", "retrieved": "2026-09-26", "error": "Timeout"},
              ]}
    md = fetch.summary_markdown(report)
    assert "1 fresh, 1 from cache, 0 missing" in md
    assert "| `fred_dgs10` | FRED | **cache** | 2026-09-26 |  | Timeout |" in md


# ------------------------------------------------------------------------ freshness

def _copy_sample(raw: Path, skip_sources: tuple[str, ...] = (), day: date | None = None):
    """Populate a raw cache from the committed sample, optionally re-dating it."""
    for f in SAMPLE.glob("*.csv"):
        name = f.stem.rsplit("_", 1)[0]
        s = BY_NAME[name]
        if s.source in skip_sources:
            continue
        d = raw / s.source.lower()
        d.mkdir(exist_ok=True)
        stamp = (day or date.fromisoformat(f.stem.rsplit("_", 1)[1])).isoformat()
        shutil.copy(f, d / f"{name}_{stamp}.csv")


@pytest.mark.skipif(not SAMPLE.exists(), reason="no sample data")
def test_stale_flag_is_about_retrieval_age_only(raw):
    run = date(2026, 10, 3)
    _copy_sample(raw, day=run - timedelta(days=1))
    fresh = freshness.series_freshness(run)
    # The Fed Board curve's last observation lags by design; it is not stale.
    assert fresh["fed_gsw"]["stale"] is False
    assert fresh["fed_gsw"]["last_observation"] < "2026-09-30"
    for f in (raw / "fred").glob("fred_dgs10_*.csv"):
        f.rename(f.with_name(f"fred_dgs10_{(run - timedelta(days=5)).isoformat()}.csv"))
    fresh = freshness.series_freshness(run)
    assert fresh["fred_dgs10"]["stale"] is True
    assert fresh["fred_dgs10"]["retrieval_age_days"] == 5
    summary = freshness.source_summary(fresh)
    assert summary["FRED"]["stale"] == ["fred_dgs10"]
    assert summary["ECB Data Portal"]["stale"] == []


# --------------------------------------------------------------- build without US data

@pytest.mark.skipif(not SAMPLE.exists(), reason="no sample data")
def test_build_completes_without_us_data(raw, tmp_path):
    _copy_sample(raw, skip_sources=("FRED", "FEDBOARD"))
    res = build.run(run_date=date(2026, 9, 30))
    m = json.loads(json.dumps(res.metrics, default=str))
    assert {"cross_market", "fx", "real_rates_us"} <= set(m["sections_unavailable"])
    assert "credit" not in m["sections_unavailable"]
    assert not any(k.startswith(("eur_minus_us", "eurusd", "ust_")) for k in m)
    assert m["freshness"]["sources"]["FRED"]["n_missing"] == len(
        [s for s in BY_NAME.values() if s.source == "FRED"])
    assert "fwd3m_in_1y_pct" in m and "all_minus_aaa_10y_bp" in m
    text = render_scaffold(m, "img/chart.png")
    assert "- **FX:** Unavailable this build" in text
    assert "FRED unavailable" in text
    assert unknown_tokens(text, m) == []
    charts._style()
    out = charts.eur_us_10y(res, tmp_path)
    assert out.exists() and out.stat().st_size > 0
