"""The weekly note: a scaffold to write by hand, and the check that keeps it honest.

Committed notes in notes/ are written by hand from the computed figures.

* `erm note` (default) writes a scaffold: the computed figures, dates and chart, with
  a [WRITE: ...] placeholder where each piece of prose goes. It contains no generated
  prose.
* `erm note --llm` (optional) has Claude draft the prose from the metrics dictionary
  only. The draft is marked as unreviewed and must be edited by hand before it passes
  the check.
* `erm check-note` extracts every number and date from a note and verifies it against
  the figures file for the note's date (data/processed/metrics_<date>.json). It also
  fails on unfilled placeholders and on the unreviewed marker. CI runs it on every note.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import paths

log = logging.getLogger(__name__)

MODEL = os.environ.get("ERM_MODEL", "claude-opus-5-5")
CHART = "forward_path.png"


class NumberValidationError(RuntimeError):
    """The drafted text contains numbers or dates that are not in the metrics dictionary."""


# --------------------------------------------------------------------------- validation

UNREVIEWED = "<!-- erm:unreviewed -->"
PLACEHOLDER = "[WRITE:"

_LABEL_PATTERNS = [
    r"\b\d+s(?:\d+s)*\d+s\b",                  # curve labels: 2s10s, 10s30s, 2s5s10s
    r"\b\d+(?:\.\d+)?[YMW]\b",                 # tenors / horizons: 10Y, 3M, 1Y, 1W
    r"\b\d+-(?:month|year|week|day)\b",        # adjectival tenors: 3-month, 10-year
    r"\bPC\d\b",                               # PC1..PC3
]
_NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?\d+(?:,\d{3})*(?:\.\d+)?")
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
# Anything that looks like a date. Alternatives are tried in order at each position,
# so full dates win over their prefixes (2026-09-28 before 2026-09).
_DATE = re.compile(
    r"(?<![\w-])(?:"
    r"\d{4}-\d{2}-\d{2}|\d{4}-Q[1-4]|Q[1-4] \d{4}|\d{4} Q[1-4]|\d{4}-\d{2}|"
    rf"\d{{1,2}} (?:{_MONTHS})\.? \d{{4}}|(?:{_MONTHS})\.? \d{{1,2}}, \d{{4}}|"
    rf"(?:{_MONTHS})\.? \d{{4}}|\d{{1,2}} (?:{_MONTHS})\b"
    r")(?![\w-])")


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(".", "")).strip().lower()


def _date_renderings(value: str) -> list[str]:
    """Ways a date from the dictionary may legitimately appear in text."""
    out = [value]
    m = re.fullmatch(r"(\d{4})-Q([1-4])", value)
    if m:
        return out + [f"Q{m[2]} {m[1]}", f"{m[1]} Q{m[2]}"]
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(value, fmt)
        except ValueError:
            continue
        months = [f"{d:%B}", f"{d:%b}"] + (["Sept"] if d.month == 9 else [])
        if fmt == "%Y-%m-%d":
            for day in (str(d.day), f"{d.day:02d}"):
                for mon in months:
                    out += [f"{day} {mon} {d.year}", f"{day} {mon}", f"{mon} {day}, {d.year}"]
        out += [f"{mon} {d.year}" for mon in months]
    return out


def allowed_dates(metrics: dict[str, Any]) -> set[str]:
    """Normalised renderings of every date string in the dictionary."""
    dates = [v for v in _walk(metrics) if isinstance(v, str)
             and re.fullmatch(r"\d{4}-(\d{2}(-\d{2})?|Q[1-4])", v)]
    return {_norm(r) for d in dates for r in _date_renderings(d)}


def allowed_numbers(metrics: dict[str, Any]) -> set[float]:
    """Absolute values of every number in the dictionary (signs may become words)."""
    return {abs(float(v)) for v in _walk(metrics)
            if isinstance(v, int | float) and not isinstance(v, bool)}


def unknown_tokens(text: str, metrics: dict[str, Any]) -> list[str]:
    """Dates and numbers in `text` that do not appear in `metrics`.

    Dates must match a date in the dictionary (in any common rendering). Numbers must
    match a dictionary value exactly, up to sign: '2.50' matches 2.5, but '3.6' does
    not match 3.63, so re-rounding fails, and so does arithmetic on figures. Tenor
    labels (10Y, 2s10s, 3-month), link targets and HTML comments are not claims and
    are ignored.
    """
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "] ", text)          # markdown link / image targets
    bad: list[str] = []
    dates_ok = allowed_dates(metrics)

    def _date(m: re.Match[str]) -> str:
        if _norm(m.group(0)) not in dates_ok:
            bad.append(m.group(0))
        return " "

    text = _DATE.sub(_date, text)
    for pat in _LABEL_PATTERNS:
        text = re.sub(pat, " ", text)
    allowed = allowed_numbers(metrics)
    for tok in _NUMBER.findall(text):
        val = abs(float(tok.replace("\u2212", "-").replace(",", "")))
        if not any(abs(val - a) < 1e-9 for a in allowed):
            bad.append(tok)
    return bad


def validate(sections: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Raise NumberValidationError listing every number or date not in the dictionary."""
    bad = {k: unknown_tokens(v if isinstance(v, str) else " ".join(v), metrics)
           for k, v in sections.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise NumberValidationError(
            "numbers or dates not found in the metrics dictionary: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()))


def check_note(path: Path, processed: Path | None = None) -> list[str]:
    """Problems with one note file; an empty list means it passes.

    The figures file is chosen by the note's name: notes/2026-09-28.md is checked
    against data/processed/metrics_2026-09-28.json.
    """
    processed = processed or paths.PROCESSED
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
        return [f"file name must be YYYY-MM-DD.md, got {path.name}"]
    figures = processed / f"metrics_{path.stem}.json"
    if not figures.exists():
        return [f"no figures file for this date ({figures.name}); run `erm build` for it"]
    text = path.read_text()
    problems = []
    if UNREVIEWED in text:
        problems.append("still marked unreviewed: delete the scaffold/draft banner lines")
    if PLACEHOLDER in text:
        problems.append(f"{text.count(PLACEHOLDER)} unfilled {PLACEHOLDER} ...] placeholder(s)")
    bad = unknown_tokens(text, json.loads(figures.read_text()))
    if bad:
        problems.append(f"not in {figures.name}: {', '.join(bad)}")
    return problems


def note_files(targets: Iterable[Path]) -> list[Path]:
    """Expand directories to the YYYY-MM-DD.md notes they contain."""
    out: list[Path] = []
    for t in targets:
        out += sorted(t.glob("????-??-??.md")) if t.is_dir() else [t]
    return out


# --------------------------------------------------------------------------- LLM draft

SYSTEM = """You draft a short weekly euro rates market note for a trading desk.

Hard rules:
1. Use ONLY the figures in the JSON the user provides. Copy numbers exactly as written
   there: same digits and decimals, no rounding, no arithmetic (do not add, subtract or
   average figures), no numbers from memory.
2. No outside knowledge: no news, events, data releases, central bank communication,
   people or reasons for moves that are not in the JSON. Do not explain why something
   moved.
3. No opinions or forecasts about future policy or markets. Describe what the curve
   prices ("forwards imply"), never what will or should happen.
4. Write changes as words plus magnitude, e.g. "rose 9bp", "fell 4bp". Units: % for
   levels of yields and rates, bp for changes and spreads.
5. Refer to maturities and horizons with tenor labels (2Y, 10Y, 3M, in 1Y, in 2Y) and to
   dates exactly as written in the JSON.
6. Forwards come from the AAA government curve and include a term premium; they are
   not a clean expectation of the deposit rate. Say this once in the ECB section. The
   term-premium scenario figures are sensitivity illustrations and must never be
   presented as the headline.
7. Plain, precise desk English. No hype, no hedging filler, no emoji."""

INSTRUCTIONS = """Draft these sections from the JSON below.

- moved: exactly three bullets, one per entry of top_moves_1w in that order. Each bullet
  gives the latest level and the 1-week change; add the 1-month change or 1-year-range
  context from the other fields where relevant.
- ecb_pricing: 2-3 sentences on what the AAA curve's 3M forwards imply relative to the
  deposit rate in 6M, 1Y and 2Y; mention the path relative to the curve's own 3M rate
  and the 3M-vs-EUR STR gap; one clause on the term premium caveat.
- fx_bullet: one sentence on EUR/USD (level, 1-week change) and its relation to the
  EUR-US 2Y differential (correlations, level versus the 1-year fit).
- credit_bullet: one sentence on the all-issuer minus AAA spread at 5Y and 10Y (level,
  1-week change, 1-year range).
- hedging: one paragraph (3-4 sentences) on what the pricing means mechanically for a
  client hedging floating-rate exposure linked to euro short rates: current EUR STR and
  deposit rate versus what forwards imply; the AAA 2Y yield versus EUR STR as a proxy
  for the cost of fixing for two years, noting swap rates are not in the dataset so the
  actual hedge level will differ. No recommendation.

JSON:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "moved": {"type": "array", "items": {"type": "string"}},
        "ecb_pricing": {"type": "string"},
        "fx_bullet": {"type": "string"},
        "credit_bullet": {"type": "string"},
        "hedging": {"type": "string"},
    },
    "required": ["moved", "ecb_pricing", "fx_bullet", "credit_bullet", "hedging"],
    "additionalProperties": False,
}


def draft_llm(metrics: dict[str, Any], client: Any | None = None,
              attempts: int = 2) -> dict[str, Any]:
    """Ask Claude for the prose, then validate every number. One retry on failure.

    `client` can be injected for tests; by default the Anthropic client reads the key
    from ANTHROPIC_API_KEY.
    """
    import anthropic

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set; run `erm note` without --llm")
        client = anthropic.Anthropic()
    payload = INSTRUCTIONS + json.dumps(metrics, indent=1)
    feedback = ""
    for attempt in range(1, attempts + 1):
        # fallbacks="default": if a safety classifier declines the request, the API
        # re-runs it on Anthropic's recommended fallback model instead of refusing.
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": payload + feedback}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model declined the request: {resp.stop_details}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("model output was truncated (max_tokens)")
        text = next(b.text for b in resp.content if b.type == "text")
        sections = json.loads(text)
        try:
            if len(sections["moved"]) != 3:
                raise NumberValidationError("expected exactly three 'moved' bullets")
            validate(sections, metrics)
            return sections
        except NumberValidationError as err:
            log.warning("attempt %d failed validation: %s", attempt, err)
            if attempt == attempts:
                raise
            feedback = (f"\n\nYour previous draft was rejected: {err}. Use only numbers "
                        "that appear in the JSON, exactly as written.")
    raise AssertionError("unreachable")


# ------------------------------------------------------------------- formatting helpers

def _n(v: float | int) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _p(v: float | int) -> str:
    """Rate levels always with two decimals (2.50%, not 2.5%); same value, so it validates."""
    return f"{v:.2f}"


def _s(v: float | int | None, unit: str = "bp") -> str:
    """Signed change for the figures tables: '+16bp', '-4bp', 'n/a'."""
    return "n/a" if v is None else f"{v:+g}{unit}"


def _rel(v: float | int, unit: str, ref: str) -> str:
    """'12bp above EUR STR' / '1.6% below its fit'."""
    return f"{_n(abs(v))}{unit} {'above' if v >= 0 else 'below'} {ref}"


# ----------------------------------------------------------------------------- output

def _title(m: dict[str, Any]) -> str:
    d = datetime.strptime(m["as_of"], "%Y-%m-%d")
    return f"# Euro rates weekly: {d.day} {d:%B %Y}"


def _footer(m: dict[str, Any]) -> list[str]:
    return [
        "---",
        f"*Data: ECB curve and EUR STR as of {m['as_of']}; US Treasury yields to "
        f"{m['eur_us_spread_date']}; US zero curve {m['us_zero_curve_date']}; HICP "
        f"{m['hicp_month']}. Forwards are from the AAA government curve, include a term "
        "premium and are not OIS-implied. Source: ECB statistics; Eurostat; FRED "
        "(St. Louis Fed); Federal Reserve Board; own calculations. Not investment advice.*",
        "",
    ]


def _figures(m: dict[str, Any]) -> list[str]:
    """Tables of the computed figures the writer works from. Every cell is from `m`."""
    moves = [f"| {mv['measure']} | {_n(mv['latest'])}{mv['latest_unit']} | "
             f"{_s(mv['chg_1w_bp'])} | {_n(mv['size_vs_typical_week'])}x |"
             for mv in m["top_moves_1w"]]
    fwd = [f"| in {h.upper()} | {_p(m[f'fwd3m_in_{h}_pct'])}% | "
           f"{_s(m[f'fwd3m_in_{h}_minus_dfr_bp'])} | {_s(m[f'fwd3m_in_{h}_minus_aaa3m_bp'])} | "
           f"{_s(m[f'fwd3m_in_{h}_chg_1w_bp'])} |" for h in ("6m", "1y", "2y")]
    ten_range = f"{_p(m['aaa_10y_1y_low_pct'])} to {_p(m['aaa_10y_1y_high_pct'])}%"
    curve = [f"| AAA {t}Y | {_p(m[f'aaa_{t}y_pct'])}% | {_s(m[f'aaa_{t}y_chg_1w_bp'])} | "
             f"{_s(m[f'aaa_{t}y_chg_1m_bp'])} | {ten_range if t == 10 else ''} |"
             for t in (2, 5, 10, 30)]
    curve += [f"| {label} | {_n(m[f'{k}_bp'])}bp | {_s(m[f'{k}_chg_1w_bp'])} | "
              f"{_s(m[f'{k}_chg_1m_bp'])} | {_n(m[f'{k}_1y_low_bp'])} to "
              f"{_n(m[f'{k}_1y_high_bp'])}bp |"
              for label, k in (("2s10s", "slope_2s10s"), ("10s30s", "slope_10s30s"),
                               ("2s5s10s", "fly_2s5s10s"))]
    cross = [f"| EUR-US {t}Y par spread | {_n(m[f'eur_minus_us_{t}y_par_bp'])}bp | "
             f"{_s(m[f'eur_minus_us_{t}y_par_chg_1w_bp'])} | "
             f"{_s(m[f'eur_minus_us_{t}y_par_chg_1m_bp'])} | |" for t in (2, 10)]
    cross += [f"| All-issuer minus AAA {t}Y | {_n(m[f'all_minus_aaa_{t}y_bp'])}bp | "
              f"{_s(m[f'all_minus_aaa_{t}y_chg_1w_bp'])} | "
              f"{_s(m[f'all_minus_aaa_{t}y_chg_1m_bp'])} | "
              f"{_n(m[f'all_minus_aaa_{t}y_1y_low_bp'])} to "
              f"{_n(m[f'all_minus_aaa_{t}y_1y_high_bp'])}bp |" for t in (5, 10)]
    head = "| | Latest | 1W change | 1M change | 1Y range |\n|---|---|---|---|---|"
    return [
        "## Figures (computed by `erm build`)",
        "",
        "Every figure below is in `data/processed/metrics_" + m["as_of"] + ".json`. Keep, "
        "trim or delete this section; `erm check-note` checks whatever remains.",
        "",
        "**What moved** (largest 1W changes relative to a typical week over the past year)",
        "",
        "| Measure | Latest | 1W change | Size vs typical week |",
        "|---|---|---|---|",
        *moves,
        "",
        f"**ECB pricing**: deposit rate {_p(m['ecb_dfr_pct'])}%, EUR STR {_p(m['estr_pct'])}% "
        f"({_s(m['estr_minus_dfr_bp'])} vs deposit rate), AAA 3M {_p(m['aaa_3m_pct'])}% "
        f"({_s(m['basis_aaa3m_minus_estr_bp'])} vs EUR STR).",
        "",
        "| 3M forward starting | Level | vs deposit rate | vs curve's 3M rate | 1W change |",
        "|---|---|---|---|---|",
        *fwd,
        "",
        "Term-premium sensitivity only, never the headline: 3M forward in 2Y minus "
        f"{m['scenario_tp_bp_per_year_values'][1]}bp per year of horizon "
        f"{_p(m['scenario_tp_10bp_per_year_fwd3m_in_2y_pct'])}%, minus "
        f"{m['scenario_tp_bp_per_year_values'][2]}bp per year "
        f"{_p(m['scenario_tp_20bp_per_year_fwd3m_in_2y_pct'])}%.",
        "",
        "**Curve**",
        "",
        head,
        *curve,
        "",
        "**Cross-asset**",
        "",
        head,
        *cross,
        "",
        f"EUR/USD {_n(m['eurusd'])} on {m['eurusd_date']} "
        f"({_s(m['eurusd_chg_1w_pct'], '%')} 1W, {_s(m['eurusd_chg_1m_pct'], '%')} 1M). "
        f"Correlation of daily changes with the EUR-US 2Y differential: "
        f"{_n(m['eurusd_corr_2y_diff_3m'])} over 3M, {_n(m['eurusd_corr_2y_diff_1y'])} over "
        f"1Y. On {m['eurusd_fit_date']}: {_s(m['eurusd_minus_fit_pct'], '%')} vs its 1Y fit "
        f"on that differential (R-squared {_n(m['eurusd_fit_r2'])}).",
        "",
        f"**Hedging inputs**: EUR STR {_p(m['estr_pct'])}%, deposit rate "
        f"{_p(m['ecb_dfr_pct'])}%, 3M forward in 1Y {_p(m['fwd3m_in_1y_pct'])}%, AAA 2Y "
        f"{_p(m['aaa_2y_pct'])}% ({_s(m['aaa_2y_minus_estr_bp'])} vs EUR STR). Swap rates are "
        "not in the dataset.",
        "",
    ]


def render_scaffold(m: dict[str, Any], chart: str) -> str:
    """The note skeleton: computed figures, dates and chart; placeholders for all prose."""
    moves = [f"- {PLACEHOLDER} {mv['measure']}: level, 1W change and context. "
             "See Figures.]" for mv in m["top_moves_1w"]]
    return "\n".join([
        UNREVIEWED,
        "> **Scaffold, not a note.** Figures, dates and the chart are computed by "
        "`erm build`. Replace every WRITE placeholder by hand, delete these two "
        "lines, then run `erm check-note` on this file.",
        "",
        _title(m),
        "",
        "## What moved this week",
        *moves,
        "",
        "## What the curve prices for the ECB",
        f"{PLACEHOLDER} pricing paragraph. What the AAA curve's 3M forwards imply against "
        "the deposit rate at each horizon, the path measured from the curve's own 3M rate, "
        "and the term-premium caveat. No view on what the ECB will do.]",
        "",
        f"![3M forward path from the AAA curve]({chart})",
        "",
        "## Cross-asset",
        f"- **FX:** {PLACEHOLDER} EUR/USD against the EUR-US 2Y differential.]",
        f"- **Sovereign credit:** {PLACEHOLDER} all-issuer minus AAA spread at 5Y and 10Y.]",
        "",
        "## For a client hedging floating-rate exposure",
        f"{PLACEHOLDER} hedger paragraph. What the pricing means mechanically for a client "
        "paying floating on euro short rates; the AAA 2Y yield against EUR STR as a rough "
        "proxy for fixing, noting swap rates are not in the dataset. No recommendation.]",
        "",
        *_figures(m),
        *_footer(m),
    ])


def render(sections: dict[str, Any], m: dict[str, Any], chart: str) -> str:
    """An LLM-drafted note, marked unreviewed until someone edits it by hand."""
    return "\n".join([
        UNREVIEWED,
        "> **LLM draft.** Drafted by Claude from the computed figures (`erm note --llm`). "
        "Review and edit by hand, delete these two lines, then run `erm check-note`.",
        "",
        _title(m),
        "",
        "## What moved this week",
        *[f"- {b}" for b in sections["moved"]],
        "",
        "## What the curve prices for the ECB",
        sections["ecb_pricing"],
        "",
        f"![3M forward path from the AAA curve]({chart})",
        "",
        "## Cross-asset",
        f"- **FX:** {sections['fx_bullet']}",
        f"- **Sovereign credit:** {sections['credit_bullet']}",
        "",
        "## For a client hedging floating-rate exposure",
        sections["hedging"],
        "",
        *_footer(m),
    ])


def write_note(use_llm: bool = False, out_dir: Path | None = None,
               force: bool = False) -> str:
    """Write the scaffold (default) or an LLM draft for the latest metrics.

    out_dir: where to write <as_of>.md and img/ (default notes/). Refuses to overwrite
    an existing file unless force=True, so a hand-written note is never replaced.
    """
    m = json.loads((paths.PROCESSED / "metrics_latest.json").read_text())
    out_dir = out_dir or paths.NOTES
    out = out_dir / f"{m['as_of']}.md"
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; use --force to overwrite it")
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{m['as_of']}_{CHART}"
    shutil.copyfile(paths.FIGURES / CHART, img_dir / img_name)
    chart = f"img/{img_name}"
    text = render(draft_llm(m), m, chart) if use_llm else render_scaffold(m, chart)
    out.write_text(text)
    log.info("wrote %s (%s, %s)", out, "LLM draft" if use_llm else "scaffold",
             date.today().isoformat())
    return text
