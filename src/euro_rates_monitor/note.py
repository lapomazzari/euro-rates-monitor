"""Generate the one-page weekly note from data/processed/metrics_latest.json.

Two drafting modes:
* LLM (default): Claude drafts the prose from the metrics dictionary only. The prompt
  forbids outside knowledge, policy opinions and invented context; every number in
  the returned text is then checked against the dictionary and the run fails loudly
  if any number is not found there.
* Template (--no-llm): deterministic sentences from the same dictionary. Used in CI,
  where no API key is available. The note says so in its first line.

Both paths go through the same number validator.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from . import paths

log = logging.getLogger(__name__)

MODEL = os.environ.get("ERM_MODEL", "claude-opus-5-5")
KEEP_NOTES = 12
CHART = "forward_path.png"


class NumberValidationError(RuntimeError):
    """The drafted text contains numbers that are not in the metrics dictionary."""


# --------------------------------------------------------------------------- validation

_LABEL_PATTERNS = [
    r"\b\d+s(?:\d+s)*\d+s\b",                  # curve labels: 2s10s, 10s30s, 2s5s10s
    r"\b\d+(?:\.\d+)?[YMW]\b",                 # tenors / horizons: 10Y, 3M, 1Y, 6M
    r"\b\d+-(?:month|year|week|day)\b",        # adjectival tenors: 3-month, 10-year
    r"\bPC\d\b",                               # PC1..PC3
]
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?")


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _date_renderings(value: str) -> list[str]:
    """Ways a date from the dictionary may legitimately appear in prose."""
    out = [value]
    m = re.fullmatch(r"(\d{4})-Q([1-4])", value)
    if m:
        return out + [f"Q{m[2]} {m[1]}", f"{m[1]} Q{m[2]}"]
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            out += [f"{d.day} {d:%B %Y}", f"{d.day} {d:%b %Y}", f"{d.day} {d:%B}",
                    f"{d.day} {d:%b}", f"{d:%B} {d.day}, {d.year}"]
        out += [f"{d:%B %Y}", f"{d:%b %Y}"]
    return out


def allowed_numbers(metrics: dict[str, Any]) -> set[float]:
    """Absolute values of every number in the dictionary (signs may become words)."""
    return {abs(float(v)) for v in _walk(metrics)
            if isinstance(v, int | float) and not isinstance(v, bool)}


def strip_labels_and_dates(text: str, metrics: dict[str, Any]) -> str:
    """Remove tenor labels and dictionary dates so only claimed quantities remain."""
    dates = [v for v in _walk(metrics) if isinstance(v, str)
             and re.fullmatch(r"\d{4}-(\d{2}(-\d{2})?|Q[1-4])", v)]
    renderings = sorted({r for d in dates for r in _date_renderings(d)}, key=len, reverse=True)
    for r in renderings:
        text = text.replace(r, " ")
    for pat in _LABEL_PATTERNS:
        text = re.sub(pat, " ", text)
    return text


def unknown_numbers(text: str, metrics: dict[str, Any]) -> list[str]:
    """Numbers in `text` that do not appear (up to sign) in `metrics`.

    Matching is exact on value: '2.50' matches 2.5, but '3.6' does not match 3.63, so
    re-rounding a figure fails. Arithmetic on figures (sums, differences) also fails,
    by design: the note may only quote computed numbers.
    """
    allowed = allowed_numbers(metrics)
    bad = []
    for tok in _NUMBER.findall(strip_labels_and_dates(text, metrics)):
        val = abs(float(tok.replace("−", "-").replace(",", "")))
        if not any(abs(val - a) < 1e-9 for a in allowed):
            bad.append(tok)
    return bad


def validate(sections: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Raise NumberValidationError listing every number not found in the dictionary."""
    bad = {k: unknown_numbers(v if isinstance(v, str) else " ".join(v), metrics)
           for k, v in sections.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise NumberValidationError(
            "numbers not found in the metrics dictionary: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()))


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
            raise RuntimeError("ANTHROPIC_API_KEY is not set; use `erm note --no-llm`")
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


# ---------------------------------------------------------------------- template draft

def _n(v: float | int) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _p(v: float | int) -> str:
    """Rate levels always with two decimals (2.50%, not 2.5%); same value, so it validates."""
    return f"{v:.2f}"


def _mv(v: float | int | None, unit: str = "bp") -> str:
    if v is None:
        return "n/a"
    if v == 0:
        return f"unchanged (0{unit})"
    return f"{'rose' if v > 0 else 'fell'} {_n(abs(v))}{unit}"


def _rel(v: float | int, unit: str, ref: str) -> str:
    """'12bp above EUR STR' / '1.6% below its fit'."""
    return f"{_n(abs(v))}{unit} {'above' if v >= 0 else 'below'} {ref}"


def draft_template(m: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback: fixed sentences filled from the dictionary."""
    moved = []
    for mv in m["top_moves_1w"]:
        unit = mv["latest_unit"]
        level = _p(mv["latest"]) if unit == "%" else _n(mv["latest"])
        moved.append(f"{mv['measure']}: {level}{unit}, {_mv(mv['chg_1w_bp'])} "
                     f"on the week ({_n(mv['size_vs_typical_week'])} times a typical weekly "
                     "move over the past year).")
    ecb = (f"The AAA curve's 3M forward starting in 6M is {_p(m['fwd3m_in_6m_pct'])}%, in 1Y "
           f"{_p(m['fwd3m_in_1y_pct'])}% and in 2Y {_p(m['fwd3m_in_2y_pct'])}%, against a "
           f"deposit rate of {_p(m['ecb_dfr_pct'])}%: the 1Y forward is "
           f"{_rel(m['fwd3m_in_1y_minus_dfr_bp'], 'bp', 'the deposit rate')} and the 2Y "
           f"forward {_rel(m['fwd3m_in_2y_minus_dfr_bp'], 'bp', 'it')}. Measured from the "
           f"curve's own 3M rate ({_p(m['aaa_3m_pct'])}%, "
           f"{_rel(m['basis_aaa3m_minus_estr_bp'], 'bp', 'EUR STR')}), the 1Y forward is "
           f"{_rel(m['fwd3m_in_1y_minus_aaa3m_bp'], 'bp', 'today')}. Forwards include a term "
           "premium, so they are not a clean expectation of policy.")
    fx = (f"EUR/USD is {_n(m['eurusd'])}, {_mv(m['eurusd_chg_1w_pct'], '%')} on the week; "
          f"its daily correlation with the EUR-US 2Y differential is "
          f"{_n(m['eurusd_corr_2y_diff_3m'])} over 3M and {_n(m['eurusd_corr_2y_diff_1y'])} "
          f"over 1Y, and on {m['eurusd_fit_date']} it stood "
          f"{_rel(m['eurusd_minus_fit_pct'], '%', 'its 1-year fit on that differential')}.")
    credit = (f"All-issuer minus AAA: {_n(m['all_minus_aaa_5y_bp'])}bp at 5Y "
              f"({_mv(m['all_minus_aaa_5y_chg_1w_bp'])} on the week) and "
              f"{_n(m['all_minus_aaa_10y_bp'])}bp at 10Y ({_mv(m['all_minus_aaa_10y_chg_1w_bp'])}"
              f"), 1-year range {_n(m['all_minus_aaa_10y_1y_low_bp'])}-"
              f"{_n(m['all_minus_aaa_10y_1y_high_bp'])}bp at 10Y.")
    hedging = (f"A client paying floating on euro short rates pays close to EUR STR "
               f"({_p(m['estr_pct'])}%) today; the curve's forwards imply a 3M rate of "
               f"{_p(m['fwd3m_in_1y_pct'])}% in 1Y. The AAA 2Y yield of {_p(m['aaa_2y_pct'])}% "
               f"sits {_n(m['aaa_2y_minus_estr_bp'])}bp above EUR STR, a rough proxy for the cost "
               "of fixing for two years. Swap rates are not in this dataset, so an actual "
               "hedge would price off the swap curve and differ from this proxy.")
    return {"moved": moved, "ecb_pricing": ecb, "fx_bullet": fx, "credit_bullet": credit,
            "hedging": hedging}


# ----------------------------------------------------------------------------- output

def render(sections: dict[str, Any], m: dict[str, Any], template: bool, chart: str) -> str:
    """Assemble the markdown note."""
    as_of = datetime.strptime(m["as_of"], "%Y-%m-%d")
    lines = []
    if template:
        lines += ["> **Template note**: generated without the LLM drafting step "
                  "(`erm note --no-llm`). Wording is fixed; all figures are computed.", ""]
    lines += [
        f"# Euro rates weekly: {as_of:%d %B %Y}",
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
        "---",
        f"*Data: ECB curve and EUR STR as of {m['as_of']}; US Treasury yields to "
        f"{m['eur_us_spread_date']}; US zero curve {m['us_zero_curve_date']}; HICP "
        f"{m['hicp_month']}. Forwards are from the AAA government curve, include a term "
        "premium and are not OIS-implied. Source: ECB statistics; Eurostat; FRED "
        "(St. Louis Fed); Federal Reserve Board; own calculations. Not investment advice.*",
        "",
    ]
    return "\n".join(lines)


def prune(keep: int = KEEP_NOTES) -> None:
    """Keep the most recent `keep` notes (and their chart images)."""
    notes = sorted(paths.NOTES.glob("????-??-??.md"))
    for old in notes[:-keep] if len(notes) > keep else []:
        old.unlink()
        img = paths.NOTES / "img" / f"{old.stem}_{CHART}"
        img.unlink(missing_ok=True)


def write_note(use_llm: bool = True) -> str:
    """Build the note for the latest metrics and write notes/<as_of>.md."""
    m = json.loads((paths.PROCESSED / "metrics_latest.json").read_text())
    sections = draft_llm(m) if use_llm else draft_template(m)
    validate(sections, m)
    img_dir = paths.NOTES / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{m['as_of']}_{CHART}"
    shutil.copyfile(paths.FIGURES / CHART, img_dir / img_name)
    text = render(sections, m, template=not use_llm, chart=f"img/{img_name}")
    out = paths.NOTES / f"{m['as_of']}.md"
    out.write_text(text)
    prune()
    log.info("wrote %s (%s, generated %s)", out.relative_to(paths.ROOT),
             "LLM draft" if use_llm else "template", date.today().isoformat())
    return text
