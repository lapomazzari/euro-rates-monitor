"""Rewrite the README headline from the latest metrics, so it never goes stale.

The headline sits between two HTML comment markers in README.md. It is fixed-template
text (no LLM), and it passes through the same number validator as the weekly note.
If `erm backtest` has been run, its finding is the first bullet: whether the forwards
have beaten a random walk is the context for reading everything after it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from . import paths
from .note import _n, _p, _rel, validate

START, END = "<!-- headline:start -->", "<!-- headline:end -->"


def _range_phrase(pct: int) -> str:
    """Where a value sits in its one-year range, in words."""
    if pct >= 100:
        return "at the top of its one-year range"
    if pct <= 0:
        return "at the bottom of its one-year range"
    if pct >= 90:
        return "near the top of its one-year range"
    if pct <= 10:
        return "near the bottom of its one-year range"
    return f"{pct}% of the way from its one-year low to its high"


def _hz(hs: list[int]) -> str:
    return ", ".join(f"{h}M" for h in hs)


def backtest_bullet(b: dict[str, Any]) -> str:
    """First headline bullet: the backtest verdict, stated plainly whatever it is."""
    tested = b["bt_horizons_tested"]
    better = b["bt_horizons_forwards_significantly_better"]
    worse = b["bt_horizons_forwards_significantly_worse"]
    ratios = ", ".join(f"{b[f'bt_{h}m_rmse_ratio']:.2f}" for h in tested)
    pvals = ", ".join(f"{b[f'bt_{h}m_dm_p']:.2f}" for h in tested)
    d = datetime.strptime(b["bt_first_origin"], "%Y-%m-%d")
    if not better and not worse:
        verdict = (f"the AAA curve's 3M forwards have **not beaten a naive random walk** by "
                   f"a statistically significant margin at any horizon tested ({_hz(tested)})")
    else:
        parts = []
        if better:
            parts.append(f"beaten a naive random walk at {_hz(better)}")
        if worse:
            parts.append(f"done worse than a naive random walk at {_hz(worse)}")
        verdict = f"the AAA curve's 3M forwards have {' and '.join(parts)}"
    text = (f"- **Forwards vs a random walk.** Since {d:%b %Y}, {verdict}. Forward RMSE was "
            f"{ratios} times the random walk's (Diebold-Mariano p = {pvals}).")
    if b["bt_horizons_withheld_full_sample"]:
        text += (f" At {_hz(b['bt_horizons_withheld_full_sample'])} the sample is too "
                 "short for inference.")
    return text + " See [What the forwards have been worth](#what-the-forwards-have-been-worth)."


def headline_markdown(m: dict[str, Any], b: dict[str, Any] | None = None) -> str:
    """The README headline block (without markers) for a metrics dictionary.

    b: the backtest headline dictionary, if available; its verdict comes first.
    """
    as_of = datetime.strptime(m["as_of"], "%Y-%m-%d")
    spread = m["all_minus_aaa_10y_bp"]
    if spread >= m["all_minus_aaa_10y_1y_high_bp"]:
        spread_pos = "at its one-year high"
    elif spread <= m["all_minus_aaa_10y_1y_low_bp"]:
        spread_pos = "at its one-year low"
    else:
        spread_pos = (f"within its one-year range of {_n(m['all_minus_aaa_10y_1y_low_bp'])}"
                      f"-{_n(m['all_minus_aaa_10y_1y_high_bp'])}bp")
    bullets = [backtest_bullet(b)] if b else []
    bullets += [
        (f"- **ECB pricing.** The AAA curve's 3-month forwards imply a short rate of "
         f"**{_p(m['fwd3m_in_1y_pct'])}% in 1 year** and **{_p(m['fwd3m_in_2y_pct'])}% in "
         f"2 years**, against a deposit rate of **{_p(m['ecb_dfr_pct'])}%** "
         f"({_n(m['fwd3m_in_1y_minus_dfr_bp'])}bp and "
         f"{_rel(m['fwd3m_in_2y_minus_dfr_bp'], 'bp', 'it')}).\n"
         f"  - Measured from the curve's own 3M rate ({_p(m['aaa_3m_pct'])}%, "
         f"{_rel(m['basis_aaa3m_minus_estr_bp'], 'bp', '€STR')}), the 1-year forward is "
         f"{_rel(m['fwd3m_in_1y_minus_aaa3m_bp'], 'bp', 'today')}.\n"
         "  - Forwards include a term premium (see [Limitations](#limitations))."),
        (f"- **Curve level.** The 10Y AAA yield is **{_p(m['aaa_10y_pct'])}%**, "
         f"{_range_phrase(m['aaa_10y_pct_of_1y_range'])}."),
        (f"- **Sovereign spread.** The all-issuer vs AAA spread at 10Y is "
         f"**{_n(spread)}bp**, {spread_pos}."),
    ]
    return "\n".join([f"## Headline (curve as of {as_of.day} {as_of:%b %Y})", "", *bullets])


def update_readme() -> None:
    """Replace the block between the markers in README.md with the current headline."""
    m = json.loads((paths.PROCESSED / "metrics_latest.json").read_text())
    bt_file = paths.PROCESSED / "backtest_headline.json"
    b = json.loads(bt_file.read_text()) if bt_file.exists() else None
    block = headline_markdown(m, b)
    validate({"headline": block}, {**m, **(b or {})})
    readme = paths.ROOT / "README.md"
    text = readme.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise RuntimeError("README.md has no headline markers")
    readme.write_text(pattern.sub(lambda _: f"{START}\n{block}\n{END}", text))
