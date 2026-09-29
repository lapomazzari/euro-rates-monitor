"""Rewrite the README headline from the latest metrics, so it never goes stale.

The headline sits between two HTML comment markers in README.md. It is fixed-template
text (no LLM), and it passes through the same number validator as the weekly note.
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


def headline_markdown(m: dict[str, Any]) -> str:
    """The README headline block (without markers) for a metrics dictionary."""
    as_of = datetime.strptime(m["as_of"], "%Y-%m-%d")
    spread = m["all_minus_aaa_10y_bp"]
    if spread >= m["all_minus_aaa_10y_1y_high_bp"]:
        spread_pos = "at its one-year high"
    elif spread <= m["all_minus_aaa_10y_1y_low_bp"]:
        spread_pos = "at its one-year low"
    else:
        spread_pos = (f"within its one-year range of {_n(m['all_minus_aaa_10y_1y_low_bp'])}"
                      f"-{_n(m['all_minus_aaa_10y_1y_high_bp'])}bp")
    bullets = [
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
    block = headline_markdown(m)
    validate({"headline": block}, m)
    readme = paths.ROOT / "README.md"
    text = readme.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise RuntimeError("README.md has no headline markers")
    readme.write_text(pattern.sub(lambda _: f"{START}\n{block}\n{END}", text))
