"""Write a small, committed extract of the raw cache to data/sample/.

The full raw cache (tens of MB) is not committed; `erm fetch` re-creates it. The sample
keeps the last 60 observations of each file, unmodified, so a reader can see the raw
formats and the tests can check our curve maths against published ECB numbers.

Usage: python scripts/make_sample.py
"""

from __future__ import annotations

import io

import pandas as pd

from euro_rates_monitor import paths
from euro_rates_monitor.load import latest_file
from euro_rates_monitor.sources import CATALOGUE

N = 60


def main() -> None:
    paths.SAMPLE.mkdir(parents=True, exist_ok=True)
    for s in CATALOGUE:
        src = latest_file(s.name)
        text = src.read_text()
        if s.source == "ECB":
            df = pd.read_csv(io.StringIO(text), dtype=str)
            df = df.groupby("KEY", sort=False).tail(N)
            out = df.to_csv(index=False)
        elif s.source == "FRED":
            lines = text.splitlines()
            out = "\n".join([lines[0], *lines[-N:]]) + "\n"
        else:  # Fed Board GSW: keep the preamble (it states the conventions) + last rows
            head, _, body = text.partition("\nDate,")
            rows = body.splitlines()
            out = head + "\nDate," + "\n".join([rows[0], *rows[-N:]]) + "\n"
        (paths.SAMPLE / src.name).write_text(out)
        print(f"data/sample/{src.name}")


if __name__ == "__main__":
    main()
