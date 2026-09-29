"""Command line: erm fetch | build | report | note | readme | sources.

Typical weekly run:  erm fetch && erm build && erm report && erm note && erm readme
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import paths


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .fetch import fetch_all

    fetch_all(force=args.force)


def _cmd_build(_: argparse.Namespace) -> None:
    from . import build

    build.write(build.run())


def _cmd_report(_: argparse.Namespace) -> None:
    """Charts plus a plain-text summary of the key figures on stdout."""
    from . import build, charts

    res = build.run()
    for p in charts.make_all(res):
        logging.info("chart %s", p.relative_to(paths.ROOT))
    m = res.metrics
    print(f"As of {m['as_of']}")
    print(f"  AAA 2Y/10Y/30Y: {m['aaa_2y_pct']}% / {m['aaa_10y_pct']}% / {m['aaa_30y_pct']}%"
          f"  (10Y 1w {m['aaa_10y_chg_1w_bp']:+}bp)")
    print(f"  2s10s {m['slope_2s10s_bp']}bp  10s30s {m['slope_10s30s_bp']}bp  "
          f"2s5s10s {m['fly_2s5s10s_bp']}bp")
    print(f"  DFR {m['ecb_dfr_pct']}%  EUR STR {m['estr_pct']}%  3M fwd in 1Y "
          f"{m['fwd3m_in_1y_pct']}% ({m['fwd3m_in_1y_minus_dfr_bp']:+}bp vs DFR)")
    print(f"  PCA top-3 explained {m['pca_top3_explained_pct']}%  anomalies: "
          f"{len(m['shape_anomalies'])}")
    print(f"  EUR-US 10Y {m['eur_minus_us_10y_par_bp']}bp  EUR/USD {m['eurusd']}  "
          f"all-AAA 10Y {m['all_minus_aaa_10y_bp']}bp")


def _cmd_note(args: argparse.Namespace) -> None:
    from .note import write_note

    print(write_note(use_llm=not args.no_llm))


def _cmd_readme(_: argparse.Namespace) -> None:
    from .headline import update_readme

    update_readme()
    logging.info("updated README.md headline")


def _cmd_sources(_: argparse.Namespace) -> None:
    from .sources import sources_markdown

    (paths.ROOT / "SOURCES.md").write_text(sources_markdown())
    logging.info("wrote SOURCES.md")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `erm` command."""
    parser = argparse.ArgumentParser(prog="erm", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="download all series into data/raw/ (cached per day)")
    p.add_argument("--force", action="store_true", help="re-download even if cached today")
    p.set_defaults(func=_cmd_fetch)
    sub.add_parser("build", help="compute analytics into data/processed/").set_defaults(
        func=_cmd_build)
    sub.add_parser("report", help="render charts into figures/ and print a summary"
                   ).set_defaults(func=_cmd_report)
    p = sub.add_parser("note", help="write the weekly note into notes/")
    p.add_argument("--no-llm", action="store_true",
                   help="fixed-template wording, no API call (used in CI)")
    p.set_defaults(func=_cmd_note)
    sub.add_parser("readme", help="rewrite the README headline from the latest metrics"
                   ).set_defaults(func=_cmd_readme)
    sub.add_parser("sources", help="regenerate SOURCES.md from the catalogue").set_defaults(
        func=_cmd_sources)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
