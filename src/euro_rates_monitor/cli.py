"""Command line: erm fetch | fetch-summary | build | report | backtest | readme | note |
check-note | sources.

Typical weekly run:  erm fetch && erm build && erm report && erm readme && erm note,
then write the note by hand and run erm check-note on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import paths


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .fetch import fetch_all, missing_required

    missing = missing_required(fetch_all(force=args.force))
    if missing:
        raise SystemExit(f"required series unavailable (no download, no cache): "
                         f"{', '.join(missing)}")


def _cmd_fetch_summary(_: argparse.Namespace) -> None:
    """Print the last fetch report as markdown (the Actions step appends it to the
    run summary page)."""
    import json

    from .fetch import REPORT, summary_markdown

    report = paths.RAW / REPORT
    if not report.exists():
        print("## Data fetch\n\nNo fetch report: `erm fetch` did not run or crashed early.")
        return
    print(summary_markdown(json.loads(report.read_text())))


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
    print(f"  EUR-US 10Y {m.get('eur_minus_us_10y_par_bp', 'n/a')}bp  "
          f"EUR/USD {m.get('eurusd', 'n/a')}  all-AAA 10Y {m.get('all_minus_aaa_10y_bp', 'n/a')}bp")
    if m["sections_unavailable"]:
        print(f"  unavailable: {', '.join(m['sections_unavailable'])}")


def _cmd_backtest(_: argparse.Namespace) -> None:
    from . import backtest, charts

    errors, summary, rd = backtest.run()
    backtest.write(errors, summary, rd)
    charts._style()
    for p in (charts.backtest_errors(errors, summary, rd, paths.FIGURES),
              charts.backtest_mean_error(summary, paths.FIGURES)):
        logging.info("chart %s", p.relative_to(paths.ROOT))


def _cmd_note(args: argparse.Namespace) -> None:
    from .note import write_note

    print(write_note(use_llm=args.llm, out_dir=args.out, force=args.force))


def _cmd_check_note(args: argparse.Namespace) -> None:
    from .note import check_note, note_files

    files = note_files(args.paths)
    if not files:
        print("no notes to check")
        return
    failed = 0
    for f in files:
        problems = check_note(f)
        print(f"{'FAIL' if problems else 'ok  '} {f}")
        for p in problems:
            print(f"     {p}")
        failed += bool(problems)
    if failed:
        raise SystemExit(f"{failed} of {len(files)} note(s) failed the check")


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
    sub.add_parser("fetch-summary", help="print the last fetch report as a markdown table"
                   ).set_defaults(func=_cmd_fetch_summary)
    sub.add_parser("build", help="compute analytics into data/processed/").set_defaults(
        func=_cmd_build)
    sub.add_parser("report", help="render charts into figures/ and print a summary"
                   ).set_defaults(func=_cmd_report)
    sub.add_parser("backtest", help="backtest forwards vs realised 3M rate; tables + charts"
                   ).set_defaults(func=_cmd_backtest)
    p = sub.add_parser("note", help="write a scaffold note to fill in by hand")
    p.add_argument("--llm", action="store_true",
                   help="optional: Claude drafts the prose (needs ANTHROPIC_API_KEY)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default notes/)")
    p.add_argument("--force", action="store_true", help="overwrite an existing note")
    p.set_defaults(func=_cmd_note)
    p = sub.add_parser("check-note",
                       help="verify every number and date in notes against their figures")
    p.add_argument("paths", nargs="+", type=Path, help="note files or directories")
    p.set_defaults(func=_cmd_check_note)
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
