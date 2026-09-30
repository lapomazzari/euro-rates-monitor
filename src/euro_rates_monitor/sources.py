"""Catalogue of every series the project uses: identifier, source, URL and licence.

This file is the single source of truth. `erm sources` renders it to SOURCES.md, so
the documentation cannot drift from what the code actually downloads.
"""

from __future__ import annotations

from dataclasses import dataclass

ECB_API = "https://data-api.ecb.europa.eu/service/data"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

LICENCE_ECB = (
    "ECB statistics: reuse free of charge if the source is quoted and the statistics "
    "are not modified (ECB statistics reuse policy)."
)
LICENCE_EUROSTAT = (
    "Eurostat data distributed via the ECB Data Portal; Eurostat permits reuse with "
    "acknowledgement of the source."
)
LICENCE_US_PUBLIC = "US government work (public domain); retrieved via FRED, St. Louis Fed."
LICENCE_FED_BOARD = "Federal Reserve Board staff dataset, public domain."

# Maturities (years) published by the ECB that we download. The ECB fits a
# Nelson-Siegel-Svensson curve to bonds with 3 months to 30 years residual maturity.
SPOT_TENORS: dict[str, float] = {
    "SR_3M": 0.25, "SR_6M": 0.5, "SR_1Y": 1, "SR_2Y": 2, "SR_3Y": 3, "SR_5Y": 5,
    "SR_7Y": 7, "SR_10Y": 10, "SR_15Y": 15, "SR_20Y": 20, "SR_30Y": 30,
}
PAR_TENORS: dict[str, float] = {"PY_2Y": 2, "PY_5Y": 5, "PY_10Y": 10, "PY_30Y": 30}
SVENSSON_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


@dataclass(frozen=True)
class Series:
    """One downloadable dataset.

    name:      local identifier, also the cache file stem.
    source:    "ECB", "FRED" or "FEDBOARD".
    key:       ECB "FLOW/KEY" (may select several series with '+'), FRED id, or URL.
    what:      one-line description.
    unit:      unit of the values as published.
    licence:   reuse terms.
    section:   the analysis it feeds (see build.SECTION_INPUTS).
    required:  if True, the euro analysis and the headline cannot be built without it,
               so `erm fetch` fails when it has neither a fresh download nor a cache.
    """

    name: str
    source: str
    key: str
    what: str
    unit: str
    licence: str
    section: str = "core"
    required: bool = False

    @property
    def url(self) -> str:
        """Human-readable URL for documentation (not necessarily the fetch URL)."""
        if self.source == "ECB":
            return f"{ECB_API}/{self.key}"
        if self.source == "FRED":
            return f"https://fred.stlouisfed.org/series/{self.key}"
        return self.key


def _yc(curve: str, items: list[str]) -> str:
    # G_N_A = AAA-rated central government bonds, G_N_C = all euro area central
    # government bonds. SV_C_YM = Svensson model, continuous compounding, yield error min.
    return f"YC/B.U2.EUR.4F.{curve}.SV_C_YM.{'+'.join(items)}"


CATALOGUE: list[Series] = [
    Series("ecb_spot_aaa", "ECB", _yc("G_N_A", list(SPOT_TENORS)),
           "Euro area AAA government zero-coupon spot rates, 3M-30Y", "% p.a., cont. comp.",
           LICENCE_ECB, "core", required=True),
    Series("ecb_spot_all", "ECB", _yc("G_N_C", list(SPOT_TENORS)),
           "Euro area all-issuer government zero-coupon spot rates, 3M-30Y",
           "% p.a., cont. comp.", LICENCE_ECB, "credit"),
    Series("ecb_params_aaa", "ECB", _yc("G_N_A", SVENSSON_PARAMS),
           "Svensson parameters of the AAA curve", "betas in %, taus in years", LICENCE_ECB,
           "core", required=True),
    Series("ecb_par_aaa", "ECB", _yc("G_N_A", list(PAR_TENORS)),
           "Euro area AAA government par yields, 2/5/10/30Y", "% p.a.", LICENCE_ECB,
           "cross_market"),
    Series("ecb_estr", "ECB", "EST/B.EU000A2X2A25.WT",
           "Euro short-term rate (EUR STR), volume-weighted trimmed mean", "% p.a.",
           LICENCE_ECB, "core", required=True),
    Series("ecb_dfr", "ECB", "FM/D.U2.EUR.4F.KR.DFR.LEV",
           "ECB deposit facility rate", "% p.a.", LICENCE_ECB, "core", required=True),
    # The older ICP/M.U2.N.000000.4.ANR series stops at 2025-12 after Eurostat's
    # move to the ECOICOP 2 classification; the HICP dataflow carries the current data.
    Series("ecb_hicp", "ECB", "HICP/M.U2.N.000000.4D0.ANR",
           "Euro area HICP, overall index, annual rate of change", "% y/y",
           LICENCE_EUROSTAT, "real_rates_ea"),
    Series("ecb_spf_lt", "ECB", "SPF/Q.U2.HICP.POINT.LT.Q.AVG",
           "ECB Survey of Professional Forecasters, longer-term HICP expectation (mean)",
           "% y/y", LICENCE_ECB, "real_rates_ea"),
    Series("ecb_spf_1y", "ECB", "SPF/M.U2.HICP.POINT.P12M.Q.AVG",
           "ECB SPF, HICP expectation one year ahead (mean; date = target month)",
           "% y/y", LICENCE_ECB, "real_rates_ea"),
    Series("ecb_eurusd", "ECB", "EXR/D.USD.EUR.SP00.A",
           "ECB euro reference exchange rate, US dollars per euro", "USD per EUR",
           LICENCE_ECB, "fx"),
    *[
        Series(f"fred_{sid.lower()}", "FRED", sid, what, unit, LICENCE_US_PUBLIC, section)
        for sid, what, unit, section in [
            ("DGS2", "US Treasury 2-year constant maturity yield", "% p.a., bond-equiv.",
             "cross_market"),
            ("DGS5", "US Treasury 5-year constant maturity yield", "% p.a., bond-equiv.",
             "cross_market"),
            ("DGS10", "US Treasury 10-year constant maturity yield", "% p.a., bond-equiv.",
             "cross_market"),
            ("DGS30", "US Treasury 30-year constant maturity yield", "% p.a., bond-equiv.",
             "cross_market"),
            ("DFEDTARU", "Fed funds target range, upper limit", "% p.a.", "cross_market"),
            ("DFEDTARL", "Fed funds target range, lower limit", "% p.a.", "cross_market"),
            ("CPIAUCSL", "US CPI, all urban consumers, SA index", "index 1982-84=100",
             "real_rates_us"),
            ("T10YIE", "US 10-year breakeven inflation (nominal minus TIPS)", "% p.a.",
             "real_rates_us"),
            ("DFII10", "US 10-year TIPS constant maturity real yield", "% p.a.",
             "real_rates_us"),
        ]
    ],
    Series("fed_gsw", "FEDBOARD", GSW_URL,
           "US Treasury zero-coupon curve, Gurkaynak-Sack-Wright (2006) Svensson fit",
           "% p.a., cont. comp.", LICENCE_FED_BOARD, "cross_market"),
]

BY_NAME: dict[str, Series] = {s.name: s for s in CATALOGUE}

SECTION_LABELS = {
    "credit": "sovereign credit (all-issuer curve)",
    "cross_market": "cross-market (US data)",
    "fx": "FX (EUR/USD vs rate differential)",
    "real_rates_ea": "euro area real rates",
    "real_rates_us": "US real rates",
}


def sources_markdown() -> str:
    """Render the catalogue as a markdown table for SOURCES.md."""
    lines = [
        "# Data sources",
        "",
        "Generated from `src/euro_rates_monitor/sources.py` by `erm sources`. Every download",
        "is cached under `data/raw/<source>/<name>_<retrieval-date>.csv` and parsed from there.",
        "",
        "| Local name | Source | Identifier | Description | Unit | Feeds | Licence |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in CATALOGUE:
        ident = s.key if s.source != "FEDBOARD" else "feds200628.csv"
        lines.append(
            f"| `{s.name}` | {s.source} | [`{ident}`]({s.url}) | {s.what} | {s.unit} "
            f"| {s.section}{' (required)' if s.required else ''} | {s.licence} |"
        )
    lines += [
        "",
        "Source line used on charts and notes: *Source: ECB statistics; Eurostat; FRED "
        "(St. Louis Fed); Federal Reserve Board; own calculations.*",
        "",
        "## Considered and not used",
        "",
        "- **EUR swap (OIS/EURIBOR) rates, Bund futures**: no stable free source with clear "
        "redistribution terms. The ECB pricing analysis therefore uses the AAA government "
        "curve and says so.",
        "- **Euro area inflation swaps / breakevens**: no free source. Real rates use realised "
        "HICP and the ECB SPF survey instead, both labelled as such.",
        "- **Euro area recession dates**: the CEPR-EABCN chronology could not be retrieved "
        "programmatically (the site blocks automated access) and FRED's `EUROREC` is an OECD "
        "growth-cycle indicator discontinued in 2022, not a recession dating. Charts mark ECB "
        "policy turning points derived from the deposit rate series instead.",
        "- **Bundesbank benchmark Bund yields** (`BBSSY` dataflow): observed yields of the "
        "current 2/5/7/10/15/20/30Y Federal securities, free via the Bundesbank API. A "
        "possible extension for bond-versus-curve analysis; not used yet (see README).",
    ]
    return "\n".join(lines) + "\n"
