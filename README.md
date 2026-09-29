# euro-rates-monitor

Tracks the euro area government curve, extracts what it prices for ECB policy, and writes a one-page weekly market note with one command.

## Headline (curve as of 28 Sep 2026)

- **ECB pricing.** The AAA curve's 3-month forwards imply a short rate of **3.42% in 1 year** and **3.46% in 2 years**. The deposit rate today is **2.50%**, so that is **92bp** and **96bp** above it.
  - Measured from the curve's own 3M rate (2.56%, which is 12bp above €STR), the 1-year forward is 86bp higher.
  - Forwards include a term premium (see [Limitations](#limitations)).
- **Curve level.** The 10Y AAA yield is **3.63%**, at the top of its one-year range.
- **Sovereign spread.** The all-issuer vs AAA spread at 10Y is **50bp**, also at its one-year high.

Latest note: [`notes/`](notes/). The headline above is a snapshot. The note is regenerated each week.

![3M forward path from the AAA curve](figures/forward_path.png)

## Run it

Python 3.12 or later (numpy 2.5 and pandas 3.0 require it). Tested on 3.12 in CI and 3.13 locally.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .

erm fetch            # download every series into data/raw/<source>/<name>_<date>.csv
erm build            # analytics -> data/processed/ (metrics_latest.json feeds the note)
erm report           # charts -> figures/, plus a summary on stdout
erm note             # weekly note -> notes/<date>.md (Claude drafts the prose)
erm note --no-llm    # same note, fixed-template wording, no API call
pytest               # curve maths, PCA, change logic, number validator
```

**Keys (both optional):**

| Variable | What it does | Without it |
|---|---|---|
| `FRED_API_KEY` | `fetch` uses the official FRED API ([free key](https://fred.stlouisfed.org/docs/api/api_key.html)). | `fetch` falls back to FRED's public CSV download (`fredgraph.csv`). Same data, same cache format. |
| `ANTHROPIC_API_KEY` | `erm note` uses Claude to draft the prose. | Only `erm note --no-llm` works. |
| `ERM_MODEL` | Changes the model. | Default: `claude-opus-5-5`. |

The API call enables Anthropic's server-side fallback: if a safety classifier declines the request, it is re-run on a fallback model instead of failing.

## How the note stays honest

1. **Numbers come from one file.** `erm build` writes every figure the note may use to `data/processed/metrics_latest.json`, already rounded to the precision the note shows.
2. **What Claude receives.** Only that dictionary, with instructions to:
   - copy numbers exactly;
   - do no arithmetic;
   - add no outside knowledge;
   - give no view on future policy.
3. **Code decides what matters.** Code, not the model, picks the three moves for "what moved": the largest 1-week changes relative to a typical week over the past year.
4. **Number check.** Every number in the returned text is extracted and matched against the dictionary. A re-rounded, summed or invented number fails the check. The run retries once, then exits with the offending numbers listed. Tenor labels (10Y, 2s10s, 3-month) and dates that appear in the dictionary are exempt.
5. **Retention.** The last 12 notes are kept in `notes/`.

**Why CI uses `--no-llm`.** The GitHub Action ([`.github/workflows/weekly-note.yml`](.github/workflows/weekly-note.yml)) runs every Saturday with no repository secrets: ECB, the Fed Board and the FRED CSV endpoint are keyless. The LLM step would need `ANTHROPIC_API_KEY` stored as a secret, so the Action runs `--no-llm`, and the note it commits opens with a "Template note" label.

## Method

Everything is computed from the **ECB AAA curve** unless stated otherwise. The all-issuer curve is used for comparison.

**1. Curve state.**
- *How the curve is built:* the ECB fits a Nelson-Siegel-Svensson curve to euro area central government bonds between 3 months and 30 years, and publishes both the six parameters and the continuously compounded spot rates. We evaluate the parameters directly. A test checks this reproduces the published spot rates to within 1e-6.
- *Summary measures:*
  - level = 10Y
  - slopes = 2s10s and 10s30s
  - curvature = the 2s5s10s butterfly (2×5Y − 2Y − 10Y; positive means the 5Y is high relative to the wings)
- *Context:* each measure is shown against its own one-year range.

**2. Implied forwards and the ECB path.**
- *Formula:* with continuous compounding, f(t1,t2) = (r2·t2 − r1·t1)/(t2 − t1). This is the rate that makes rolling from t1 to t2 equivalent to investing to t2 directly.
- *The path:* 3-month forwards starting every quarter out to 2 years form the path the curve prices for the 3M rate.
- *Two comparisons:*
  - against the deposit rate (the ECB's steering rate);
  - against the curve's own 3M rate, which removes a constant gap between government bonds and €STR.
- *Sensitivity:* the chart also shows forwards minus 10bp and 20bp per year of horizon. These lines show how much a term premium could change the reading and are never the headline (see [Limitations](#limitations)).

**3. PCA.**
- *Setup:* principal components of daily changes (bp) at 1–30Y over the last three years. On the current window the first three explain **98.2%** of daily variance (85.0 / 10.6 / 2.7).
- *Check that they are level, slope and curvature:* the number of sign changes in each loading vector is 0, 1 and 2 respectively. The tests check this on a synthetic curve with known structure.
- *Curve-shape anomaly:* the residual after three factors, as a z-score, shows where today's curve shape departs from its usual structure (see Limitations for why it is not rich/cheap).

**4. Cross-market.**
- *Maturity spreads:* EUR AAA **par** yields minus US Treasury constant-maturity yields, which are also par yields, so like is compared with like.
- *Implied paths:* ECB zero curve against the Fed Board's Gürkaynak–Sack–Wright zero curve.
- *EUR/USD:* against the 2Y and 10Y differentials, with rolling 3M and 1Y correlations of daily changes, and a 1-year regression of the level on the 2Y differential. The regression is descriptive, not a fair value.
- *Sovereign credit:* the all-issuer minus AAA spread at 5Y and 10Y is tracked as a euro area sovereign credit / fragmentation indicator.

**5. Real rates.** Two euro area proxies, both labelled:
- *Realised:* 10Y AAA minus the latest HICP y/y.
- *Survey:* 10Y AAA minus the ECB Survey of Professional Forecasters' longer-term expectation.
- *US comparison:* the TIPS real yield, which is market-based.

Current values are in `data/processed/metrics_latest.json`.

**6. Weekly and monthly changes.** Latest value minus the last value on or before the same calendar date one week or one month earlier. This is robust to TARGET2 and US holidays. If the base value is more than 5 days stale, the change is reported as missing rather than silently covering a longer window.

**ECB policy turning points** on the 2s10s chart come from the deposit rate series itself: the first hike after a cutting phase, and vice versa. One rule applies: a change that reverses the previous change within 7 days is a corridor adjustment, not a turn (9 Oct 2008).

| | |
|---|---|
| ![Curve](figures/curve_snapshot.png) | ![2s10s](figures/slope_2s10s.png) |
| ![PCA](figures/pca_loadings.png) | ![EUR-US 10Y](figures/eur_us_10y.png) |

## Limitations

- **Term premium.** Forwards = expected short rate + term premium. The premium cannot be observed.
  - It is not linear in the horizon.
  - It varies over time.
  - Model estimates have been negative in some periods (e.g. during large-scale asset purchases).

  The k = 0/10/20bp-per-year lines are a sensitivity illustration, not an estimate. Read the path as "what is priced", not "what the market expects".
- **Government curve, not OIS.** Desks read ECB pricing from €STR swaps and ECB-dated €STR forwards, which have no stable free source. Government bills and bonds can trade away from €STR because of collateral scarcity and supply; the gap reached several tens of bp in 2022. The note prints today's gap between the 3M AAA rate and €STR, and the path relative to the curve's own 3M rate. That removes a *constant* gap, not a changing one.
- **AAA vs all-issuer curves.**
  - The AAA curve uses only bonds rated AAA by Fitch. Its country composition changes when ratings change, which moves the curve (and the AAA vs all-issuer spread) without any market move.
  - The all-issuer curve includes lower-rated sovereigns, so it carries credit and liquidity premia and is not a risk-free curve.
  - Neither is a single country's curve.
- **Fitted, not traded.** The ECB curve smooths across bonds, so it cannot show which individual bond is rich or cheap. The PCA residual is labelled "curve-shape anomaly" for that reason.
- **Data revisions and timing.**
  - HICP is first published as a flash estimate and can be revised.
  - ECB and FRED series can be revised or re-based. Each download is stored with its retrieval date and SHA-256 hash in `data/raw/manifest.jsonl`, so a note can be traced to the exact data.
  - Euro and US data are fixed at different times of day.
  - The Fed Board zero curve is published with a lag of about 1–2 weeks. The note states each as-of date.
- **Realised inflation is not an expectation.** The realised-inflation real rate is backward-looking and moves with energy prices and base effects. The SPF measure looks forward but is a quarterly survey of forecasters, not a market price. No free euro area inflation-swap or breakeven series exists; the US side does have one (TIPS), so the two sides are not measured the same way.

## Data and licences

- **Sources:** every series, with identifier, source, URL and licence, is listed in [SOURCES.md](SOURCES.md), which is generated from [`sources.py`](src/euro_rates_monitor/sources.py).
- **Licences:** ECB statistics may be reused with the source quoted; US series are public domain.
- **What is committed:** the raw cache (about 36MB) is not committed. `data/sample/` holds the last 60 observations of each raw file, unmodified. `data/processed/` holds the derived tables.
- **Left out:** swap rates, Bund futures and euro area inflation swaps, because no stable free source with clear terms exists. They are not approximated.

**Possible extension: individual Bund yields.** The Bundesbank publishes daily *observed* yields of the current 2/5/7/10/15/20/30Y Federal securities through a free API (dataflow `BBSSY`, e.g. `BBSSY/D.REN.EUR.A630.000000WT1010.A` for the 10Y, back to 2001, with the current ISIN in the metadata). That would allow a genuine bond-versus-curve comparison. Caveats:
- the series follow the *current* benchmark, so residual maturity jumps at each new issue;
- Germany only, against a multi-country AAA curve;
- the prices are fixed around noon Frankfurt time.

## Layout

```
src/euro_rates_monitor/
  sources.py      series catalogue (ids, URLs, licences) -> SOURCES.md
  fetch.py        downloads + dated cache + manifest
  load.py         parse cached files
  curve.py        Svensson evaluation, level/slope/curvature, ranges
  forwards.py     forward rates, ECB path, term-premium scenarios
  pca.py          PCA on daily changes, curve-shape anomaly
  crossmarket.py  EUR vs US, EUR/USD vs differentials, AAA vs all-issuer
  realrates.py    realised and survey-based real rates
  changes.py      holiday-robust 1w / 1m changes
  build.py        runs everything, writes data/processed/ and metrics_latest.json
  charts.py       the five charts
  note.py         LLM / template drafting, number validator, notes/ retention
  cli.py          erm fetch | build | report | note | sources
tests/            pytest suite
notebooks/        walkthrough.ipynb: each calculation step by step
scripts/          make_sample.py
```
