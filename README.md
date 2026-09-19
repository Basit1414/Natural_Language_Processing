# Central-Bank Speech NLP

A reproducible analysis of 20,728 speeches in the Bank for International Settlements (BIS) central-bank speech archive. The project explores archive coverage, infers each speaker's institutional jurisdiction, detects inflation-persistence language, estimates continuous hawkishness/dovishness scores, and tests tone trends for frequent speakers and countries.

The complete findings and methodological discussion are in [REPORT.md](REPORT.md). The executed review notebook is [notebooks/BIS_speech_analysis.ipynb](notebooks/BIS_speech_analysis.ipynb).

![Annual BIS speech volume](outputs/figures/speeches_per_year.png)

## Highlights

- 20,728 speeches spanning 10 September 1996 to 22 June 2026, with 1,019 named speakers.
- Deterministic institution-based jurisdiction attribution resolves 99.8% of records while preserving regional, supranational, international, multi-match, and unresolved cases.
- 1,404 speeches (6.8%) contain a high-precision inflation-persistence expression; the full-year share peaks at 23.5% in 2023.
- A calibrated word-and-character TF-IDF classifier trained on the public World Central Banks (WCB) labels achieves 0.630 held-out accuracy and 0.616 macro-F1.
- Speech scoring retains both directional tone and mixedness, allowing hawkish and dovish conditional scenarios to coexist in one speech.
- Speaker trend inference uses half-year means, HAC standard errors, and Benjamini–Hochberg false-discovery control.

These estimates are descriptive. Corpus selection, model transfer error, incomplete metadata, and uneven publication practices preclude causal or structural country comparisons.

## Quick start

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/download_wcb_data.py
python run_analysis.py
python scripts/build_report.py
python scripts/build_notebook.py
pytest
python scripts/validate_outputs.py
```

`run_analysis.py` uses the assignment-required relative input path `./BIS_speeches.csv` by default. Paths can be overridden:

```bash
python run_analysis.py --data ./BIS_speeches.csv --wcb-data ./data/external/wcb --output ./outputs
```

The download scripts verify SHA-256 hashes for the analyzed snapshot. Because the BIS download URL is mutable, a future archive update intentionally fails the pinned check until the new data are reviewed.

## Methods

### Basic exploration

The loader validates the six expected fields (`url`, `title`, `description`, `date`, `text`, `author`), normalizes whitespace, audits missingness and duplicates, and computes word counts. Two impossible 2027 year values are repaired with explicit URL-level overrides after verification against the live BIS pages; raw dates are retained.

### Jurisdiction attribution

Country means the jurisdiction of the speaker's institution—not nationality, venue, host, or country discussed. A curated alias registry is matched only within the introductory byline, with explicit speaker roles prioritized. A speaker fallback is allowed only when at least three directly attributed speeches have 95% agreement. Every result includes provenance and ambiguous cases remain reviewable.

### Inflation-persistence signal

Six auditable expression families cover persistent/sticky inflation, inflation remaining or expected to remain elevated, continuing price pressures, longer-than-expected inflation, second-round effects, and de-anchored expectations.

### Continuous policy tone

The model uses the official WCB train/validation/test splits and predicts `hawkish`, `dovish`, `neutral`, and `irrelevant` sentence probabilities. Policy sentences are split at contrast markers before aggregation:

- hawkishness: `H / (H + D)`, from 0 (dovish) to 1 (hawkish);
- mixedness: `1 - |H - D| / (H + D)`; and
- signal strength: `(H + D) / number of policy clauses`.

Country rankings require at least 50 tone-valid speeches and 10 observed years. Stability uses the standard deviation of annual means, requiring at least three speeches in each of 10 years.

## Repository layout

```text
.
├── config/institution_aliases.csv       # Auditable institution registry
├── notebooks/BIS_speech_analysis.ipynb  # Executed review notebook
├── outputs/
│   ├── figures/                         # Seven analysis figures
│   ├── tables/                          # Inspectable CSV results
│   └── summary.json                     # Machine-readable headline results
├── scripts/
│   ├── build_notebook.py
│   ├── build_report.py
│   ├── download_data.py
│   ├── download_wcb_data.py
│   └── validate_outputs.py
├── src/bis_speeches/                    # Cleaning, attribution, signals, model, stats
├── tests/                               # Regression and behavior tests
├── REPORT.md                            # Full assignment response
└── run_analysis.py                      # End-to-end entry point
```

Raw BIS data, third-party WCB parquet files, and the fitted model are not committed. They are recreated by the scripts above. Small tables and figures are committed so results can be inspected without rerunning the model.

## Data and model sources

- [BIS central-bank speech archive](https://www.bis.org/speeches/central-bank/download)
- [World Central Banks paper](https://arxiv.org/abs/2505.17048)
- [WCB annotated dataset](https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000)
- [WorldCentralBanks code repository](https://github.com/gtfintechlab/WorldCentralBanks)

The repository's source code is MIT-licensed. The BIS archive and WCB data are governed by their publishers' terms and are not redistributed here. The WCB dataset card lists a noncommercial share-alike license; confirm applicable terms before reuse.
