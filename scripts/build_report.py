#!/usr/bin/env python3
"""Build the assignment report from validated pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    display = frame.copy()
    for column, pattern in formats.items():
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else pattern.format(value)
            )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def main() -> None:
    summary = json.loads((ROOT / "outputs" / "summary.json").read_text())
    basic = pd.read_csv(TABLES / "basic_summary.csv")
    lengths = pd.read_csv(TABLES / "speech_length_summary.csv")
    top_countries = pd.read_csv(TABLES / "top10_countries.csv")
    annual_persistence = pd.read_csv(TABLES / "inflation_persistence_annual.csv")
    model_metrics = pd.read_csv(TABLES / "tone_test_metrics.csv").iloc[0]
    class_metrics = pd.read_csv(TABLES / "tone_class_metrics.csv")
    examples = pd.read_csv(TABLES / "tone_examples.csv")
    speakers = pd.read_csv(TABLES / "top10_speakers.csv")
    trends = pd.read_csv(TABLES / "speaker_trend_tests.csv")
    hawkish = pd.read_csv(TABLES / "top10_hawkish_countries.csv")
    dovish = pd.read_csv(TABLES / "top10_dovish_countries.csv")
    stable = pd.read_csv(TABLES / "top10_stable_countries.csv")

    basic_display = pd.DataFrame(
        {
            "Measure": [
                "Speeches",
                "Coverage",
                "Unique named speakers",
                "Inflation mentioned",
            ],
            "Result": [
                f"{summary['data']['speeches']:,}",
                f"{summary['data']['start_date']} to {summary['data']['end_date']}",
                f"{summary['data']['unique_speakers']:,}",
                f"{summary['data']['inflation_mention_count']:,} ({percent(summary['data']['inflation_mention_share'])})",
            ],
        }
    )
    length_display = lengths.rename(columns={"statistic": "Statistic", "words": "Words"})
    length_display["Statistic"] = length_display["Statistic"].str.replace("_", " ").str.title()
    length_display["Words"] = length_display["Words"].map(lambda value: f"{value:,.1f}")

    country_display = top_countries.rename(
        columns={
            "rank": "Rank",
            "jurisdiction": "Country",
            "speeches": "Speeches",
            "speakers": "Speakers",
        }
    )
    country_display["Speeches"] = country_display["Speeches"].map(lambda value: f"{value:,}")

    recent_persistence = annual_persistence.loc[annual_persistence["year"].ge(2019)].copy()
    recent_persistence = recent_persistence.rename(
        columns={
            "year": "Year",
            "speeches": "All speeches",
            "persistence_speeches": "Flagged",
            "persistence_share": "Share",
        }
    )
    recent_persistence["Share"] = recent_persistence["Share"].map(percent)

    class_display = class_metrics.rename(
        columns={
            "class": "Class",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "support": "Support",
        }
    )
    for column in ["Precision", "Recall", "F1"]:
        class_display[column] = class_display[column].map(lambda value: f"{value:.3f}")
    class_display["Support"] = class_display["Support"].astype(int)

    speaker_display = speakers[
        ["rank", "author", "speeches", "tone_valid_speeches", "mean_hawkishness", "mean_mixedness"]
    ].rename(
        columns={
            "rank": "Rank",
            "author": "Speaker",
            "speeches": "All speeches",
            "tone_valid_speeches": "Tone-valid",
            "mean_hawkishness": "Mean score",
            "mean_mixedness": "Mixedness",
        }
    )
    for column in ["Mean score", "Mixedness"]:
        speaker_display[column] = speaker_display[column].map(lambda value: f"{value:.3f}")

    trend_display = trends[
        [
            "speaker",
            "halfyear_periods",
            "slope_per_year",
            "hac_t_statistic",
            "p_value",
            "fdr_q_value",
            "significant_fdr_5pct",
        ]
    ].rename(
        columns={
            "speaker": "Speaker",
            "halfyear_periods": "Half-years",
            "slope_per_year": "Slope/year",
            "hac_t_statistic": "HAC t",
            "p_value": "p",
            "fdr_q_value": "BH q",
            "significant_fdr_5pct": "q < .05",
        }
    )
    trend_display["Slope/year"] = trend_display["Slope/year"].map(lambda value: f"{value:+.4f}")
    trend_display["HAC t"] = trend_display["HAC t"].map(lambda value: f"{value:.2f}")
    trend_display["p"] = trends["p_value"].map(p_value)
    trend_display["BH q"] = trends["fdr_q_value"].map(p_value)
    trend_display["q < .05"] = trend_display["q < .05"].map({True: "Yes", False: "No"})

    def tone_rank_table(frame: pd.DataFrame, stability: bool = False) -> str:
        if stability:
            selected = frame[
                ["rank", "jurisdiction", "central_bank", "speeches", "years_with_3plus_speeches", "mean_hawkishness", "annual_tone_sd"]
            ].rename(
                columns={
                    "rank": "Rank",
                    "jurisdiction": "Country",
                    "central_bank": "Primary archive institution",
                    "speeches": "Speeches",
                    "years_with_3plus_speeches": "Qualifying years",
                    "mean_hawkishness": "Mean score",
                    "annual_tone_sd": "Annual SD",
                }
            )
            return table(selected, {"Mean score": "{:.3f}", "Annual SD": "{:.3f}"})
        selected = frame[
            ["rank", "jurisdiction", "central_bank", "speeches", "mean_hawkishness", "bootstrap_ci_lower", "bootstrap_ci_upper"]
        ].copy()
        selected["95% bootstrap CI"] = selected.apply(
            lambda row: f"[{row.bootstrap_ci_lower:.3f}, {row.bootstrap_ci_upper:.3f}]", axis=1
        )
        selected = selected.drop(columns=["bootstrap_ci_lower", "bootstrap_ci_upper"]).rename(
            columns={
                "rank": "Rank",
                "jurisdiction": "Country",
                "central_bank": "Primary archive institution",
                "speeches": "Speeches",
                "mean_hawkishness": "Mean score",
            }
        )
        return table(selected, {"Mean score": "{:.3f}"})

    hawkish_example = examples.loc[examples["example_type"].eq("hawkish")].iloc[0]
    dovish_example = examples.loc[examples["example_type"].eq("dovish")].iloc[0]
    significant = trends.loc[trends["significant_fdr_5pct"]]
    significant_text = "; ".join(
        f"{row.speaker} ({row.slope_per_year:+.4f}/year, q={p_value(row.fdr_q_value)})"
        for row in significant.itertuples(index=False)
    )

    content = f"""# Central-Bank Speech Communication: BIS Archive Analysis

## Executive summary

This analysis covers **{summary['data']['speeches']:,} speeches from {summary['data']['start_date']} through {summary['data']['end_date']}**. Inflation-persistence language appears in **{summary['inflation_persistence']['count']:,} speeches ({percent(summary['inflation_persistence']['share'])})** and rose sharply in 2022–23, peaking at **{percent(summary['inflation_persistence']['peak_share'])} in {summary['inflation_persistence']['peak_year']}**. A calibrated, supervised sentence classifier reaches **{model_metrics.test_accuracy:.1%} test accuracy** and **{model_metrics.test_macro_f1:.3f} macro-F1** on the external World Central Banks (WCB) held-out test split; its country and speaker results should therefore be interpreted as structured descriptive estimates, not ground truth.

## 1. Basic data exploration

The script loads the assignment-required relative path `./BIS_speeches.csv`, collapses whitespace in `date`, `author`, `title`, `description`, and `text`, parses dates, and retains all substantive records. Two archive records carried impossible 2027 years; their years were minimally corrected to 2025 and 2026 after checking the BIS URL identifiers and live pages, and the original values remain in `date_raw`.

{table(basic_display)}

Speech length is the number of whitespace-delimited tokens after whitespace normalization.

{table(length_display)}

The additional economically motivated statistic is the share of speeches that mention inflation: **{summary['data']['inflation_mention_count']:,} ({percent(summary['data']['inflation_mention_share'])})**. Inflation is central to most central-bank price-stability mandates, so this gives a simple measure of how much of the archive directly engages with a core monetary-policy objective.

![Speeches per year](outputs/figures/speeches_per_year.png)

One important limitation is selection and measurement error: BIS describes this as an archive of central-bank speeches, not a census, and warns that completeness and machine-extracted text quality are not guaranteed. The archive also over-represents institutions that publish frequently in English or are more consistently collected; 2026 is only a partial year through 22 June.

## 2. Country attribution via metadata

Country is defined as the jurisdiction represented by the speaker's institution at the speech date—not the speaker's nationality, event location, host institution, or a country merely discussed. The resolver:

1. normalizes and searches only the leading byline/affiliation window of `description` (falling back to the opening text when needed);
2. applies a curated, longest-specificity institution alias registry, while scoring explicit roles such as “Governor of” above honorees or hosts;
3. assigns a conservative author-level fallback only when at least three directly mapped speeches agree at least 95%; and
4. retains the method, matched alias, conflict candidates, regional/supranational category, and unresolved queue for audit.

This resolves **{summary['country_attribution']['institution_alias']:,} directly**, **{summary['country_attribution']['speaker_fallback']:,} by conservative speaker fallback**, and leaves **{summary['country_attribution']['unresolved']:,} unresolved**, for **{percent(summary['country_attribution']['resolved_share'])} total coverage**. It also flags **{summary['country_attribution']['direct_conflicts_flagged']:,}** bylines containing aliases from multiple jurisdictions for QA. The rankings below include only national jurisdictions; the ECB, BIS, and regional currency unions are preserved separately rather than forced into a headquarters country.

{table(country_display)}

![Selected-country speech volume](outputs/figures/selected_countries_per_year.png)

The main attribution challenge is that descriptions can mention several institutions—for example, a national governor who also holds a BIS role, or an ECB speaker hosted by another central bank. Restricting matching to the byline, using role-aware ordering, preserving multi-match flags, and never inferring from venue materially reduces that error, but the mapping is still estimated metadata rather than a BIS-supplied country field.

## 3. Inflation-persistence keyword signal

The high-precision indicator uses six transparent pattern families: persistent/sticky inflation; inflation that remains high/elevated/above target; inflation expected to remain high; continuing price or cost pressures; inflation lasting longer than expected; and second-round effects or de-anchored expectations. A speech is flagged when at least one expression appears; the exact regular expressions are in `outputs/tables/inflation_persistence_patterns.csv`.

**Result:** {summary['inflation_persistence']['count']:,} speeches ({percent(summary['inflation_persistence']['share'])}) are flagged.

{table(recent_persistence)}

![Inflation-persistence share](outputs/figures/inflation_persistence_share.png)

The series accelerates from **6.2% in 2021** to **20.4% in 2022** and **23.5% in 2023**, then retreats to **12.5% in 2024** and **10.3% in 2025**. The 2026 value is elevated but is partial and should not be compared mechanically with full years. Keyword rules are auditable, but they can miss paraphrases and may misread negation, quotation, historical discussion, or a risk scenario as the speaker's own current concern.

## 4. Speech-level hawkishness and dovishness

### Learning paradigm and features

The tone model is supervised multiclass sentence classification. It trains on the public [World Central Banks annotated dataset](https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000), whose stance labels are `hawkish`, `dovish`, `neutral`, and `irrelevant`, using the official 17,500/3,750/3,750 train/validation/test split. The model combines 1–2 word TF-IDF features and 3–5 character TF-IDF features in a class-balanced, averaged logistic-loss SGD classifier. Hyperparameter `alpha` is chosen on validation macro-F1 and probabilities are sigmoid-calibrated on the validation split; the test set is touched once for final reporting. The design is a lightweight, reproducible alternative to the larger WCB RoBERTa model described in the [WCB paper](https://arxiv.org/abs/2505.17048) and [official repository](https://github.com/gtfintechlab/WorldCentralBanks).

Each BIS speech is split into sentences and then at contrast markers such as “but,” “however,” and semicolons. Only policy-relevant clauses are scored. Let `H` and `D` be the summed calibrated hawkish and dovish probabilities across a speech:

- hawkishness = `H / (H + D)`, on 0 (dovish) to 1 (hawkish);
- mixedness = `1 - |H - D| / (H + D)`, where 1 means balanced directional evidence; and
- signal strength = `(H + D) / number of clauses`.

This permits one speech to contain tightening and easing scenarios. Speeches with no policy clause or signal strength below 0.15 receive an `insufficient_signal` flag and are excluded from aggregates; **{summary['tone_model']['tone_valid_speeches']:,} speeches ({percent(summary['tone_model']['tone_valid_share'])})** remain tone-valid.

### Validation

- Best validation `alpha`: **{summary['tone_model']['best_alpha']}**
- Held-out accuracy: **{model_metrics.test_accuracy:.3f}**
- Held-out macro-F1: **{model_metrics.test_macro_f1:.3f}**
- Held-out weighted-F1: **{model_metrics.test_weighted_f1:.3f}**
- Held-out log loss: **{model_metrics.test_log_loss:.3f}**
- Multiclass Brier score: **{model_metrics.test_multiclass_brier:.3f}**
- 15-bin top-label expected calibration error: **{model_metrics.test_top_label_ece_15_bins:.3f}**

{table(class_display)}

![Held-out confusion matrix](outputs/figures/tone_confusion_matrix.png)

### Concrete classifications

- **Hawkish:** “{hawkish_example.sentence}” The model assigns `p(hawkish)={hawkish_example.p_hawkish:.3f}`. Persistently elevated and rising inflation expectations are framed as a risk, which supports tighter policy.
- **Dovish:** “{dovish_example.sentence}” The model assigns `p(dovish)={dovish_example.p_dovish:.3f}`. Below-target projected inflation, falling energy prices, low core inflation, and subdued demand support accommodation.

## 5. Speaker-level tone over time

The top ten speakers are selected by total archive frequency. Speech scores are averaged in half-year windows: this reduces speech-level noise while preserving more timing information than annual averages. Each speech receives equal weight, so unusually long speeches do not dominate.

{table(speaker_display)}

![Speaker half-year tone](outputs/figures/speaker_tone_halfyear.png)

For each speaker, an OLS trend is fitted to half-year means. Reported standard errors are heteroskedasticity- and autocorrelation-consistent (HAC, two half-year lags); Benjamini–Hochberg correction controls the false-discovery rate across ten tests.

{table(trend_display)}

Four trends remain statistically distinguishable from zero at 5% FDR: **{significant_text}**. These estimates describe within-speaker archival trends, not causal changes in preferences; changing economic regimes, topic mix, job roles, and sample coverage can all move the measured score.

## 6. Country-level tone and stability

Country means give equal weight to each tone-valid speech. To limit thin-sample rankings, a country must have at least **50 tone-valid speeches across at least 10 observed years**. The stability ranking uses the standard deviation of annual mean tone, requires at least three speeches in each annual cell and at least ten qualifying years, and is therefore a comparison of temporal dispersion rather than the precision of the overall mean. Mean-score uncertainty is shown with a seeded, speech-level nonparametric bootstrap.

The “hawkish” and “dovish” lists are **relative rankings within the eligible sample**. A country can be among the ten most hawkish even when its mean is below the neutral midpoint of 0.5.

### Relatively most hawkish

{tone_rank_table(hawkish)}

### Relatively most dovish

{tone_rank_table(dovish)}

![Country tone extremes](outputs/figures/country_tone_extremes.png)

### Most stable annual average tone

{tone_rank_table(stable, stability=True)}

![Country tone stability](outputs/figures/country_tone_stability.png)

## 7. Assumptions, limitations, and interpretation

- **Trend tests:** OLS assumes a linear mean trend and comparable scores over time. HAC inference relaxes independent, homoskedastic residuals but does not remove omitted-variable bias; BH correction addresses multiple testing, not model misspecification.
- **Country rankings:** speeches are not random samples of all central-bank communication. Bootstrap intervals quantify within-archive sampling variability, not corpus selection bias, model error, or country-attribution uncertainty.
- **Model transfer:** WCB labels span multiple central banks but remain sentence annotations that may differ from BIS speech prose, historical vocabulary, and translated English. Held-out macro-F1 of {model_metrics.test_macro_f1:.3f} is useful but leaves substantial classification error.
- **Mixed content:** a score near 0.5 can mean genuinely neutral language or offsetting hawkish and dovish statements. `tone_mixedness` and `tone_signal_strength` must be read with the main score.
- **Licensing:** the external WCB dataset is published under a noncommercial share-alike license on its dataset card; this repository downloads but does not redistribute it. Confirm licensing before commercial reuse.

## 8. Reproducibility

From the repository root:

```bash
python -m venv .venv
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

The raw BIS archive and WCB parquet files are hash-pinned but excluded from Git. Small result tables, figures, the executed notebook, and this report are retained for inspection.

## References

- [BIS central-bank speech archive and download notes](https://www.bis.org/speeches/central-bank/download)
- [World Central Banks paper](https://arxiv.org/abs/2505.17048)
- [WCB dataset card](https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000)
- [WCB model card](https://huggingface.co/gtfintechlab/model_WCB_stance_label)
- [WorldCentralBanks official code repository](https://github.com/gtfintechlab/WorldCentralBanks)
"""
    (ROOT / "REPORT.md").write_text(content, encoding="utf-8")
    print(f"Wrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
