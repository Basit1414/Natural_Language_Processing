#!/usr/bin/env python3
"""Generate and execute the compact review notebook from pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "BIS_speech_analysis.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
# BIS Central-Bank Speech Archive: Exploration and Policy Tone

This executed notebook is the review companion to `run_analysis.py` and `REPORT.md`. It loads the assignment-required `./BIS_speeches.csv` through the reusable package for the basic audit, then reads the deterministic outputs of the full pipeline. Run `python run_analysis.py` before re-executing this notebook.
"""
        ),
        code(
            """
from pathlib import Path
import json
import sys

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
if not (ROOT / "outputs").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from bis_speeches.data import basic_summary, load_and_clean, quality_summary

TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
summary = json.loads((ROOT / "outputs" / "summary.json").read_text())
pd.set_option("display.max_columns", 20)
"""
        ),
        markdown("## A. Basic data exploration"),
        code(
            """
speeches = load_and_clean(ROOT / "BIS_speeches.csv")
headline, length_summary = basic_summary(speeches)
display(headline)
display(length_summary)
display(quality_summary(speeches))
"""
        ),
        code("display(Image(filename=str(FIGURES / 'speeches_per_year.png')))"),
        markdown(
            """
The economically motivated additional feature is whether a speech mentions inflation: it connects the communication corpus to a central price-stability objective. BIS warns that the archive is neither guaranteed complete nor free of machine-extraction artifacts, and 2026 is a partial year.
"""
        ),
        markdown("## B. Country attribution via metadata"),
        code(
            """
display(pd.read_csv(TABLES / "country_attribution_qa.csv"))
display(pd.read_csv(TABLES / "top10_countries.csv"))
display(Image(filename=str(FIGURES / "selected_countries_per_year.png")))
"""
        ),
        markdown(
            """
Country means the jurisdiction of the speaker's institution, not nationality or event venue. The method searches a bounded byline against a curated institution-alias registry, records multi-match conflicts, and uses a speaker fallback only when at least three direct observations agree at least 95%. National, territorial, regional, supranational, international, and unresolved records remain distinct.
"""
        ),
        markdown("## C. Inflation-persistence keyword signal"),
        code(
            """
persistence = pd.read_csv(TABLES / "inflation_persistence_annual.csv")
display(pd.read_csv(TABLES / "inflation_persistence_patterns.csv"))
display(persistence.tail(10))
display(Image(filename=str(FIGURES / "inflation_persistence_share.png")))
"""
        ),
        markdown(
            """
The indicator rises sharply in 2022–23 and subsequently recedes; the elevated 2026 share is partial-year evidence. Keyword matching is transparent but can miss paraphrases and mishandle negation, quotation, or hypothetical scenarios.
"""
        ),
        markdown("## Additional A. Speech-level tone model"),
        code(
            """
display(pd.read_csv(TABLES / "tone_validation_grid.csv"))
display(pd.read_csv(TABLES / "tone_test_metrics.csv"))
display(pd.read_csv(TABLES / "tone_class_metrics.csv"))
display(pd.read_csv(TABLES / "tone_examples.csv"))
display(Image(filename=str(FIGURES / "tone_confusion_matrix.png")))
"""
        ),
        markdown(
            """
The supervised classifier uses WCB hawkish/dovish/neutral/irrelevant sentence labels, word and character TF-IDF, a class-balanced averaged logistic-loss SGD model, and sigmoid calibration. Contrastive clauses are scored separately. Speech hawkishness is `H/(H+D)`; mixedness is `1-|H-D|/(H+D)`; signal strength is `(H+D)/clauses`. This distinguishes no-signal speeches from speeches that contain both hawkish and dovish scenarios.
"""
        ),
        markdown("## Additional B. Speaker-level trends"),
        code(
            """
display(pd.read_csv(TABLES / "top10_speakers.csv"))
trends = pd.read_csv(TABLES / "speaker_trend_tests.csv")
display(trends)
display(Image(filename=str(FIGURES / "speaker_tone_halfyear.png")))
"""
        ),
        markdown(
            """
Half-year windows balance noise reduction and temporal resolution. Linear trend tests use two-lag HAC standard errors, and Benjamini–Hochberg q-values control false discoveries across the ten speakers. These are descriptive within-speaker trends, not causal estimates.
"""
        ),
        markdown("## Additional C. Country-level tone and stability"),
        code(
            """
display(Markdown("### Relatively most hawkish"))
display(pd.read_csv(TABLES / "top10_hawkish_countries.csv"))
display(Markdown("### Relatively most dovish"))
display(pd.read_csv(TABLES / "top10_dovish_countries.csv"))
display(Markdown("### Most stable annual mean tone"))
display(pd.read_csv(TABLES / "top10_stable_countries.csv"))
display(Image(filename=str(FIGURES / "country_tone_extremes.png")))
display(Image(filename=str(FIGURES / "country_tone_stability.png")))
"""
        ),
        markdown(
            """
Rankings require at least 50 tone-valid speeches and 10 observed years. Stability is the standard deviation of annual means, using only country-years with at least three speeches and requiring 10 such years. The lists are relative rankings; model transfer error, corpus selection, and uneven publication practices limit cross-country structural interpretation. See `REPORT.md` for full assumptions, exact results, citations, and interpretation.
"""
        ),
    ]

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    nbf.write(executed, TARGET)
    print(f"Wrote and executed {TARGET}")


if __name__ == "__main__":
    main()
