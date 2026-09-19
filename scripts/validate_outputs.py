#!/usr/bin/env python3
"""Fail fast when committed analysis artifacts are incomplete or inconsistent."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text())
    speeches = pd.read_csv(TABLES / "speech_level_results.csv", low_memory=False)
    qa = pd.read_csv(TABLES / "country_attribution_qa.csv").set_index("method")
    metrics = pd.read_csv(TABLES / "tone_test_metrics.csv").iloc[0]
    hawkish = pd.read_csv(TABLES / "top10_hawkish_countries.csv")
    dovish = pd.read_csv(TABLES / "top10_dovish_countries.csv")
    stable = pd.read_csv(TABLES / "top10_stable_countries.csv")

    require(len(speeches) == summary["data"]["speeches"] == 20_728, "speech row count is 20,728")
    require(speeches["speech_id"].nunique() == len(speeches), "speech IDs are unique")
    require(speeches["url"].nunique() == len(speeches), "source URLs are unique")
    require(
        speeches["hawkishness_score"].between(0, 1).all(),
        "all hawkishness scores are bounded by 0 and 1",
    )
    require(
        speeches["tone_mixedness"].between(0, 1).all(),
        "all mixedness scores are bounded by 0 and 1",
    )
    require(
        int(speeches["tone_valid"].sum()) == summary["tone_model"]["tone_valid_speeches"],
        "tone-valid total agrees with summary.json",
    )
    require(int(qa.loc["total", "speeches"]) == len(speeches), "country QA totals reconcile")
    require(
        int(qa.loc["unresolved", "speeches"])
        == summary["country_attribution"]["unresolved"],
        "unresolved attribution total agrees with summary.json",
    )
    require(abs(metrics["test_accuracy"] - 0.6296) < 1e-10, "held-out accuracy is reproducible")
    require(
        abs(metrics["test_macro_f1"] - 0.61635830) < 1e-8,
        "held-out macro-F1 is reproducible",
    )
    for name, frame in [("hawkish", hawkish), ("dovish", dovish), ("stable", stable)]:
        require(len(frame) == 10, f"{name} ranking has ten rows")
        require(frame["speeches"].ge(50).all(), f"{name} ranking respects 50-speech minimum")
        require(frame["years_observed"].ge(10).all(), f"{name} ranking respects 10-year minimum")
    require(
        stable["years_with_3plus_speeches"].ge(10).all(),
        "stability ranking has ten qualifying annual cells per country",
    )

    required_figures = [
        "speeches_per_year.png",
        "selected_countries_per_year.png",
        "inflation_persistence_share.png",
        "tone_confusion_matrix.png",
        "speaker_tone_halfyear.png",
        "country_tone_extremes.png",
        "country_tone_stability.png",
    ]
    for filename in required_figures:
        path = FIGURES / filename
        require(path.exists() and path.stat().st_size > 10_000, f"{filename} exists and is nontrivial")
        image = mpimg.imread(path)
        require(min(image.shape[:2]) >= 500, f"{filename} has reviewable dimensions")

    notebook_path = ROOT / "notebooks" / "BIS_speech_analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    require(not errors, "executed notebook contains no error outputs")
    require((ROOT / "REPORT.md").stat().st_size > 10_000, "full report is present")
    print("All output validations passed.")


if __name__ == "__main__":
    main()
