#!/usr/bin/env python3
"""Run the complete BIS speech analysis from the project root.

The assignment-requested relative input path is the default: ./BIS_speeches.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bis_speeches.countries import attribute_jurisdictions, top_countries
from bis_speeches.data import basic_summary, load_and_clean, quality_summary
from bis_speeches.plots import (
    plot_confusion,
    plot_country_extremes,
    plot_country_stability,
    plot_persistence_share,
    plot_selected_countries,
    plot_speaker_trends,
    plot_speeches_per_year,
)
from bis_speeches.signals import (
    add_inflation_persistence_flags,
    annual_persistence_share,
    pattern_reference_table,
)
from bis_speeches.statistics import (
    country_tone_rankings,
    speaker_trend_tests,
    top_speaker_names,
)
from bis_speeches.tone import score_speeches, train_validate_tone_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./BIS_speeches.csv"),
        help="BIS CSV path (default: ./BIS_speeches.csv)",
    )
    parser.add_argument(
        "--wcb-data",
        type=Path,
        default=Path("./data/external/wcb"),
        help="Directory containing WCB train/validation/test parquet splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs"),
        help="Output directory",
    )
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, float_format="%.8f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def main() -> None:
    args = parse_args()
    output = args.output
    tables = output / "tables"
    figures = output / "figures"
    models = output / "models"
    for directory in [tables, figures, models]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading, cleaning, and validating the BIS archive", flush=True)
    speeches = load_and_clean(args.data)
    data_quality = quality_summary(speeches)
    headline, length_summary = basic_summary(speeches)
    write_csv(data_quality, tables / "data_quality.csv")
    write_csv(headline, tables / "basic_summary.csv")
    write_csv(length_summary, tables / "speech_length_summary.csv")
    corrections = speeches.loc[
        speeches["date_corrected"],
        ["speech_id", "url", "date_raw", "date", "title"],
    ]
    write_csv(corrections, tables / "date_corrections.csv")

    print("[2/7] Attributing institutional jurisdictions", flush=True)
    speeches, country_qa = attribute_jurisdictions(
        speeches, ROOT / "config" / "institution_aliases.csv"
    )
    countries_top10 = top_countries(speeches)
    write_csv(country_qa, tables / "country_attribution_qa.csv")
    write_csv(countries_top10, tables / "top10_countries.csv")
    method_by_year = (
        speeches.groupby(["year", "country_method"], as_index=False, observed=True)
        .agg(speeches=("speech_id", "size"))
        .sort_values(["year", "country_method"])
    )
    write_csv(method_by_year, tables / "country_method_by_year.csv")
    unresolved = speeches.loc[
        speeches["country_method"].eq("unresolved"),
        ["speech_id", "date", "author", "title", "description", "url"],
    ]
    write_csv(unresolved, tables / "country_unresolved_review.csv")

    print("[3/7] Detecting inflation-persistence language", flush=True)
    speeches = add_inflation_persistence_flags(speeches)
    persistence_annual = annual_persistence_share(speeches)
    write_csv(pattern_reference_table(), tables / "inflation_persistence_patterns.csv")
    write_csv(persistence_annual, tables / "inflation_persistence_annual.csv")

    print("[4/7] Training and evaluating the WCB-supervised tone model", flush=True)
    tone_model = train_validate_tone_model(args.wcb_data)
    joblib.dump(tone_model, models / "tone_model.joblib", compress=3)
    write_csv(tone_model.validation_scores, tables / "tone_validation_grid.csv")
    write_csv(pd.DataFrame([tone_model.test_metrics]), tables / "tone_test_metrics.csv")
    write_csv(tone_model.class_metrics, tables / "tone_class_metrics.csv")
    write_csv(tone_model.confusion, tables / "tone_confusion_matrix.csv", index=True)
    write_csv(tone_model.examples, tables / "tone_examples.csv")

    print("[5/7] Scoring speeches and estimating speaker/country summaries", flush=True)
    speeches = score_speeches(speeches, tone_model)
    top_speakers = top_speaker_names(speeches)
    speaker_trends, speaker_panel = speaker_trend_tests(speeches, top_speakers)
    speaker_counts = (
        speeches.loc[speeches["author"].isin(top_speakers)]
        .groupby("author", as_index=False, observed=True)
        .agg(
            speeches=("speech_id", "size"),
            tone_valid_speeches=("tone_valid", "sum"),
            mean_hawkishness=("hawkishness_score", lambda x: x[speeches.loc[x.index, "tone_valid"]].mean()),
            mean_mixedness=("tone_mixedness", lambda x: x[speeches.loc[x.index, "tone_valid"]].mean()),
        )
        .sort_values("speeches", ascending=False)
        .reset_index(drop=True)
    )
    speaker_counts.insert(0, "rank", range(1, len(speaker_counts) + 1))
    coverage, hawkish, dovish, stable = country_tone_rankings(speeches)
    write_csv(speaker_counts, tables / "top10_speakers.csv")
    write_csv(speaker_panel, tables / "speaker_halfyear_tone.csv")
    write_csv(speaker_trends, tables / "speaker_trend_tests.csv")
    write_csv(coverage, tables / "country_tone_coverage.csv")
    write_csv(hawkish, tables / "top10_hawkish_countries.csv")
    write_csv(dovish, tables / "top10_dovish_countries.csv")
    write_csv(stable, tables / "top10_stable_countries.csv")

    speech_export_columns = [
        "speech_id",
        "url",
        "title",
        "author",
        "date",
        "year",
        "word_count",
        "jurisdiction",
        "jurisdiction_category",
        "country_method",
        "institution_alias",
        "inflation_mentioned",
        "inflation_persistence",
        "persistence_pattern",
        "persistence_phrase",
        "policy_clause_count",
        "hawkishness_score",
        "tone_mixedness",
        "tone_signal_strength",
        "tone_valid",
        "tone_label",
    ]
    write_csv(speeches[speech_export_columns], tables / "speech_level_results.csv")

    print("[6/7] Rendering figures", flush=True)
    annual_speeches = plot_speeches_per_year(speeches, figures / "speeches_per_year.png")
    selected_country_annual = plot_selected_countries(
        speeches,
        ["United States", "United Kingdom", "Japan"],
        figures / "selected_countries_per_year.png",
    )
    plot_persistence_share(
        persistence_annual, figures / "inflation_persistence_share.png"
    )
    plot_confusion(tone_model.confusion, figures / "tone_confusion_matrix.png")
    plot_speaker_trends(
        speaker_panel, top_speakers, figures / "speaker_tone_halfyear.png"
    )
    plot_country_extremes(
        hawkish, dovish, figures / "country_tone_extremes.png"
    )
    plot_country_stability(stable, figures / "country_tone_stability.png")
    write_csv(annual_speeches, tables / "annual_speech_counts.csv")
    write_csv(selected_country_annual, tables / "selected_country_annual_counts.csv")

    peak = persistence_annual.loc[persistence_annual["persistence_share"].idxmax()]
    qa_lookup = country_qa.set_index("method")["speeches"]
    summary = {
        "data": {
            "source_path": str(args.data),
            "sha256": sha256(args.data),
            "speeches": int(len(speeches)),
            "start_date": speeches["date"].min().date().isoformat(),
            "end_date": speeches["date"].max().date().isoformat(),
            "unique_speakers": int(speeches.loc[speeches["author"].ne(""), "author"].nunique()),
            "date_corrections": int(speeches["date_corrected"].sum()),
            "word_count": {
                row.statistic: float(row.words) for row in length_summary.itertuples(index=False)
            },
            "inflation_mention_count": int(speeches["inflation_mentioned"].sum()),
            "inflation_mention_share": float(speeches["inflation_mentioned"].mean()),
        },
        "country_attribution": {
            "institution_alias": int(qa_lookup["institution_alias"]),
            "speaker_fallback": int(qa_lookup["speaker_fallback"]),
            "unresolved": int(qa_lookup["unresolved"]),
            "direct_conflicts_flagged": int(qa_lookup["direct_conflicts"]),
            "resolved_share": float(1 - qa_lookup["unresolved"] / len(speeches)),
        },
        "inflation_persistence": {
            "count": int(speeches["inflation_persistence"].sum()),
            "share": float(speeches["inflation_persistence"].mean()),
            "peak_year": int(peak["year"]),
            "peak_share": float(peak["persistence_share"]),
        },
        "tone_model": {
            "best_alpha": tone_model.best_alpha,
            **tone_model.test_metrics,
            "tone_valid_speeches": int(speeches["tone_valid"].sum()),
            "tone_valid_share": float(speeches["tone_valid"].mean()),
        },
        "speaker_trends": {
            "significant_after_fdr": speaker_trends.loc[
                speaker_trends["significant_fdr_5pct"], "speaker"
            ].tolist(),
            "number_significant_after_fdr": int(
                speaker_trends["significant_fdr_5pct"].sum()
            ),
        },
        "rankings": {
            "top_countries_by_volume": countries_top10["jurisdiction"].tolist(),
            "most_hawkish": hawkish["jurisdiction"].tolist(),
            "most_dovish": dovish["jurisdiction"].tolist(),
            "most_stable": stable["jurisdiction"].tolist(),
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=serializable)
        handle.write("\n")

    print("[7/7] Complete", flush=True)
    print(f"Outputs written to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
