"""Consistent, publication-ready charts for the analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NAVY = "#173F5F"
BLUE = "#20639B"
GREEN = "#3CAEA3"
GOLD = "#F6D55C"
RED = "#ED553B"
GRAY = "#667085"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_speeches_per_year(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    set_style()
    annual = (
        frame.loc[frame["analysis_eligible"]]
        .groupby("year", as_index=False, observed=True)
        .agg(speeches=("speech_id", "size"))
        .sort_values("year")
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(annual["year"].astype(int), annual["speeches"], color=BLUE, width=0.82)
    ax.set(title="BIS central-bank speeches per year", xlabel="Year", ylabel="Speeches")
    ax.text(
        0.995,
        0.98,
        "2026 is partial through 22 June",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY,
        fontsize=9,
    )
    sns.despine(ax=ax)
    _save(fig, path)
    return annual


def plot_selected_countries(
    frame: pd.DataFrame, countries: list[str], path: str | Path
) -> pd.DataFrame:
    set_style()
    annual = (
        frame.loc[
            frame["analysis_eligible"]
            & frame["jurisdiction"].isin(countries)
            & frame["year"].notna()
        ]
        .groupby(["year", "jurisdiction"], as_index=False, observed=True)
        .agg(speeches=("speech_id", "size"))
        .sort_values(["jurisdiction", "year"])
    )
    full_years = pd.DataFrame(
        {
            "year": range(int(frame["year"].min()), int(frame["year"].max()) + 1)
        }
    )
    completed = []
    for country in countries:
        country_frame = full_years.merge(
            annual.loc[annual["jurisdiction"].eq(country), ["year", "speeches"]],
            on="year",
            how="left",
        )
        country_frame["jurisdiction"] = country
        country_frame["speeches"] = country_frame["speeches"].fillna(0).astype(int)
        completed.append(country_frame)
    annual_complete = pd.concat(completed, ignore_index=True)
    palette = [BLUE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for country, color in zip(countries, palette, strict=False):
        subset = annual_complete.loc[annual_complete["jurisdiction"].eq(country)]
        ax.plot(subset["year"], subset["speeches"], label=country, linewidth=2, color=color)
    ax.set(
        title="Annual speech volume for three selected countries",
        xlabel="Year",
        ylabel="Speeches",
    )
    ax.legend(frameon=False, ncol=3)
    sns.despine(ax=ax)
    _save(fig, path)
    return annual_complete


def plot_persistence_share(annual: pd.DataFrame, path: str | Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        annual["year"],
        annual["persistence_share"] * 100,
        color=RED,
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    ax.set(
        title="Share of speeches expressing inflation-persistence concern",
        xlabel="Year",
        ylabel="Share of speeches (%)",
    )
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    _save(fig, path)


def plot_confusion(confusion: pd.DataFrame, path: str | Path) -> None:
    set_style()
    row_sum = confusion.sum(axis=1).replace(0, np.nan)
    normalized = confusion.div(row_sum, axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Within-class share"},
        ax=ax,
    )
    ax.set(title="WCB held-out test confusion matrix", xlabel="Predicted", ylabel="Actual")
    _save(fig, path)


def plot_speaker_trends(panel: pd.DataFrame, speakers: list[str], path: str | Path) -> None:
    set_style()
    fig, axes = plt.subplots(5, 2, figsize=(13, 16), sharey=True)
    for ax, speaker in zip(axes.flat, speakers, strict=False):
        subset = panel.loc[panel["author"].eq(speaker)]
        ax.plot(
            subset["period_index"],
            subset["mean_hawkishness"],
            color=BLUE,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
        ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle="--")
        ax.set_title(speaker, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Half-year mean hawkishness for the ten most frequent speakers", y=1.005)
    fig.supxlabel("Year (H2 shown at year + 0.5)")
    fig.supylabel("Hawkishness score (0=dovish, 1=hawkish)")
    fig.tight_layout()
    _save(fig, path)


def plot_country_extremes(
    hawkish: pd.DataFrame, dovish: pd.DataFrame, path: str | Path
) -> None:
    set_style()
    left = dovish.sort_values("mean_hawkishness", ascending=True)
    right = hawkish.sort_values("mean_hawkishness", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True)
    axes[0].barh(left["jurisdiction"], left["mean_hawkishness"], color=GREEN)
    axes[0].set_title("Ten most dovish eligible countries")
    axes[1].barh(right["jurisdiction"], right["mean_hawkishness"], color=RED)
    axes[1].set_title("Ten most hawkish eligible countries")
    for ax in axes:
        ax.axvline(0.5, color=GRAY, linewidth=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean speech hawkishness")
        sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, path)


def plot_country_stability(stable: pd.DataFrame, path: str | Path) -> None:
    set_style()
    ordered = stable.sort_values("annual_tone_sd", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(ordered["jurisdiction"], ordered["annual_tone_sd"], color=NAVY)
    ax.set(
        title="Countries with the most stable annual average tone",
        xlabel="Standard deviation of annual mean hawkishness (lower is more stable)",
        ylabel="",
    )
    sns.despine(ax=ax)
    _save(fig, path)
