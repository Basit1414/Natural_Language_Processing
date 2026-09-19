"""Aggregation and inferential statistics for speaker and country tone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


RANDOM_STATE = 20260919


def top_speaker_names(frame: pd.DataFrame, n: int = 10) -> list[str]:
    return (
        frame.loc[frame["analysis_eligible"] & frame["author"].ne(""), "author"]
        .value_counts()
        .head(n)
        .index.tolist()
    )


def speaker_halfyear_panel(frame: pd.DataFrame, speakers: list[str]) -> pd.DataFrame:
    valid = frame.loc[
        frame["author"].isin(speakers) & frame["tone_valid"] & frame["date"].notna()
    ].copy()
    valid["half"] = np.where(valid["date"].dt.month.le(6), "H1", "H2")
    valid["period"] = valid["date"].dt.year.astype(str) + "-" + valid["half"]
    valid["period_index"] = valid["date"].dt.year + np.where(valid["half"].eq("H2"), 0.5, 0.0)
    return (
        valid.groupby(["author", "period", "period_index"], as_index=False, observed=True)
        .agg(
            mean_hawkishness=("hawkishness_score", "mean"),
            median_hawkishness=("hawkishness_score", "median"),
            speeches=("speech_id", "size"),
            mean_mixedness=("tone_mixedness", "mean"),
            mean_signal_strength=("tone_signal_strength", "mean"),
        )
        .sort_values(["author", "period_index"])
    )


def speaker_trend_tests(
    frame: pd.DataFrame,
    speakers: list[str],
    *,
    minimum_periods: int = 6,
    hac_lags: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test linear half-year trends with HAC errors and BH multiple-testing control."""
    panel = speaker_halfyear_panel(frame, speakers)
    total_counts = (
        frame.loc[frame["author"].isin(speakers)]
        .groupby("author", observed=True)["speech_id"]
        .size()
        .to_dict()
    )
    valid_counts = (
        frame.loc[frame["author"].isin(speakers) & frame["tone_valid"]]
        .groupby("author", observed=True)["speech_id"]
        .size()
        .to_dict()
    )
    rows: list[dict[str, float | int | str]] = []
    for speaker in speakers:
        group = panel.loc[panel["author"].eq(speaker)].copy()
        if len(group) < minimum_periods or group["mean_hawkishness"].nunique() < 2:
            rows.append(
                {
                    "speaker": speaker,
                    "total_speeches": int(total_counts.get(speaker, 0)),
                    "tone_valid_speeches": int(valid_counts.get(speaker, 0)),
                    "halfyear_periods": int(len(group)),
                    "slope_per_year": np.nan,
                    "hac_t_statistic": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                }
            )
            continue
        centered_time = group["period_index"] - group["period_index"].mean()
        design = sm.add_constant(centered_time.to_numpy())
        fitted = sm.OLS(group["mean_hawkishness"].to_numpy(), design).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}
        )
        rows.append(
            {
                "speaker": speaker,
                "total_speeches": int(total_counts.get(speaker, 0)),
                "tone_valid_speeches": int(valid_counts.get(speaker, 0)),
                "halfyear_periods": int(len(group)),
                "slope_per_year": float(fitted.params[1]),
                "hac_t_statistic": float(fitted.tvalues[1]),
                "p_value": float(fitted.pvalues[1]),
                "r_squared": float(fitted.rsquared),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    testable = result["p_value"].notna()
    if testable.any():
        result.loc[testable, "fdr_q_value"] = multipletests(
            result.loc[testable, "p_value"], method="fdr_bh"
        )[1]
    result["significant_fdr_5pct"] = result["fdr_q_value"].lt(0.05).fillna(False)
    result["trend_direction"] = np.select(
        [
            result["slope_per_year"].gt(0),
            result["slope_per_year"].lt(0),
        ],
        ["more hawkish", "more dovish"],
        default="not estimated",
    )
    return result, panel


def _bootstrap_mean_interval(values: np.ndarray, resamples: int = 1_000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_STATE + len(values))
    means = np.empty(resamples)
    for index in range(resamples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def country_tone_rankings(
    frame: pd.DataFrame,
    *,
    minimum_speeches: int = 50,
    minimum_years: int = 10,
    minimum_annual_speeches: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank national jurisdictions by average tone and interannual stability."""
    valid = frame.loc[
        frame["tone_valid"]
        & frame["jurisdiction_category"].eq("national")
        & frame["year"].notna()
    ].copy()
    annual = (
        valid.groupby(["jurisdiction", "year"], as_index=False, observed=True)
        .agg(
            annual_mean_hawkishness=("hawkishness_score", "mean"),
            annual_speeches=("speech_id", "size"),
        )
        .sort_values(["jurisdiction", "year"])
    )
    usable_annual = annual.loc[annual["annual_speeches"].ge(minimum_annual_speeches)]
    coverage = (
        valid.groupby("jurisdiction", as_index=False, observed=True)
        .agg(
            speeches=("speech_id", "size"),
            speakers=("author", "nunique"),
            years_observed=("year", "nunique"),
            mean_hawkishness=("hawkishness_score", "mean"),
            median_hawkishness=("hawkishness_score", "median"),
            mean_mixedness=("tone_mixedness", "mean"),
            mean_signal_strength=("tone_signal_strength", "mean"),
        )
        .sort_values("speeches", ascending=False)
    )
    primary_institution = (
        valid.loc[valid["institution_alias"].ne("")]
        .groupby(["jurisdiction", "institution_alias"], as_index=False, observed=True)
        .size()
        .sort_values(
            ["jurisdiction", "size", "institution_alias"],
            ascending=[True, False, True],
        )
        .drop_duplicates("jurisdiction")
        .rename(columns={"institution_alias": "central_bank"})[
            ["jurisdiction", "central_bank"]
        ]
    )
    coverage = coverage.merge(primary_institution, on="jurisdiction", how="left")
    stable_year_counts = (
        usable_annual.groupby("jurisdiction", observed=True)["year"]
        .nunique()
        .rename("years_with_3plus_speeches")
    )
    stability = (
        usable_annual.groupby("jurisdiction", observed=True)["annual_mean_hawkishness"]
        .agg(annual_tone_sd="std", annual_tone_range=lambda values: values.max() - values.min())
        .reset_index()
    )
    coverage = coverage.merge(stable_year_counts, on="jurisdiction", how="left")
    coverage = coverage.merge(stability, on="jurisdiction", how="left")
    coverage["years_with_3plus_speeches"] = (
        coverage["years_with_3plus_speeches"].fillna(0).astype(int)
    )

    intervals = []
    for jurisdiction, group in valid.groupby("jurisdiction", observed=True):
        lower, upper = _bootstrap_mean_interval(group["hawkishness_score"].to_numpy())
        intervals.append(
            {
                "jurisdiction": jurisdiction,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
            }
        )
    coverage = coverage.merge(pd.DataFrame(intervals), on="jurisdiction", how="left")
    eligible = coverage.loc[
        coverage["speeches"].ge(minimum_speeches)
        & coverage["years_observed"].ge(minimum_years)
    ].copy()
    hawkish = eligible.nlargest(10, "mean_hawkishness").reset_index(drop=True)
    dovish = eligible.nsmallest(10, "mean_hawkishness").reset_index(drop=True)
    hawkish.insert(0, "rank", range(1, len(hawkish) + 1))
    dovish.insert(0, "rank", range(1, len(dovish) + 1))

    stable_eligible = eligible.loc[
        eligible["years_with_3plus_speeches"].ge(minimum_years)
        & eligible["annual_tone_sd"].notna()
    ].copy()
    stable = stable_eligible.nsmallest(10, "annual_tone_sd").reset_index(drop=True)
    stable.insert(0, "rank", range(1, len(stable) + 1))
    return coverage, hawkish, dovish, stable
