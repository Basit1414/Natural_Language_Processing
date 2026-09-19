"""Dataset loading, cleaning, and auditable quality checks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["url", "title", "description", "date", "text", "author"]

# The archive contains two syntactically valid 2027 dates whose years conflict with
# their BIS URL identifiers and live BIS pages. We retain the archive's month/day and
# repair only the year so the transformation is minimal and reproducible.
DATE_CORRECTIONS = {
    "https://www.bis.org/review/r250710i.htm": "2025-07-09",
    "https://www.bis.org/review/r260603d.htm": "2026-05-28",
}


def normalize_whitespace(value: object) -> str:
    """Collapse Unicode-ish whitespace and convert missing values to an empty string."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_speech_id(url: str) -> str:
    """Return a short stable ID derived from the original BIS URL."""
    return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def load_and_clean(path: str | Path = "./BIS_speeches.csv") -> pd.DataFrame:
    """Load the BIS CSV and create analysis-ready fields without dropping observations."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_data.py` from the project root."
        )

    frame = pd.read_csv(path, low_memory=False)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema {list(frame.columns)}; expected {EXPECTED_COLUMNS}."
        )

    frame = frame.copy()
    frame["date_raw"] = frame["date"].astype(str)
    for column in ["url", "title", "description", "text", "author"]:
        frame[column] = frame[column].map(normalize_whitespace)

    parsed = pd.to_datetime(frame["date_raw"], errors="coerce")
    frame["date_corrected"] = False
    for url, corrected_date in DATE_CORRECTIONS.items():
        mask = frame["url"].eq(url)
        if mask.any():
            parsed.loc[mask] = pd.Timestamp(corrected_date)
            frame.loc[mask, "date_corrected"] = True

    frame["date"] = parsed
    frame["speech_id"] = frame["url"].map(stable_speech_id)
    frame["year"] = frame["date"].dt.year.astype("Int64")
    frame["month"] = frame["date"].dt.month.astype("Int64")
    frame["word_count"] = frame["text"].str.split().str.len().fillna(0).astype(int)
    frame["inflation_mentioned"] = frame["text"].str.contains(
        r"\binflation(?:ary)?\b", case=False, regex=True, na=False
    )
    frame["analysis_eligible"] = frame["date"].notna() & frame["text"].ne("")
    return frame


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact, machine-readable quality checks."""
    rows = [
        ("rows", len(frame)),
        ("missing_date", int(frame["date"].isna().sum())),
        ("missing_author", int(frame["author"].eq("").sum())),
        ("missing_title", int(frame["title"].eq("").sum())),
        ("missing_text", int(frame["text"].eq("").sum())),
        ("missing_description", int(frame["description"].eq("").sum())),
        ("duplicate_rows", int(frame[EXPECTED_COLUMNS].duplicated().sum())),
        ("duplicate_urls", int(frame["url"].duplicated().sum())),
        ("duplicate_texts", int(frame["text"].duplicated().sum())),
        ("corrected_dates", int(frame["date_corrected"].sum())),
        ("analysis_eligible", int(frame["analysis_eligible"].sum())),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def basic_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return headline corpus metrics and speech-length statistics."""
    valid = frame.loc[frame["analysis_eligible"]]
    headline = pd.DataFrame(
        [
            ("Total speeches", int(len(valid))),
            ("Start date", valid["date"].min().date().isoformat()),
            ("End date", valid["date"].max().date().isoformat()),
            ("Unique speakers", int(valid.loc[valid["author"].ne(""), "author"].nunique())),
            (
                "Speeches mentioning inflation",
                int(valid["inflation_mentioned"].sum()),
            ),
            (
                "Share mentioning inflation",
                float(valid["inflation_mentioned"].mean()),
            ),
        ],
        columns=["metric", "value"],
    )
    lengths = valid["word_count"].astype(float)
    length_summary = pd.DataFrame(
        {
            "statistic": ["minimum", "maximum", "mean", "median", "standard_deviation"],
            "words": [
                float(lengths.min()),
                float(lengths.max()),
                float(lengths.mean()),
                float(lengths.median()),
                float(lengths.std(ddof=1)),
            ],
        }
    )
    return headline, length_summary


def dataframe_checksum(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash selected columns for deterministic regression checks."""
    values = pd.util.hash_pandas_object(frame[columns], index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan
