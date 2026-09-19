"""Deterministic institutional-jurisdiction attribution with provenance."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_BOUNDARY = re.compile(
    r"(?:,\s+|\s+)(?:at|before|during|on\s+the\s+occasion\s+of|for\s+the|to\s+the)\s+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Alias:
    jurisdiction: str
    category: str
    raw: str
    normalized: str
    pattern: re.Pattern[str]


def normalize_for_match(value: object) -> str:
    if pd.isna(value):
        return ""
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def affiliation_window(description: str, text: str = "") -> str:
    """Keep only the leading byline, avoiding venue/host institutions where possible."""
    source = (description or "").strip()
    if not source:
        source = (text or "")[:1000].strip()
    source = source[:1000]
    boundary = EVENT_BOUNDARY.search(source)
    if boundary and boundary.start() >= 35:
        source = source[: boundary.start()]
    return source


def load_aliases(path: str | Path) -> list[Alias]:
    registry = pd.read_csv(path).dropna()
    required = {"jurisdiction", "category", "alias"}
    if not required.issubset(registry.columns):
        raise ValueError(f"Alias registry must contain {sorted(required)}")
    aliases: list[Alias] = []
    for row in registry.itertuples(index=False):
        normalized = normalize_for_match(row.alias)
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])")
        aliases.append(
            Alias(
                jurisdiction=str(row.jurisdiction),
                category=str(row.category),
                raw=str(row.alias),
                normalized=normalized,
                pattern=pattern,
            )
        )
    # Specific names take precedence when one name contains another.
    return sorted(aliases, key=lambda item: len(item.normalized), reverse=True)


def match_affiliation(window: str, aliases: list[Alias]) -> dict[str, object]:
    normalized = normalize_for_match(window)
    author_marker = re.search(
        r"\b(?:speech|remarks?|address|lecture|comments?|statement|presentation|"
        r"testimony|keynote|text)\b.{0,120}?\bby\b",
        normalized,
    )
    speaker_starts = author_marker.end() if author_marker else 0
    matches: list[tuple[int, int, int, Alias]] = []
    for alias in aliases:
        found = alias.pattern.search(normalized)
        if found:
            prefix = normalized[max(0, found.start() - 120) : found.start()]
            score = 0
            if re.search(
                r"\b(?:governor|president|chairman|chair|deputy governor|vice governor|"
                r"board member|member|director|chief economist)(?:\s+of)?(?:\s+the)?\s*$",
                prefix,
            ):
                score += 6
            elif re.search(r"\b(?:governor|president|member|director)\b.{0,55}\bof(?:\s+the)?\s*$", prefix):
                score += 4
            if author_marker:
                score += 2 if found.start() >= speaker_starts else -5
            if re.search(r"\b(?:hosted|organised|organized|sponsored)\s+by(?:\s+the)?\s*$", prefix):
                score -= 7
            matches.append((-score, found.start(), -len(alias.normalized), alias))

    if not matches:
        return {
            "jurisdiction_direct": "",
            "jurisdiction_category_direct": "",
            "institution_alias": "",
            "country_direct_conflict": False,
            "country_direct_candidates": "",
        }

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    jurisdictions = list(dict.fromkeys(item[3].jurisdiction for item in matches))
    first = matches[0][3]
    return {
        "jurisdiction_direct": first.jurisdiction,
        "jurisdiction_category_direct": first.category,
        "institution_alias": first.raw,
        "country_direct_conflict": len(jurisdictions) > 1,
        "country_direct_candidates": " | ".join(jurisdictions),
    }


def _speaker_fallback_table(
    direct: pd.DataFrame, min_direct: int, min_dominance: float
) -> pd.DataFrame:
    observed = direct.loc[
        direct["author"].ne("") & direct["jurisdiction_direct"].ne("")
    ].copy()
    if observed.empty:
        return pd.DataFrame(columns=["author", "speaker_jurisdiction", "speaker_category"])

    counts = (
        observed.groupby(
            ["author", "jurisdiction_direct", "jurisdiction_category_direct"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby("author", observed=True)["n"].transform("sum")
    counts["dominance"] = counts["n"] / totals
    counts = counts.sort_values(
        ["author", "n", "jurisdiction_direct"], ascending=[True, False, True]
    )
    winners = counts.drop_duplicates("author")
    winners = winners.loc[
        winners["n"].ge(min_direct) & winners["dominance"].ge(min_dominance)
    ].copy()
    return winners.rename(
        columns={
            "jurisdiction_direct": "speaker_jurisdiction",
            "jurisdiction_category_direct": "speaker_category",
            "n": "speaker_direct_evidence_n",
            "dominance": "speaker_direct_dominance",
        }
    )[
        [
            "author",
            "speaker_jurisdiction",
            "speaker_category",
            "speaker_direct_evidence_n",
            "speaker_direct_dominance",
        ]
    ]


def attribute_jurisdictions(
    frame: pd.DataFrame,
    alias_path: str | Path,
    *,
    min_speaker_evidence: int = 3,
    min_speaker_dominance: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a jurisdiction using byline aliases, then a conservative speaker fallback."""
    result = frame.copy()
    aliases = load_aliases(alias_path)
    result["affiliation_window"] = [
        affiliation_window(description, text)
        for description, text in zip(result["description"], result["text"], strict=True)
    ]
    direct = result["affiliation_window"].map(lambda value: match_affiliation(value, aliases))
    direct_frame = pd.DataFrame(direct.tolist(), index=result.index)
    result = pd.concat([result, direct_frame], axis=1)

    fallback = _speaker_fallback_table(result, min_speaker_evidence, min_speaker_dominance)
    result = result.merge(fallback, on="author", how="left", validate="many_to_one")

    direct_available = result["jurisdiction_direct"].ne("")
    speaker_available = result["speaker_jurisdiction"].notna()
    result["jurisdiction"] = np.select(
        [direct_available, speaker_available],
        [result["jurisdiction_direct"], result["speaker_jurisdiction"]],
        default="Unresolved",
    )
    result["jurisdiction_category"] = np.select(
        [direct_available, speaker_available],
        [result["jurisdiction_category_direct"], result["speaker_category"]],
        default="unresolved",
    )
    result["country_method"] = np.select(
        [direct_available, speaker_available],
        ["institution_alias", "speaker_fallback"],
        default="unresolved",
    )

    # Conflicting direct matches are retained for audit; the earliest byline match is
    # used because some descriptions contain a second professional role.
    qa = pd.DataFrame(
        [
            ("institution_alias", int(result["country_method"].eq("institution_alias").sum())),
            ("speaker_fallback", int(result["country_method"].eq("speaker_fallback").sum())),
            ("unresolved", int(result["country_method"].eq("unresolved").sum())),
            ("direct_conflicts", int(result["country_direct_conflict"].sum())),
            ("total", int(len(result))),
        ],
        columns=["method", "speeches"],
    )
    qa["share"] = qa["speeches"] / len(result)
    return result, qa


def top_countries(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the most represented national jurisdictions."""
    national = frame.loc[
        frame["analysis_eligible"] & frame["jurisdiction_category"].eq("national")
    ]
    result = (
        national.groupby("jurisdiction", as_index=False, observed=True)
        .agg(speeches=("speech_id", "size"), speakers=("author", "nunique"))
        .sort_values(["speeches", "jurisdiction"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result
