"""Transparent keyword signals used alongside the supervised tone model."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


# Deliberately narrow patterns: the goal is a transparent indicator of expressed
# persistence concern, not an exhaustive inflation classifier.
INFLATION_PERSISTENCE_PATTERNS: dict[str, str] = {
    "persistent_or_sticky_inflation": (
        r"\b(?:persistent|persistently|persistence|sticky|entrenched)\s+"
        r"(?:underlying\s+|core\s+|headline\s+)?inflation\b"
    ),
    "inflation_remains_high": (
        r"\binflation\s+(?:remains?|remained|continues?\s+to\s+be|is\s+still)\s+"
        r"(?:too\s+)?(?:high|elevated|above\s+(?:the\s+)?target)\b"
    ),
    "inflation_expected_to_remain_high": (
        r"\binflation\s+(?:is\s+)?(?:expected|projected|forecast)\s+to\s+"
        r"(?:remain|stay|be)\s+(?:too\s+)?(?:high|elevated|above\s+(?:the\s+)?target)\b"
    ),
    "persistent_price_pressures": (
        r"\b(?:price|inflationary|cost)\s+pressures?\s+"
        r"(?:remain|remained|persist|persisted|continue|continued|are\s+expected\s+to\s+continue)\b"
    ),
    "inflation_for_longer": (
        r"\b(?:inflation|price\s+pressures?)\b.{0,45}\b"
        r"(?:longer\s+than\s+(?:previously\s+)?expected|for\s+longer|more\s+persistent)\b"
    ),
    "second_round_or_deanchoring": (
        r"\b(?:second[ -]round\s+effects?|de[ -]?anchor(?:ed|ing)?\s+"
        r"(?:inflation\s+)?expectations?)\b"
    ),
}


def compiled_patterns() -> dict[str, re.Pattern[str]]:
    return {
        name: re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for name, pattern in INFLATION_PERSISTENCE_PATTERNS.items()
    }


def detect_inflation_persistence(text: str) -> tuple[bool, str, str]:
    """Return flag, matched pattern name, and matched phrase for one speech."""
    for name, pattern in compiled_patterns().items():
        match = pattern.search(text or "")
        if match:
            return True, name, re.sub(r"\s+", " ", match.group(0)).strip()
    return False, "", ""


def add_inflation_persistence_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach auditable persistence concern columns."""
    result = frame.copy()
    matches = result["text"].map(detect_inflation_persistence)
    result[["inflation_persistence", "persistence_pattern", "persistence_phrase"]] = (
        pd.DataFrame(matches.tolist(), index=result.index)
    )
    return result


def pattern_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pattern_name": name, "regular_expression": pattern}
            for name, pattern in INFLATION_PERSISTENCE_PATTERNS.items()
        ]
    )


def annual_persistence_share(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame["analysis_eligible"] & frame["year"].notna()]
    result = (
        valid.groupby("year", as_index=False, observed=True)
        .agg(
            speeches=("speech_id", "size"),
            persistence_speeches=("inflation_persistence", "sum"),
        )
        .sort_values("year")
    )
    result["persistence_share"] = (
        result["persistence_speeches"] / result["speeches"]
    )
    return result


def any_pattern(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text or "") for pattern in patterns)
