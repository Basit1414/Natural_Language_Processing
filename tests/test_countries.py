import pandas as pd

from bis_speeches.countries import (
    affiliation_window,
    load_aliases,
    match_affiliation,
    normalize_for_match,
)


def test_normalization_handles_accents_and_punctuation():
    assert normalize_for_match("Banco de México") == "banco de mexico"


def test_event_host_is_excluded_by_boundary():
    description = (
        "Speech by Jane Doe, Governor of the Bank of Canada, at a conference "
        "hosted by the Bank of England."
    )
    window = affiliation_window(description)
    assert "Bank of Canada" in window
    assert "Bank of England" not in window


def test_speaker_role_beats_honouree_in_preamble():
    aliases = load_aliases("config/institution_aliases.csv")
    description = (
        "Lecture in honour of a former Governor of the Bank of Greece, by the President "
        "of the Deutsche Bundesbank, Professor Example, at an event."
    )
    matched = match_affiliation(affiliation_window(description), aliases)
    assert matched["jurisdiction_direct"] == "Germany"

