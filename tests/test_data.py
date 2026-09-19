from pathlib import Path

import pandas as pd

from bis_speeches.data import DATE_CORRECTIONS, load_and_clean, normalize_whitespace


def test_normalize_whitespace_and_missing():
    assert normalize_whitespace("  monetary\n policy\tworks ") == "monetary policy works"
    assert normalize_whitespace(pd.NA) == ""


def test_known_date_corrections_are_applied(tmp_path: Path):
    rows = []
    for index, (url, corrected) in enumerate(DATE_CORRECTIONS.items()):
        rows.append(
            {
                "url": url,
                "title": f"Speech {index}",
                "description": "Speech by a governor",
                "date": "2027-01-01 00:00:00",
                "text": "Monetary policy text.",
                "author": "Governor",
            }
        )
    path = tmp_path / "BIS_speeches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cleaned = load_and_clean(path)
    assert cleaned["date_corrected"].all()
    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == list(DATE_CORRECTIONS.values())

