from bis_speeches.signals import detect_inflation_persistence


def test_persistence_phrase_is_detected():
    detected, name, phrase = detect_inflation_persistence(
        "Inflation remains elevated and monetary policy must stay vigilant."
    )
    assert detected
    assert name == "inflation_remains_high"
    assert "Inflation remains elevated" in phrase


def test_generic_inflation_mention_is_not_persistence():
    detected, name, phrase = detect_inflation_persistence(
        "Inflation was 2 percent in the latest release."
    )
    assert not detected
    assert name == ""
    assert phrase == ""

