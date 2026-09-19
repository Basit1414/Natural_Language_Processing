from bis_speeches.tone import extract_policy_clauses


def test_contrastive_policy_sentence_is_split():
    text = (
        "Inflation remains elevated, but weaker demand may justify easing the policy rate. "
        "This unrelated sentence discusses a conference venue."
    )
    clauses = extract_policy_clauses(text)
    assert len(clauses) == 2
    assert "Inflation remains elevated" in clauses[0]
    assert "weaker demand" in clauses[1]


def test_non_policy_text_returns_no_clauses():
    assert extract_policy_clauses("Welcome to today's payments conference.") == []
