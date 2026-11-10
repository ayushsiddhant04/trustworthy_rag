
from relevance import RelevanceChecker


def test_relevance_checker_returns_expected_fields():
    checker = RelevanceChecker(threshold=0.30)

    result = checker.check(
        "Water freezes at 0 degrees Celsius.",
        "Pure water freezes at 0 degrees Celsius."
    )

    assert "similarity" in result
    assert "is_relevant" in result
    assert isinstance(result["similarity"], float)
    assert result["is_relevant"] is True


def test_irrelevant_evidence_is_rejected():
    checker = RelevanceChecker(threshold=0.30)

    result = checker.check(
        "Water freezes at 0 degrees Celsius.",
        "The city recorded heavy rainfall during monsoon."
    )

    assert result["is_relevant"] is False
