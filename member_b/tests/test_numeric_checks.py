from numeric_checks import check_numeric_conflict


def test_deadline_mismatch_detected():
    result = check_numeric_conflict(
        "The application deadline is 20 November.",
        ["The application deadline is 15 November."],
    )
    assert result["numeric_conflict"] is True


def test_no_shared_cue_does_not_compare_unrelated_numbers():
    result = check_numeric_conflict(
        "There are 20 students.",
        ["The fee is 5000 rupees."],
    )
    assert result["numeric_conflict"] is False
