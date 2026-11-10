from decision_policy import decide


def test_supported_high_confidence_is_state():
    assert decide("supported", 0.95)["action"] == "state"


def test_supported_low_confidence_is_caveat():
    assert decide("supported", 0.65)["action"] == "caveat"


def test_insufficient_is_withheld():
    assert decide("insufficient", 0.5)["action"] == "withhold"


def test_numeric_conflict_is_withheld():
    assert decide("supported", 0.99, numeric_conflict=True)["action"] == "withhold"
