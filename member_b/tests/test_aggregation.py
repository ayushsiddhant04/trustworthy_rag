from aggregation import aggregate_evidence


def test_supported():
    result = aggregate_evidence([{"chunk_id": "a", "max_entailment": 0.9, "max_contradiction": 0.1, "max_neutral": 0.0}])
    assert result["status"] == "supported"


def test_contradicted():
    result = aggregate_evidence([{"chunk_id": "a", "max_entailment": 0.1, "max_contradiction": 0.9, "max_neutral": 0.0}])
    assert result["status"] == "contradicted"


def test_insufficient():
    result = aggregate_evidence([{"chunk_id": "a", "max_entailment": 0.2, "max_contradiction": 0.2, "max_neutral": 0.6}])
    assert result["status"] == "insufficient"


def test_conflicting_passages():
    rows = [
        {"chunk_id": "a", "max_entailment": 0.9, "max_contradiction": 0.1, "max_neutral": 0.0},
        {"chunk_id": "b", "max_entailment": 0.1, "max_contradiction": 0.9, "max_neutral": 0.0},
    ]
    assert aggregate_evidence(rows)["status"] == "conflicting"
