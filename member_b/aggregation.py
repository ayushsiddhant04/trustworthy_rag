"""Aggregate best NLI scores across retrieved passages."""
try:
    from .config import DEFAULT_ENTAILMENT_THRESHOLD, DEFAULT_CONTRADICTION_THRESHOLD
except ImportError:
    from config import DEFAULT_ENTAILMENT_THRESHOLD, DEFAULT_CONTRADICTION_THRESHOLD


def aggregate_evidence(passage_results,
                       entail_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
                       contradiction_threshold=DEFAULT_CONTRADICTION_THRESHOLD):
    """Each passage result contains max_entailment/max_contradiction and metadata."""
    supporters = [p for p in passage_results if p["max_entailment"] >= entail_threshold]
    contradictors = [p for p in passage_results if p["max_contradiction"] >= contradiction_threshold]

    if supporters and contradictors:
        status = "conflicting"
        confidence = min(
            max(p["max_entailment"] for p in supporters),
            max(p["max_contradiction"] for p in contradictors),
        )
    elif supporters:
        status = "supported"
        confidence = max(p["max_entailment"] for p in supporters)
    elif contradictors:
        status = "contradicted"
        confidence = max(p["max_contradiction"] for p in contradictors)
    else:
        status = "insufficient"
        confidence = max(
            (p.get("max_neutral", 0.0) for p in passage_results),
            default=0.0,
        )

    return {
        "status": status,
        "confidence": float(confidence),
        "supporting_passages": [p["chunk_id"] for p in supporters],
        "contradicting_passages": [p["chunk_id"] for p in contradictors],
    }
