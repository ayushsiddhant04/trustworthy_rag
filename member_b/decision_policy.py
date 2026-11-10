"""Translate verification status into a safe UI action."""
try:
    from .config import HIGH_CONFIDENCE_THRESHOLD
except ImportError:
    from config import HIGH_CONFIDENCE_THRESHOLD


def decide(status, confidence, numeric_conflict=False,
           high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD):
    if status == "conflicting":
        if numeric_conflict:
            return {"action": "caveat", "reason": "Retrieved passages disagree (including a detected date/numeric mismatch between sources); show the disagreement rather than presenting this claim as settled."}
        return {"action": "caveat", "reason": "Retrieved passages disagree; show the disagreement rather than presenting this claim as settled."}
    if numeric_conflict:
        return {"action": "withhold", "reason": "A relevant number/date mismatch was detected; do not state this claim as fact."}
    if status == "supported":
        if confidence >= high_confidence_threshold:
            return {"action": "state", "reason": "The retrieved evidence supports this claim."}
        return {"action": "caveat", "reason": "The evidence appears supportive, but the verification confidence is below the high-confidence threshold."}
    if status == "contradicted":
        return {"action": "withhold", "reason": "The retrieved evidence contradicts this claim."}
    return {"action": "withhold", "reason": "The retrieved evidence does not establish this claim."}
