"""Build a conservative user-facing answer from Member B decision outputs."""
from __future__ import annotations


def compose_safe_answer(verification_results: list[dict]) -> dict:
    """Compose verified answer and assign consistent application-level verdict.

    Verdict Taxonomy:
    - answer: Stated claims only; zero caveats, zero withheld claims.
    - answer_with_caveats: Stated claims + caveat claims; zero withheld claims.
    - partial_answer: Stated claims + withheld claims; zero caveat claims.
    - partial_answer_with_caveats: Stated claims + caveat claims + withheld claims.
    - abstain: Zero stated or caveated claims survive; all claims are withheld.
    """
    stated, caveated, withheld = [], [], []
    for item in verification_results or []:
        claim = (item.get("claim") or item.get("text") or item.get("claim_text") or "").strip()
        action = item.get("action", "withhold")
        status = item.get("status", "insufficient")
        if not claim:
            continue
        if action == "state" and status == "supported":
            stated.append(claim)
        elif action == "caveat":
            caveated.append({"claim": claim, "status": status, "reason": item.get("reason", "")})
        else:
            withheld.append({"claim": claim, "status": status, "reason": item.get("reason", "")})

    lines = list(stated)
    for item in caveated:
        if item["status"] == "conflicting":
            lines.append(f"The available document passages disagree about this point: {item['claim']}")
        else:
            lines.append(f"The evidence is not fully conclusive about this point: {item['claim']}")

    if not lines:
        answer = "There is not enough reliable document evidence to provide a verified answer to this question."
        verdict = "abstain"
    elif caveated and not withheld:
        answer = " ".join(lines)
        verdict = "answer_with_caveats"
    elif withheld and not caveated:
        answer = " ".join(lines)
        verdict = "partial_answer"
    elif caveated and withheld:
        answer = " ".join(lines)
        verdict = "partial_answer_with_caveats"
    else:
        answer = " ".join(lines)
        verdict = "answer"

    return {
        "answer": answer,
        "verdict": verdict,
        "stated_claims": stated,
        "caveated_claims": caveated,
        "withheld_claims": withheld,
        "counts": {
            "stated": len(stated),
            "caveated": len(caveated),
            "withheld": len(withheld),
        },
    }
