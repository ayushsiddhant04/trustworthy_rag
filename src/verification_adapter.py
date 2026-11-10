"""Adapter between Member A's pipeline contract and Member B's verifier."""
from __future__ import annotations

from typing import Any
from .config import SETTINGS


def create_verification_engine(**kwargs):
    """Import Member B lazily so ingestion/UI work without model dependencies."""
    try:
        from member_b.verifier import VerificationEngine
    except ImportError as exc:
        raise RuntimeError(
            "Member B's verifier package or its dependencies are unavailable. "
            "Install the full requirements and ensure member_b/ is on the project path."
        ) from exc
    return VerificationEngine(**kwargs)


def create_relevance_checker(**kwargs):
    """Import Member B relevance checker lazily."""
    try:
        from member_b.relevance import RelevanceChecker
    except ImportError as exc:
        raise RuntimeError(
            "Member B's relevance checker is unavailable. "
            "Ensure sentence-transformers is installed."
        ) from exc
    return RelevanceChecker(**kwargs)


def verify_pipeline_output(pipeline_output: dict, engine=None,
                           enable_relevance_gate: bool | None = None,
                           relevance_checker=None) -> dict:
    """Verify each claim against its own retrieved evidence, with optional relevance gating."""
    if not isinstance(pipeline_output, dict):
        raise TypeError("pipeline_output must be a dictionary returned by src.pipeline.run")
    verifier = engine or create_verification_engine()

    gate_enabled = SETTINGS.enable_relevance_gate if enable_relevance_gate is None else bool(enable_relevance_gate)
    checker = None
    if gate_enabled:
        checker = relevance_checker or create_relevance_checker(
            model_name=SETTINGS.relevance_model,
            threshold=SETTINGS.relevance_threshold,
        )

    verified = []
    for item in pipeline_output.get("verification_inputs", []):
        claim = item.get("claim") or item.get("text") or ""
        evidence = item.get("evidence")
        if evidence is None:
            # Compatibility with earlier Member A draft, where passages and refs
            # were kept in separate arrays.
            refs = item.get("passage_refs", [])
            passages = item.get("passages", [])
            evidence = []
            for index, passage in enumerate(passages):
                ref = refs[index] if index < len(refs) else {}
                evidence.append({
                    "chunk_id": ref.get("chunk_id", f"chunk_{index+1}"),
                    "text": passage,
                    "page": ref.get("page"),
                    "source": ref.get("source"),
                    "paragraph": ref.get("paragraph"),
                })

        if gate_enabled and checker is not None:
            filtered_to_use = []
            filtered_out = []
            scores = []
            for p in (evidence or []):
                p_text = p.get("text", "") if isinstance(p, dict) else str(p)
                chk = checker.check(claim, p_text)
                sim = float(chk.get("similarity", 0.0))
                is_rel = bool(chk.get("is_relevant", False))
                chunk_id = p.get("chunk_id") if isinstance(p, dict) else "unknown_chunk"
                scores.append({"chunk_id": chunk_id, "similarity": sim, "is_relevant": is_rel})
                if is_rel:
                    filtered_to_use.append(p)
                else:
                    filtered_out.append(p)
            evidence_for_verifier = filtered_to_use
        else:
            evidence_for_verifier = evidence or []
            filtered_out = []
            scores = []

        result = verifier.verify_claim(claim, evidence_for_verifier)
        result["claim_id"] = item.get("claim_id", f"C{len(verified)+1}")
        result["claim"] = claim
        result["candidate_evidence"] = evidence or []
        result["filtered_evidence"] = filtered_out
        result["relevance_filter_applied"] = gate_enabled
        result["relevance_scores"] = scores
        verified.append(result)

    output = dict(pipeline_output)
    output["verification_results"] = verified
    output["overall_verdict"] = summarize_verdict(verified)
    return output


def summarize_verdict(results: list[dict]) -> str:
    actions = [item.get("action") for item in results]
    if not actions or all(action == "withhold" for action in actions):
        return "abstain"
    has_caveat = any(a == "caveat" for a in actions)
    has_withhold = any(a == "withhold" for a in actions)
    has_state = any(a == "state" for a in actions)
    if has_state and not has_caveat and not has_withhold:
        return "answer"
    if has_state and has_caveat and not has_withhold:
        return "answer_with_caveats"
    if has_state and has_withhold and not has_caveat:
        return "partial_answer"
    if has_caveat and has_withhold:
        return "partial_answer_with_caveats"
    if has_caveat:
        return "answer_with_caveats"
    return "answer"
