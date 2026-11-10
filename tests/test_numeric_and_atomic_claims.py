"""Regression tests for date normalization, context-aware number checks, and atomic claim extraction."""
from __future__ import annotations

from pathlib import Path
import pytest

from member_b.numeric_checks import check_numeric_conflict, extract_dates, dates_compatible
from src.claim_extraction import extract_claims, split_sentences
from src.pipeline import run as run_pipeline, clear_pipeline_cache
from src.safe_composer import compose_safe_answer
from src.verification_adapter import verify_pipeline_output


# 1. Calendar year 2026 versus fee 500
def test_calendar_year_versus_fee():
    claim = "The application fee is 500 rupees in 2026."
    evidence = ["The application fee is 500 rupees."]
    res = check_numeric_conflict(claim, evidence)
    assert res["numeric_conflict"] is False
    assert "026" not in res["reason"]
    assert "2026" not in res["reason"]


# 2. Equivalent date representations
def test_equivalent_date_representations():
    # Canonical date matching
    d1 = extract_dates("15 November 2026")[0][0]
    d2 = extract_dates("November 15, 2026")[0][0]
    d3 = extract_dates("15/11/2026")[0][0]
    d4 = extract_dates("15-11-2026")[0][0]
    d5 = extract_dates("2026-11-15")[0][0]
    d6 = extract_dates("15th Nov 2026")[0][0]

    assert dates_compatible(d1, d2) is True
    assert dates_compatible(d1, d3) is True
    assert dates_compatible(d1, d4) is True
    assert dates_compatible(d1, d5) is True
    assert dates_compatible(d1, d6) is True

    # No conflict when date format differs
    res_equiv = check_numeric_conflict(
        "The application deadline is November 15, 2026.",
        ["The application deadline is 15/11/2026."],
    )
    assert res_equiv["numeric_conflict"] is False

    # Conflict when dates differ
    res_diff = check_numeric_conflict(
        "The application deadline is 20 November 2026.",
        ["The application deadline is 15 November 2026."],
    )
    assert res_diff["numeric_conflict"] is True
    assert "deadline mismatch" in res_diff["reason"].lower()


# 3. Correct fee and incorrect deadline
def test_correct_fee_and_incorrect_deadline():
    evidence = ["The last date for submitting the application is 15 November 2026. The application fee is 500 rupees."]
    claim_deadline = "The application deadline is 30 November 2026."
    claim_fee = "The registration fee is 500 rupees."

    res_dl = check_numeric_conflict(claim_deadline, evidence)
    assert res_dl["numeric_conflict"] is True
    assert "deadline mismatch" in res_dl["reason"].lower()
    assert "fee" not in res_dl["reason"].lower()

    res_fee = check_numeric_conflict(claim_fee, evidence)
    assert res_fee["numeric_conflict"] is False


# 4. Incorrect fee and correct deadline
def test_incorrect_fee_and_correct_deadline():
    evidence = ["The last date for submitting the application is 15 November 2026. The application fee is 500 rupees."]
    claim_deadline = "The application deadline is 15 November 2026."
    claim_fee = "The registration fee is 1500 rupees."

    res_dl = check_numeric_conflict(claim_deadline, evidence)
    assert res_dl["numeric_conflict"] is False

    res_fee = check_numeric_conflict(claim_fee, evidence)
    assert res_fee["numeric_conflict"] is True
    assert "fee mismatch" in res_fee["reason"].lower()
    assert "deadline" not in res_fee["reason"].lower()


# 5. Unrelated numbers in the same passage
def test_unrelated_numbers_in_same_passage():
    claim = "Students must maintain 75 percent attendance."
    evidence = ["In 2026, students must maintain 75 percent attendance. The fee is 500 rupees and 120 candidates enrolled."]
    res = check_numeric_conflict(claim, evidence)
    assert res["numeric_conflict"] is False


# 6. Compound factual claims extraction
def test_compound_factual_claims_extraction():
    compound = "The deadline is 30 November 2026 and the fee is 500 rupees."
    claims = extract_claims(compound, use_llm=False)
    assert len(claims) == 2
    assert "The deadline is 30 November 2026." in claims
    assert "The fee is 500 rupees." in claims

    # Grammatical non-compound sentences must not be split
    np1 = "Application forms and fee receipts must be submitted."
    claims_np1 = extract_claims(np1, use_llm=False)
    assert len(claims_np1) == 1
    assert claims_np1[0] == "Application forms and fee receipts must be submitted."

    np2 = "Notice A and Notice B were published."
    claims_np2 = extract_claims(np2, use_llm=False)
    assert len(claims_np2) == 1


# 7. Independent verification of extracted claims
def test_independent_verification_of_extracted_claims(monkeypatch):
    clear_pipeline_cache()
    doc_path = Path("data/demo/application_notice.txt")
    question = "What is the application deadline and fee amount?"
    compound_answer = "The application deadline is 30 November 2026 and the registration fee is 500 rupees."

    pipe_out = run_pipeline(question, doc_path, answer=compound_answer, mode="manual")
    # Must decompose into 2 independent claims
    assert len(pipe_out["claims"]) == 2
    claim_texts = [c["claim"] for c in pipe_out["claims"]]
    assert any("30 november" in c.lower() for c in claim_texts)
    assert any("500 rupees" in c.lower() for c in claim_texts)

    # Stub engine with predictable NLI responses
    class FakeNLI:
        def predict_pairs(self, pairs):
            out = []
            for ev, cl in pairs:
                if "500" in ev and "500" in cl:
                    out.append({"scores": {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}, "logits": [0, 3, -1]})
                elif "15 november" in ev.lower() and "30 november" in cl.lower():
                    out.append({"scores": {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01}, "logits": [3, -1, -1]})
                else:
                    out.append({"scores": {"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8}, "logits": [0, 0, 2]})
            return out

    from member_b.verifier import VerificationEngine
    from member_b import verifier as verifier_module

    class FakeTok:
        def encode(self, t, add_special_tokens=False):
            return t.split()
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    monkeypatch.setattr(verifier_module, "load_tokenizer", lambda m: FakeTok())
    engine = VerificationEngine(nli=FakeNLI())

    verif_out = verify_pipeline_output(pipe_out, engine=engine)
    results = verif_out["verification_results"]
    assert len(results) == 2

    # Deadline claim
    dl_res = [r for r in results if "30 november" in r["claim"].lower()][0]
    assert dl_res["status"] == "contradicted"
    assert dl_res["action"] == "withhold"

    # Fee claim
    fee_res = [r for r in results if "500 rupees" in r["claim"].lower()][0]
    assert fee_res["status"] == "supported"
    assert fee_res["action"] == "state"

    # Safe composition: deadline is withheld, fee is stated
    safe = compose_safe_answer(results)
    assert safe["verdict"] in ("partial_answer", "answer_with_caveats")
    assert "The registration fee is 500 rupees." in safe["answer"]
    assert "30 November" not in safe["answer"]
    assert len(safe["withheld_claims"]) == 1
    assert len(safe["stated_claims"]) == 1

