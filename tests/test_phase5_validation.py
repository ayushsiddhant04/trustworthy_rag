"""Phase 5 Comprehensive End-to-End Application Validation Suite.

Validates:
1. Document Review workflow across formats (PDF, DOCX, TXT) and answer modes (generate, manual, demo_bad).
2. Real DeBERTa-v3 model inference on the three synthetic benchmark scenarios without mocks.
3. Safe answer composition across all decision types (state, caveat, withhold, abstention).
4. Audit CSV export formatting, RFC 4180 compliance, and secret leakage prevention.
5. Session state transitions, document replacement, and failure edge cases.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path
import docx
import fitz
import pandas as pd
import pytest

from member_b.verifier import VerificationEngine
from src.answer_generation import generate_answer_with_meta
from src.config import SETTINGS
from src.ingestion import load_document, DocumentIngestionError
from src.pipeline import run as run_pipeline, clear_pipeline_cache
from src.safe_composer import compose_safe_answer
from src.ui.components import format_citation, generate_audit_csv
from src.verification_adapter import verify_pipeline_output


# ==============================================================================
# TASK 2: Document Review Workflow (PDF, TXT, DOCX, & Modes)
# ==============================================================================

def test_document_review_formats_and_citations():
    """Verify ingestion and provenance citations for PDF, DOCX, and TXT."""
    clear_pipeline_cache()

    # 1. Multi-page Synthetic PDF
    pdf_doc = fitz.open()
    page1 = pdf_doc.new_page()
    page1.insert_text((50, 72), "Fellowship Regulations Page 1.\nSubmission deadline is 15 November 2026.")
    page2 = pdf_doc.new_page()
    page2.insert_text((50, 72), "Fellowship Regulations Page 2.\nApplicants must submit all forms online.")
    pdf_bytes = pdf_doc.write()
    pdf_doc.close()

    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_stream.name = "test_document.pdf"
    res_pdf = run_pipeline("What is the submission deadline?", pdf_stream, mode="generate")
    assert res_pdf["file_type"] == "pdf"
    assert len(res_pdf["retrieved_passages"]) > 0
    # Must preserve page number
    top_p = res_pdf["retrieved_passages"][0]
    assert top_p.get("page") in (1, 2)
    assert format_citation(top_p).startswith("Page ")

    # 2. Multi-paragraph Synthetic DOCX
    docx_doc = docx.Document()
    docx_doc.add_paragraph("Academic Attendance Policy 2026.")
    docx_doc.add_paragraph("Minimum mandatory attendance is 75 percent for all registered students.")
    docx_doc.add_paragraph("Medical certificates must be submitted within 3 working days.")
    docx_stream = io.BytesIO()
    docx_doc.save(docx_stream)
    docx_bytes = docx_stream.getvalue()

    docx_file = io.BytesIO(docx_bytes)
    docx_file.name = "policy.docx"
    res_docx = run_pipeline("What is the required attendance percentage?", docx_file, mode="generate")
    assert res_docx["file_type"] == "docx"
    assert len(res_docx["retrieved_passages"]) > 0
    top_d = res_docx["retrieved_passages"][0]
    assert top_d.get("paragraph") is not None
    assert format_citation(top_d).startswith("Paragraph ")

    # 3. TXT Format
    txt_path = Path("data/demo/application_notice.txt")
    res_txt = run_pipeline("What is the application fee?", txt_path, mode="generate")
    assert res_txt["file_type"] == "txt"
    assert len(res_txt["retrieved_passages"]) > 0
    top_t = res_txt["retrieved_passages"][0]
    assert top_t.get("paragraph") is not None
    assert format_citation(top_t).startswith("Paragraph ")


def test_document_review_answer_modes():
    """Verify behavior of generate, manual, and demo_bad answer modes."""
    clear_pipeline_cache()
    txt_path = Path("data/demo/application_notice.txt")

    # Mode 1: Generate (offline extractive fallback)
    res_gen = run_pipeline("What is the application fee?", txt_path, mode="generate")
    assert res_gen["mode"] == "generate"
    assert "500 rupees" in res_gen["raw_answer"]
    assert len(res_gen["claims"]) > 0

    # Mode 2: Manual (user pasted answer)
    pasted = "The candidate must pay 500 rupees online before November 15."
    res_man = run_pipeline("What is the fee and deadline?", txt_path, answer=pasted, mode="manual")
    assert res_man["mode"] == "manual"
    assert res_man["raw_answer"] == pasted
    assert len(res_man["claims"]) >= 1

    # Mode 3: Demo Bad (known-bad untrusted answer)
    res_bad = run_pipeline(
        "What is the application deadline?",
        txt_path,
        answer="The application deadline is 30 December 2099.",
        mode="demo_bad",
    )
    assert res_bad["mode"] == "demo_bad"
    assert any("Known-bad answer mode was selected" in w for w in res_bad["warnings"])


def test_offline_abstention_on_unrelated_query():
    """Verify that when no evidence is found, pipeline abstains without fabricating claims."""
    clear_pipeline_cache()
    txt_path = Path("data/demo/application_notice.txt")

    # Asking about quantum mechanics in an admissions circular
    res = run_pipeline("Explain quantum entanglement and quantum teleportation protocols.", txt_path, mode="generate")
    assert res["has_evidence"] is False
    assert res["answer_status"] == "no_evidence"
    assert len(res["claims"]) == 0
    assert len(res["verification_inputs"]) == 0
    assert any("Document evidence did not contain information addressing this question" in w for w in res["warnings"])


# ==============================================================================
# TASK 3: Real Model Integration on Synthetic Scenarios (No Mocks)
# ==============================================================================

@pytest.fixture(scope="module")
def real_verification_engine():
    """Initialize real DeBERTa-v3 VerificationEngine once for the test module."""
    t0 = time.perf_counter()
    engine = VerificationEngine()
    load_time = time.perf_counter() - t0
    # Yield engine and timing metadata
    engine._load_duration = load_time
    return engine


def test_real_model_scenario_1_numeric_contradiction(real_verification_engine):
    """Scenario 1: Numeric & date discrepancy -> status: contradicted, action: withhold."""
    clear_pipeline_cache()
    doc_path = Path("data/demo/application_notice.txt")
    question = "What is the application submission deadline and fee amount?"
    problematic_answer = "The application deadline is 30 November 2026 and the registration fee is 500 rupees."

    pipe_out = run_pipeline(question=question, doc=doc_path, answer=problematic_answer, mode="manual")
    assert len(pipe_out["verification_inputs"]) > 0

    t0 = time.perf_counter()
    verif_out = verify_pipeline_output(pipe_out, engine=real_verification_engine)
    duration = time.perf_counter() - t0

    results = verif_out["verification_results"]
    assert len(results) > 0

    # Deadline claim should be contradicted due to date discrepancy (15 Nov vs 30 Nov)
    deadline_results = [r for r in results if "november" in r.get("claim", "").lower() or "deadline" in r.get("claim", "").lower()]
    assert len(deadline_results) > 0
    deadline_res = deadline_results[0]
    assert deadline_res["status"] == "contradicted"
    assert deadline_res["action"] == "withhold"
    assert "deadline mismatch" in deadline_res.get("reason", "").lower()

    # Fee claim should be independently supported and stated
    fee_results = [r for r in results if "500" in r.get("claim", "")]
    assert len(fee_results) > 0
    fee_res = fee_results[0]
    assert fee_res["status"] == "supported"
    assert fee_res["action"] == "state"

    # Composed answer must withhold contradicted claim and state supported fee claim
    safe = compose_safe_answer(results)
    assert safe["verdict"] in ("partial_answer", "answer_with_caveats")
    assert "The registration fee is 500 rupees." in safe["answer"]
    withheld_texts = [w["claim"] for w in safe["withheld_claims"]]
    assert any("30 november" in w.lower() for w in withheld_texts)


def test_real_model_scenario_2_missing_evidence_abstention(real_verification_engine):
    """Scenario 2: Missing fact & extraneous claim -> status: insufficient, action: withhold."""
    clear_pipeline_cache()
    doc_path = Path("data/demo/attendance_policy.txt")
    question = "Does the attendance policy grant automatic exemptions for sports competitions?"
    problematic_answer = "Students participating in sports tournaments are automatically exempted from the 75% attendance rule without prior application."

    pipe_out = run_pipeline(question=question, doc=doc_path, answer=problematic_answer, mode="manual")
    assert len(pipe_out["verification_inputs"]) > 0

    t0 = time.perf_counter()
    verif_out = verify_pipeline_output(pipe_out, engine=real_verification_engine)
    duration = time.perf_counter() - t0

    results = verif_out["verification_results"]
    assert len(results) > 0
    sports_res = results[0]
    assert sports_res["status"] in ("insufficient", "contradicted")
    assert sports_res["action"] == "withhold"

    safe = compose_safe_answer(results)
    assert safe["verdict"] == "abstain"
    assert "There is not enough reliable document evidence" in safe["answer"]


def test_real_model_scenario_3_conflicting_regulatory_notices(real_verification_engine):
    """Scenario 3: Conflicting notices -> status: conflicting, action: caveat."""
    claim = "The application deadline is 15 November 2026."
    evidence = [
        {
            "chunk_id": "notice_a",
            "text": "Notice A: The application deadline is 15 November 2026.",
            "source": "Notice A.pdf",
            "page": 1,
        },
        {
            "chunk_id": "notice_b",
            "text": "Notice B: The application deadline is 20 November 2026.",
            "source": "Notice B.pdf",
            "page": 2,
        },
    ]

    t0 = time.perf_counter()
    conf_res = real_verification_engine.verify_claim(claim, evidence)
    duration = time.perf_counter() - t0

    assert conf_res["status"] == "conflicting"
    assert conf_res["action"] == "caveat"
    assert "disagree" in conf_res["reason"].lower()

    # Both notices must be preserved in the evidence array
    ev_texts = [e.get("text", "") for e in conf_res.get("evidence", [])]
    assert any("15 November" in t for t in ev_texts)
    assert any("20 November" in t for t in ev_texts)

    # Safe composer must caveat rather than suppress
    safe = compose_safe_answer([conf_res])
    assert safe["verdict"] == "answer_with_caveats"
    assert "The available document passages disagree about this point" in safe["answer"]


# ==============================================================================
# TASK 4: Safe Answer Composition Across All Decision Combinations
# ==============================================================================

def test_safe_composer_matrix_of_decisions():
    """Verify all policy decisions: state, caveat, withhold, and complete abstention."""
    # 1. Fully supported claim
    results_supported = [
        {"claim": "The registration fee is 500 rupees.", "status": "supported", "action": "state"}
    ]
    safe_sup = compose_safe_answer(results_supported)
    assert safe_sup["verdict"] == "answer"
    assert safe_sup["answer"] == "The registration fee is 500 rupees."
    assert len(safe_sup["stated_claims"]) == 1
    assert len(safe_sup["caveated_claims"]) == 0
    assert len(safe_sup["withheld_claims"]) == 0

    # 2. Conflicting claim
    results_conf = [
        {"claim": "The deadline is 15 November.", "status": "conflicting", "action": "caveat"}
    ]
    safe_conf = compose_safe_answer(results_conf)
    assert safe_conf["verdict"] == "answer_with_caveats"
    assert "disagree about this point" in safe_conf["answer"]
    assert len(safe_conf["caveated_claims"]) == 1

    # 3. Insufficient / Contradicted claim withheld
    results_withheld = [
        {"claim": "Sports exemption applies automatically.", "status": "insufficient", "action": "withhold"}
    ]
    safe_withheld = compose_safe_answer(results_withheld)
    assert safe_withheld["verdict"] == "abstain"
    assert "not enough reliable document evidence" in safe_withheld["answer"]
    assert len(safe_withheld["withheld_claims"]) == 1

    # 4. Mixed claims: 1 stated, 1 caveated, 1 withheld -> partial_answer_with_caveats
    results_mixed = [
        {"claim": "Fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "Deadline is 15 November.", "status": "conflicting", "action": "caveat"},
        {"claim": "Exam is optional.", "status": "contradicted", "action": "withhold"},
    ]
    safe_mixed = compose_safe_answer(results_mixed)
    assert safe_mixed["verdict"] == "partial_answer_with_caveats"
    assert "Fee is 500 rupees." in safe_mixed["answer"]
    assert "disagree about this point" in safe_mixed["answer"]
    assert "Exam is optional" not in safe_mixed["answer"]
    assert len(safe_mixed["stated_claims"]) == 1
    assert len(safe_mixed["caveated_claims"]) == 1
    assert len(safe_mixed["withheld_claims"]) == 1

    # 5. Stated + Caveated (no withheld) -> answer_with_caveats
    results_stated_cav = [
        {"claim": "Fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "Deadline is 15 November.", "status": "conflicting", "action": "caveat"},
    ]
    safe_sc = compose_safe_answer(results_stated_cav)
    assert safe_sc["verdict"] == "answer_with_caveats"

    # 6. Stated + Withheld (no caveats) -> partial_answer
    results_stated_wh = [
        {"claim": "Fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "Exam is optional.", "status": "contradicted", "action": "withhold"},
    ]
    safe_sw = compose_safe_answer(results_stated_wh)
    assert safe_sw["verdict"] == "partial_answer"


# ==============================================================================
# TASK 5: Audit CSV Export Formatting & RFC 4180 Compliance
# ==============================================================================

def test_audit_csv_export_format_and_escaping():
    """Verify CSV export formatting, RFC 4180 compliance, and secret protection."""
    results = [
        {
            "claim_id": "C1",
            "claim": 'The fee is 500 "rupees", payable online.',
            "status": "supported",
            "action": "state",
            "confidence": 0.9854,
            "reason": "Direct numeric match, no conflicts detected.",
            "evidence": [
                {
                    "source": "Notice A.pdf",
                    "page": 1,
                    "text": 'Notice A states: "Fee is 500 rupees"\nLine 2 text.',
                }
            ],
        },
        {
            "claim_id": "C2",
            "claim": "Exam date is June 1st.",
            "status": "insufficient",
            "action": "withhold",
            "confidence": 0.12,
            "reason": "Missing grounding.",
            "evidence": [],
        },
    ]

    csv_bytes = generate_audit_csv(results)
    assert isinstance(csv_bytes, bytes)
    # UTF-8 BOM
    csv_text = csv_bytes.decode("utf-8-sig")

    # Must be valid parseable CSV
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 2

    r1 = rows[0]
    assert r1["claim_id"] == "C1"
    assert r1["status"] == "supported"
    assert r1["action"] == "state"
    assert r1["confidence"] == "0.9854"
    assert r1["evidence_sources"] == "Notice A.pdf"
    assert r1["evidence_citations"] == "Page 1"
    assert r1["evidence_pages"] == "1"

    # Verify no accidental secret leakage
    forbidden_tokens = ["api_key", "secret", "password", "token", "Bearer", "BEGIN PRIVATE KEY"]
    for token in forbidden_tokens:
        assert token.lower() not in csv_text.lower()


# ==============================================================================
# TASK 7: Session State, Cache Keys & Error Boundaries
# ==============================================================================

def test_empty_evidence_verifier_boundary(real_verification_engine):
    """VerificationEngine must handle empty evidence gracefully without raising."""
    res = real_verification_engine.verify_claim("Arbitrary claim.", [])
    assert res["status"] == "insufficient"
    assert res["action"] == "withhold"
    assert res["confidence"] == 0.0


def test_file_size_rejection_boundary():
    """Documents exceeding max_upload_size_bytes must be rejected immediately."""
    oversized = io.BytesIO(b"0" * (SETTINGS.max_upload_size_bytes + 1024))
    oversized.name = "giant_doc.txt"
    with pytest.raises(DocumentIngestionError, match="exceeds maximum allowed file size"):
        load_document(oversized)


def test_invalid_file_extension_rejection():
    """Files with unsupported extensions must raise a clear DocumentIngestionError."""
    bad_file = io.BytesIO(b"binary data")
    bad_file.name = "unsupported.exe"
    with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
        load_document(bad_file)
