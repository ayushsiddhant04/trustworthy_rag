from io import BytesIO
import pytest
from unittest.mock import MagicMock

from src.config import SETTINGS
from src.ingestion import load_document, DocumentIngestionError
from src.chunking import chunk_document
from src.pipeline import run
from src.verification_adapter import verify_pipeline_output
from src.safe_composer import compose_safe_answer
from member_b.decision_policy import decide
from member_b.verifier import VerificationEngine


class NamedBytesIO(BytesIO):
    def __init__(self, payload, name):
        super().__init__(payload)
        self.name = name


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class FakeNLI:
    def predict_pairs(self, pairs):
        output = []
        for evidence, claim in pairs:
            e = evidence.lower()
            c = claim.lower()
            # If evidence contradicts claim on date
            if "20 november" in e and "15 november" in c:
                scores = {"entailment": 0.04, "contradiction": 0.92, "neutral": 0.04}
            elif "15 november" in e and "20 november" in c:
                scores = {"entailment": 0.04, "contradiction": 0.92, "neutral": 0.04}
            elif "15 november" in e and "15 november" in c:
                scores = {"entailment": 0.92, "contradiction": 0.04, "neutral": 0.04}
            elif "20 november" in e and "20 november" in c:
                scores = {"entailment": 0.92, "contradiction": 0.04, "neutral": 0.04}
            else:
                scores = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}
            output.append({"scores": scores, "logits": [0.0, 0.0, 0.0]})
        return output


def _make_engine(monkeypatch):
    import member_b.verifier as verifier_mod
    monkeypatch.setattr(verifier_mod, "load_tokenizer", lambda model_name: FakeTokenizer())
    return VerificationEngine(nli=FakeNLI())


# 1. Offline fallback does not generate a meta-claim
def test_offline_fallback_does_not_generate_meta_claim():
    payload = "Students must maintain 75 percent attendance to sit for examinations.".encode()
    # Question with no lexical overlap in document
    result = run(
        question="What is the speed of light in vacuum?",
        doc=NamedBytesIO(payload, "policy.txt"),
        mode="generate",
    )
    assert result["answer_status"] == "no_evidence"
    assert result["has_evidence"] is False
    assert result["claims"] == []
    assert result["verification_inputs"] == []
    assert "not contain enough information" in result["raw_answer"]


# 2. No-evidence output produces abstention
def test_no_evidence_output_produces_abstention():
    payload = "Students must maintain 75 percent attendance to sit for examinations.".encode()
    result = run(
        question="What is the astrophysics redshift?",
        doc=NamedBytesIO(payload, "policy.txt"),
        mode="generate",
    )
    # Verification of empty inputs produces abstention
    verified = verify_pipeline_output(result, engine=MagicMock())
    assert verified["overall_verdict"] == "abstain"
    assert verified["verification_results"] == []
    safe = compose_safe_answer(verified["verification_results"])
    assert safe["verdict"] == "abstain"
    assert "not enough reliable" in safe["answer"]


# 3. Genuine conflicting notices produce caveat
def test_genuine_conflicting_notices_produce_caveat(monkeypatch):
    engine = _make_engine(monkeypatch)
    result = engine.verify_claim(
        "The application deadline is 15 November.",
        [
            {"chunk_id": "a", "text": "The application deadline is 15 November.", "page": 1, "source": "notice_a.pdf"},
            {"chunk_id": "b", "text": "The application deadline is 20 November.", "page": 2, "source": "notice_b.pdf"},
        ],
    )
    assert result["status"] == "conflicting"
    assert result["action"] == "caveat"
    assert "disagree" in result["reason"].lower()
    # Both supporting and contradicting evidence windows preserved
    relations = {e["relation"] for e in result["evidence"]}
    assert "entailment" in relations
    assert "contradiction" in relations


# 4. Single-source date mismatch is withheld
def test_single_source_date_mismatch_is_withheld(monkeypatch):
    engine = _make_engine(monkeypatch)
    result = engine.verify_claim(
        "The application deadline is 20 November.",
        [
            {"chunk_id": "a", "text": "The application deadline is 15 November.", "page": 1, "source": "notice_a.pdf"},
        ],
    )
    assert result["status"] == "contradicted"
    assert result["action"] == "withhold"
    assert result["numeric_check"]["numeric_conflict"] is True


# 5. Relevance gating disabled preserves baseline
def test_relevance_gating_disabled_preserves_baseline():
    fake_engine = MagicMock()
    fake_engine.verify_claim.return_value = {
        "status": "supported", "confidence": 0.9, "action": "state",
        "reason": "Supported", "evidence": [],
    }
    output = {
        "verification_inputs": [
            {
                "claim_id": "C1",
                "claim": "The deadline is 15 November.",
                "evidence": [
                    {"chunk_id": "c1", "text": "The deadline is 15 November.", "page": 1, "source": "notice.pdf"},
                    {"chunk_id": "c2", "text": "Cafeteria serves lunch at noon.", "page": 2, "source": "notice.pdf"},
                ],
            }
        ]
    }
    verified = verify_pipeline_output(output, engine=fake_engine, enable_relevance_gate=False)
    res = verified["verification_results"][0]
    assert res["relevance_filter_applied"] is False
    assert res["filtered_evidence"] == []
    assert len(res["candidate_evidence"]) == 2
    # Baseline passed all 2 passages to verifier
    assert len(fake_engine.verify_claim.call_args[0][1]) == 2


# 6. Relevance gating enabled records filtered evidence
def test_relevance_gating_enabled_records_filtered_evidence():
    fake_engine = MagicMock()
    fake_engine.verify_claim.return_value = {
        "status": "supported", "confidence": 0.9, "action": "state",
        "reason": "Supported", "evidence": [],
    }
    class MockRelevanceChecker:
        def check(self, claim, evidence):
            if "cafeteria" in evidence.lower():
                return {"similarity": 0.08, "is_relevant": False}
            return {"similarity": 0.88, "is_relevant": True}

    output = {
        "verification_inputs": [
            {
                "claim_id": "C1",
                "claim": "The deadline is 15 November.",
                "evidence": [
                    {"chunk_id": "c1", "text": "The deadline is 15 November.", "page": 1, "source": "notice.pdf"},
                    {"chunk_id": "c2", "text": "Cafeteria serves lunch at noon.", "page": 2, "source": "notice.pdf"},
                ],
            }
        ]
    }
    verified = verify_pipeline_output(
        output,
        engine=fake_engine,
        enable_relevance_gate=True,
        relevance_checker=MockRelevanceChecker(),
    )
    res = verified["verification_results"][0]
    assert res["relevance_filter_applied"] is True
    assert len(res["filtered_evidence"]) == 1
    assert res["filtered_evidence"][0]["chunk_id"] == "c2"
    assert len(res["candidate_evidence"]) == 2
    # Only relevant passage passed to verifier
    passed_to_verifier = fake_engine.verify_claim.call_args[0][1]
    assert len(passed_to_verifier) == 1
    assert passed_to_verifier[0]["chunk_id"] == "c1"


# 7. Conflicting evidence is not silently hidden
def test_conflicting_evidence_is_not_silently_hidden(monkeypatch):
    engine = _make_engine(monkeypatch)
    class MockRelevanceChecker:
        def check(self, claim, evidence):
            # Both notices discuss deadlines, so both are semantically relevant
            return {"similarity": 0.85, "is_relevant": True}

    output = {
        "verification_inputs": [
            {
                "claim_id": "C1",
                "claim": "The application deadline is 15 November.",
                "evidence": [
                    {"chunk_id": "a", "text": "The application deadline is 15 November.", "page": 1, "source": "notice_a.pdf"},
                    {"chunk_id": "b", "text": "The application deadline is 20 November.", "page": 2, "source": "notice_b.pdf"},
                ],
            }
        ]
    }
    verified = verify_pipeline_output(
        output,
        engine=engine,
        enable_relevance_gate=True,
        relevance_checker=MockRelevanceChecker(),
    )
    res = verified["verification_results"][0]
    assert res["status"] == "conflicting"
    assert res["action"] == "caveat"
    assert len(res["filtered_evidence"]) == 0
    assert len(res["candidate_evidence"]) == 2


# 8. TXT paragraph references appear correctly
def test_txt_paragraph_references_appear_correctly():
    txt_content = b"Paragraph one.\n\nParagraph two with more details.\n\nParagraph three."
    doc = load_document(NamedBytesIO(txt_content, "notes.txt"))
    assert doc["file_type"] == "txt"
    chunks = chunk_document(doc["sections"], chunk_size=10, overlap=2)
    assert all(c["page"] is None for c in chunks)
    assert all(c["paragraph"] is not None for c in chunks)
    assert chunks[0]["source"] == "notes.txt"


# 9. DOCX paragraph references appear correctly
def test_docx_paragraph_references_appear_correctly():
    from docx import Document
    buf = BytesIO()
    d = Document()
    d.add_paragraph("First paragraph of notice.")
    d.add_paragraph("Second paragraph with important policy details.")
    d.save(buf)

    doc = load_document(NamedBytesIO(buf.getvalue(), "doc_notice.docx"))
    assert doc["file_type"] == "docx"
    chunks = chunk_document(doc["sections"], chunk_size=10, overlap=2)
    assert all(c["page"] is None for c in chunks)
    assert all(c["paragraph"] is not None for c in chunks)
    assert chunks[0]["source"] == "doc_notice.docx"


# 10. File size limits are enforced
def test_file_size_limits_are_enforced(monkeypatch):
    import dataclasses
    import src.ingestion as ing_mod
    mock_settings = dataclasses.replace(SETTINGS, max_upload_size_bytes=100)
    monkeypatch.setattr(ing_mod, "SETTINGS", mock_settings)
    oversized = b"A" * 150
    with pytest.raises(DocumentIngestionError, match="exceeds maximum allowed file size"):
        load_document(NamedBytesIO(oversized, "big.txt"))


# 11. PDF page limits are enforced
def test_pdf_page_limits_are_enforced(monkeypatch):
    import fitz
    import dataclasses
    import src.ingestion as ing_mod
    mock_settings = dataclasses.replace(SETTINGS, max_pdf_pages=2)
    monkeypatch.setattr(ing_mod, "SETTINGS", mock_settings)
    pdf = fitz.open()
    for _ in range(4):
        p = pdf.new_page()
        p.insert_text((50, 50), "Sample page text")
    data = pdf.tobytes()
    pdf.close()

    with pytest.raises(DocumentIngestionError, match="exceeds the maximum allowed limit"):
        load_document(NamedBytesIO(data, "large.pdf"))


# 12. Dynamic upload limit resolution for Vercel and local
def test_resolve_upload_limit_bytes(monkeypatch):
    from src.config import _resolve_upload_limit_bytes

    # Default should be 25 MB
    monkeypatch.delenv("MAX_UPLOAD_SIZE_BYTES", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_SIZE_MB", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("DEPLOYMENT_TARGET", raising=False)
    assert _resolve_upload_limit_bytes() == 25 * 1024 * 1024

    # Vercel flag should resolve to 4 MB (Vercel 4.5 MB request ceiling)
    monkeypatch.setenv("VERCEL", "1")
    assert _resolve_upload_limit_bytes() == 4 * 1024 * 1024

    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("DEPLOYMENT_TARGET", "vercel")
    assert _resolve_upload_limit_bytes() == 4 * 1024 * 1024

    # Explicit MB override takes precedence
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")
    assert _resolve_upload_limit_bytes() == 10 * 1024 * 1024

    # Explicit Byte override takes highest precedence
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "2048")
    assert _resolve_upload_limit_bytes() == 2048

