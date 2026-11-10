from io import BytesIO
import pytest

from src.ingestion import DocumentIngestionError, load_document


class NamedBytesIO(BytesIO):
    def __init__(self, payload, name):
        super().__init__(payload)
        self.name = name


def test_txt_ingestion_preserves_paragraphs():
    doc = load_document(NamedBytesIO(b"First paragraph.\n\nSecond paragraph.", "policy.txt"))
    assert doc["file_type"] == "txt"
    assert len(doc["sections"]) == 2
    assert doc["sections"][1]["paragraph"] == 2


def test_docx_ingestion_reads_paragraphs_and_tables():
    from docx import Document
    from io import BytesIO as Buffer
    memory = Buffer()
    d = Document()
    d.add_paragraph("The deadline is 15 November.")
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Fee"
    table.cell(0, 1).text = "500 rupees"
    d.save(memory)
    result = load_document(NamedBytesIO(memory.getvalue(), "notice.docx"))
    assert result["file_type"] == "docx"
    assert any("deadline" in item["text"].lower() for item in result["sections"])
    assert any("500 rupees" in item["text"] for item in result["sections"])


def test_scanned_or_empty_pdf_has_clear_error():
    import fitz
    pdf = fitz.open()
    pdf.new_page()
    payload = pdf.tobytes()
    pdf.close()
    with pytest.raises(DocumentIngestionError, match="scanned/image-only"):
        load_document(NamedBytesIO(payload, "scan.pdf"))


def test_pdf_text_extraction_has_one_based_page_numbers():
    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "The application deadline is 15 November.")
    payload = pdf.tobytes()
    pdf.close()
    result = load_document(NamedBytesIO(payload, "notice.pdf"))
    assert result["sections"][0]["page"] == 1
