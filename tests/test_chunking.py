from src.chunking import chunk_text, chunk_document


def test_chunking_keeps_overlap_and_size():
    words = [f"word{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), chunk_size=120, overlap=30, source="x.txt")
    assert len(chunks) == 3
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-30:] == second[:30]
    assert chunks[0]["word_count"] == 120


def test_document_chunks_keep_page_and_source():
    sections = [
        {"source": "notice.pdf", "page": 2, "text": "one two three four five"},
        {"source": "notice.pdf", "page": 3, "text": "six seven eight nine ten"},
    ]
    chunks = chunk_document(sections, chunk_size=4, overlap=1)
    assert {item["page"] for item in chunks} == {2, 3}
    assert all(item["source"] == "notice.pdf" for item in chunks)
