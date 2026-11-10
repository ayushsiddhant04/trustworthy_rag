from src.retrieval import retrieve_with_meta


def test_tfidf_returns_top_three_with_provenance():
    chunks = [
        {"chunk_id": "a", "text": "The application deadline is 15 November.", "page": 1, "source": "n.pdf"},
        {"chunk_id": "b", "text": "Students must maintain 75 percent attendance.", "page": 2, "source": "n.pdf"},
        {"chunk_id": "c", "text": "The application fee is 500 rupees.", "page": 3, "source": "n.pdf"},
        {"chunk_id": "d", "text": "The library closes at six in the evening.", "page": 4, "source": "n.pdf"},
    ]
    result = retrieve_with_meta("application deadline", chunks, top_k=3, use_embeddings=False)
    assert len(result) == 3
    assert result[0]["chunk_id"] == "a"
    assert "score" in result[0] and "tfidf_score" in result[0]
    assert result[0]["page"] == 1
