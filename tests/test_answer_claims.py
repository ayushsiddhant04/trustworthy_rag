import pytest
from src.answer_generation import generate_answer
from src.claim_extraction import extract_claims, _parse_json_claims


def test_manual_mode_returns_pasted_answer_verbatim():
    text = "The deadline is 15 November."
    assert generate_answer("When?", [], answer=text, mode="manual") == text


def test_demo_bad_requires_explicit_answer():
    with pytest.raises(ValueError, match="deliberately incorrect"):
        generate_answer("When?", [], mode="demo_bad")


def test_offline_generation_selects_source_sentence():
    answer = generate_answer(
        "What is the attendance requirement?",
        [{"text": "Students must maintain 75 percent attendance to sit the examination."}],
        mode="generate",
    )
    assert "75 percent" in answer


def test_claim_fallback_splits_sentences():
    claims = extract_claims("The deadline is 15 November. The fee is 500 rupees.", use_llm=False)
    assert len(claims) == 2


def test_safe_json_parser_handles_fenced_output_and_invalid_output():
    assert _parse_json_claims('```json\n["The deadline is 15 November."]\n```') == ["The deadline is 15 November."]
    assert _parse_json_claims("not json") == []
