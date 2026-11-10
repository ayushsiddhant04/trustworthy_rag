# Member B imports CrossEncoder at module import time. Stub only that symbol so
# adapter behaviour can be unit-tested without downloading heavy model packages.
import sys
import types

try:
    import sentence_transformers
except ImportError:
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    class _UnusedCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Real CrossEncoder is intentionally not used in unit tests")
    class _UnusedSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    fake_sentence_transformers.CrossEncoder = _UnusedCrossEncoder
    fake_sentence_transformers.SentenceTransformer = _UnusedSentenceTransformer
    fake_util = types.ModuleType("sentence_transformers.util")
    fake_util.cos_sim = lambda a, b: 0.0
    fake_sentence_transformers.util = fake_util
    sys.modules["sentence_transformers"] = fake_sentence_transformers
    sys.modules["sentence_transformers.util"] = fake_util

from member_b import verifier as verifier_module
from member_b.verifier import VerificationEngine


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
            if "20 november" in e and "15 november" in c:
                scores = {"entailment": 0.04, "contradiction": 0.92, "neutral": 0.04}
            elif any(value in e for value in ["15 november", "75 percent", "500 rupees"]) and any(value in c for value in ["15 november", "75 percent", "500 rupees"]):
                scores = {"entailment": 0.92, "contradiction": 0.04, "neutral": 0.04}
            else:
                scores = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}
            output.append({"scores": scores, "logits": [0.0, 0.0, 0.0]})
        return output


def _engine(monkeypatch):
    monkeypatch.setattr(verifier_module, "load_tokenizer", lambda model_name: FakeTokenizer())
    return VerificationEngine(nli=FakeNLI())


def test_member_b_verifier_accepts_member_a_evidence_shape(monkeypatch):
    engine = _engine(monkeypatch)
    result = engine.verify_claim(
        "The application deadline is 15 November.",
        [{"chunk_id": "c1", "text": "The application deadline is 15 November.", "page": 1, "source": "notice.pdf"}],
    )
    assert result["status"] == "supported"
    assert result["action"] == "state"
    assert result["evidence"][0]["page"] == 1


def test_member_b_detects_conflicting_passages(monkeypatch):
    engine = _engine(monkeypatch)
    result = engine.verify_claim(
        "The application deadline is 15 November.",
        [
            {"chunk_id": "a", "text": "The application deadline is 15 November.", "page": 1, "source": "notice_a.pdf"},
            {"chunk_id": "b", "text": "The application deadline is 20 November.", "page": 2, "source": "notice_b.pdf"},
        ],
    )
    assert result["status"] == "conflicting"
    # When multiple notices genuinely disagree, the policy produces caveat
    # so the conflicting schedules can be displayed rather than suppressed.
    assert result["action"] == "caveat"
