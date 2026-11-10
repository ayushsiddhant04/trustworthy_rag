from io import BytesIO

from src.pipeline import run
from src.verification_adapter import verify_pipeline_output
from src.safe_composer import compose_safe_answer


class NamedBytesIO(BytesIO):
    def __init__(self, payload, name):
        super().__init__(payload)
        self.name = name


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_claim(self, claim, evidence):
        self.calls.append((claim, evidence))
        return {
            "claim": claim,
            "status": "supported",
            "confidence": 0.91,
            "action": "state",
            "reason": "Evidence supports this claim.",
            "evidence": [{"relation": "entailment", "text": evidence[0]["text"], "page": evidence[0]["page"], "source": evidence[0]["source"]}],
        }


def test_pipeline_returns_claim_specific_verification_contract():
    payload = (
        "Application Notice\n\nThe application deadline is 15 November 2026. "
        "Applications received after that date will not be considered."
    ).encode()
    result = run(
        "What is the application deadline?",
        NamedBytesIO(payload, "notice.txt"),
        answer="The application deadline is 15 November 2026.",
        mode="manual",
        top_k=3,
    )
    assert result["raw_answer"] == result["answer"]
    assert result["claims"]
    item = result["verification_inputs"][0]
    assert item["claim_id"] == "C1"
    assert item["evidence"]
    assert item["evidence"][0]["source"] == "notice.txt"
    assert item["passage_refs"]


def test_adapter_calls_verifier_once_per_claim_with_own_evidence():
    output = {
        "raw_answer": "Deadline is 15 November. Fee is 500 rupees.",
        "verification_inputs": [
            {"claim_id": "C1", "claim": "Deadline is 15 November.", "evidence": [{"chunk_id": "a", "text": "Deadline is 15 November.", "page": 1, "source": "n.pdf"}]},
            {"claim_id": "C2", "claim": "Fee is 500 rupees.", "evidence": [{"chunk_id": "b", "text": "Fee is 500 rupees.", "page": 2, "source": "n.pdf"}]},
        ],
    }
    fake = FakeVerifier()
    verified = verify_pipeline_output(output, engine=fake)
    assert len(fake.calls) == 2
    assert len(verified["verification_results"]) == 2
    assert verified["overall_verdict"] == "answer"


def test_safe_composer_withholds_unsupported_claims():
    result = compose_safe_answer([
        {"claim": "The deadline is 15 November.", "status": "supported", "action": "state"},
        {"claim": "An income certificate is mandatory.", "status": "insufficient", "action": "withhold"},
    ])
    assert "15 November" in result["answer"]
    assert "income certificate" not in result["answer"]
    assert len(result["withheld_claims"]) == 1


def test_safe_composer_abstains_when_no_claim_survives():
    result = compose_safe_answer([
        {"claim": "An income certificate is mandatory.", "status": "insufficient", "action": "withhold"}
    ])
    assert result["verdict"] == "abstain"
    assert "not enough reliable" in result["answer"]
