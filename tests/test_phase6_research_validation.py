"""Phase 6: Automated Research Validation & Canonical Benchmark Tests.

Validates:
1. Canonical dataset schema, fields, and non-empty values.
2. 4-class balance across categories, including >= 30 conflicting examples.
3. Grounded passage relationships (entails, contradicts, neutral, irrelevant).
4. Strict document-level split independence (zero leakage across calibration, validation, and test splits).
5. Complete isolation from demo documents (data/demo/).
6. Temperature calibration artifact validity (ECE, Brier, NLL metrics recorded).
7. Decision policy and application verdict taxonomy consistency across all 5 states.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.safe_composer import compose_safe_answer
from member_b.benchmark import parse_benchmark_dataset, compute_evaluation_metrics, LABELS
from member_b.calibration import load_temperature, compute_ece, compute_brier_score, compute_nll
from member_b.decision_policy import decide


ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# TASK 2 & 3: CANONICAL DATASET SCHEMA & BALANCE TESTS
# ==============================================================================

def test_canonical_dataset_schema_and_integrity():
    """Verify canonical benchmark dataset conforms strictly to the unified research schema."""
    canon_path = ROOT / "member_b" / "data" / "canonical_dataset_full.csv"
    assert canon_path.is_file(), f"Canonical dataset not found at {canon_path}"

    df = pd.read_csv(canon_path)
    assert len(df) >= 150, f"Expected at least 150 examples, found {len(df)}"

    required_cols = [
        "example_id",
        "document_id",
        "source_id",
        "claim",
        "true_status",
        "evidence_passages",
        "annotation_notes",
        "dataset_split",
        "dataset_version",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        # No empty cells in required columns
        assert df[col].notna().all(), f"Found null values in column {col}"
        assert (df[col].astype(str).str.strip() != "").all(), f"Found blank string in column {col}"

    # Verify example_id uniqueness
    assert df["example_id"].is_unique, "Duplicate example_id found in canonical dataset"

    # Verify claim text uniqueness within dataset
    assert df["claim"].is_unique, "Duplicate claim text found in canonical dataset"


def test_canonical_class_distribution_and_conflicts():
    """Verify all 4 classes are represented with at least 30 conflicting examples."""
    canon_path = ROOT / "member_b" / "data" / "canonical_dataset_full.csv"
    df = pd.read_csv(canon_path)

    counts = df["true_status"].value_counts().to_dict()
    for label in ("supported", "contradicted", "insufficient", "conflicting"):
        assert label in counts, f"Class {label} missing from canonical dataset"
        assert counts[label] >= 30, f"Class {label} has {counts[label]} examples, expected >= 30"

    assert counts["conflicting"] >= 30, f"Expected at least 30 conflicting examples, found {counts['conflicting']}"


def test_passage_level_relationships_and_grounding():
    """Verify evidence_passages contains valid JSON with passage-level relationship tags."""
    canon_path = ROOT / "member_b" / "data" / "canonical_dataset_full.csv"
    df = pd.read_csv(canon_path)

    valid_relationships = {"entails", "contradicts", "neutral", "irrelevant"}

    for _, row in df.iterrows():
        raw_passages = row["evidence_passages"]
        passages = json.loads(raw_passages)
        assert isinstance(passages, list), f"evidence_passages is not a list for {row['example_id']}"

        status = row["true_status"]
        if status in ("supported", "contradicted", "conflicting"):
            assert len(passages) >= 1, f"Expected evidence passages for {row['example_id']} ({status})"

        if status == "conflicting":
            # Must associate multiple passages containing both positive and negative/divergent evidence
            assert len(passages) >= 2, f"Conflicting example {row['example_id']} must have >= 2 passages"
            rels = {p.get("relationship") for p in passages}
            assert "entails" in rels and "contradicts" in rels, (
                f"Conflicting example {row['example_id']} must contain both entails and contradicts passages, got {rels}"
            )

        for p in passages:
            assert "chunk_id" in p and p["chunk_id"].strip(), f"Missing chunk_id in passage for {row['example_id']}"
            assert "text" in p and p["text"].strip(), f"Missing text in passage for {row['example_id']}"
            assert "source" in p and p["source"].strip(), f"Missing source in passage for {row['example_id']}"
            rel = p.get("relationship")
            assert rel in valid_relationships, (
                f"Invalid passage relationship '{rel}' in {row['example_id']}. Expected one of {valid_relationships}"
            )


# ==============================================================================
# TASK 5: STRICT GROUPED SPLITTING & ZERO-LEAKAGE TESTS
# ==============================================================================

def test_split_independence_and_zero_leakage():
    """Verify calibration, validation, and test splits have ZERO document or claim leakage."""
    calib_path = ROOT / "member_b" / "data" / "calibration_split.csv"
    val_path = ROOT / "member_b" / "data" / "validation_split.csv"
    test_path = ROOT / "member_b" / "data" / "test_split.csv"

    assert calib_path.is_file()
    assert val_path.is_file()
    assert test_path.is_file()

    calib_df = pd.read_csv(calib_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    calib_docs = set(calib_df["document_id"])
    val_docs = set(val_df["document_id"])
    test_docs = set(test_df["document_id"])

    # Document-level disjointness
    assert len(calib_docs & val_docs) == 0, f"Document leakage between calib and val: {calib_docs & val_docs}"
    assert len(calib_docs & test_docs) == 0, f"Document leakage between calib and test: {calib_docs & test_docs}"
    assert len(val_docs & test_docs) == 0, f"Document leakage between val and test: {val_docs & test_docs}"

    # Claim-level disjointness
    calib_claims = set(calib_df["claim"])
    val_claims = set(val_df["claim"])
    test_claims = set(test_df["claim"])

    assert len(calib_claims & val_claims) == 0, "Claim text leakage between calib and val"
    assert len(calib_claims & test_claims) == 0, "Claim text leakage between calib and test"
    assert len(val_claims & test_claims) == 0, "Claim text leakage between val and test"

    # Held-out test set size
    assert len(test_df) >= 100, f"Held-out test set has {len(test_df)} examples, expected >= 100"


def test_isolation_from_demo_documents():
    """Verify canonical datasets do not borrow or reuse demo directory files."""
    canon_path = ROOT / "member_b" / "data" / "canonical_dataset_full.csv"
    df = pd.read_csv(canon_path)

    demo_sources = {
        "university_regulations_notice_a.txt",
        "university_regulations_notice_b.txt",
        "leave_policy_amendment_2024.txt",
        "sample_contradiction.txt",
    }
    canon_sources = set(df["source_id"].unique())
    leakage = demo_sources & canon_sources
    assert len(leakage) == 0, f"Demo files leaked into canonical benchmark: {leakage}"


# ==============================================================================
# TASK 7: TEMPERATURE CALIBRATION TESTS
# ==============================================================================

def test_temperature_calibration_artifact():
    """Verify saved calibration artifact has valid temperature and metrics."""
    artifact_path = ROOT / "member_b" / "results" / "calibration_canonical_artifact.json"
    if not artifact_path.is_file():
        pytest.skip("Calibration artifact not generated yet.")

    artifact = load_temperature(artifact_path)
    assert artifact.temperature > 0, f"Invalid temperature: {artifact.temperature}"
    assert artifact.is_calibrated is True

    calib_meta = artifact.metadata.get("calibration_set", {})
    assert "ece_uncalibrated" in calib_meta
    assert "ece_calibrated" in calib_meta
    assert "brier_uncalibrated" in calib_meta
    assert "brier_calibrated" in calib_meta
    assert "nll_uncalibrated" in calib_meta
    assert "nll_calibrated" in calib_meta

    # Verify that calibration was fit to reduce NLL or ECE on calibration set
    assert calib_meta["nll_calibrated"] <= calib_meta["nll_uncalibrated"] + 1e-4, (
        f"Calibration failed to minimize NLL: {calib_meta['nll_uncalibrated']} -> {calib_meta['nll_calibrated']}"
    )


# ==============================================================================
# TASK 9: DECISION POLICY & APPLICATION VERDICT TAXONOMY CONSISTENCY
# ==============================================================================

def test_decision_policy_mappings():
    """Verify atomic decide() mappings for all 4 statuses."""
    # Supported -> state
    assert decide("supported", 0.95)["action"] == "state"
    # Contradicted -> withhold
    assert decide("contradicted", 0.95)["action"] == "withhold"
    # Insufficient -> withhold
    assert decide("insufficient", 0.50)["action"] == "withhold"
    # Conflicting -> caveat
    assert decide("conflicting", 0.85)["action"] == "caveat"
    # Numeric conflict override
    assert decide("supported", 0.95, numeric_conflict=True)["action"] == "withhold"


def test_safe_answer_five_verdict_states():
    """Verify all 5 application-level verdict states defined in Task 9."""
    # 1. 'answer': Supported only
    v1 = compose_safe_answer([
        {"claim": "The registration fee is 500 rupees.", "status": "supported", "action": "state"}
    ])
    assert v1["verdict"] == "answer"
    assert "500 rupees" in v1["answer"]

    # 2. 'answer_with_caveats': Supported + Caveated (no withheld)
    v2 = compose_safe_answer([
        {"claim": "The registration fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "The deadline is 15 November.", "status": "conflicting", "action": "caveat"},
    ])
    assert v2["verdict"] == "answer_with_caveats"
    assert "500 rupees" in v2["answer"]
    assert "disagree about this point" in v2["answer"]

    # 3. 'partial_answer': Supported + Withheld (no caveats)
    v3 = compose_safe_answer([
        {"claim": "The registration fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "Sports exemption is automatic.", "status": "insufficient", "action": "withhold"},
    ])
    assert v3["verdict"] == "partial_answer"
    assert "500 rupees" in v3["answer"]
    assert "Sports exemption" not in v3["answer"]

    # 4. 'partial_answer_with_caveats': Supported + Caveated + Withheld
    v4 = compose_safe_answer([
        {"claim": "The registration fee is 500 rupees.", "status": "supported", "action": "state"},
        {"claim": "The deadline is 15 November.", "status": "conflicting", "action": "caveat"},
        {"claim": "Sports exemption is automatic.", "status": "insufficient", "action": "withhold"},
    ])
    assert v4["verdict"] == "partial_answer_with_caveats"
    assert "500 rupees" in v4["answer"]
    assert "disagree about this point" in v4["answer"]
    assert "Sports exemption" not in v4["answer"]

    # 5. 'abstain': All withheld
    v5 = compose_safe_answer([
        {"claim": "Sports exemption is automatic.", "status": "insufficient", "action": "withhold"},
        {"claim": "Fee is 1000 rupees.", "status": "contradicted", "action": "withhold"},
    ])
    assert v5["verdict"] == "abstain"
    assert "not enough reliable document evidence" in v5["answer"]

