"""Tests confirming Configuration A and Configuration B differ strictly by numeric/date checking (Task 7).

Verifies:
1. Configuration A (enable_numeric_checks=False) and Configuration B (enable_numeric_checks=True)
   share identical model architecture, tokenizer, thresholds, windowing, and calibration temperature.
2. Only the numeric/date checking rule varies between configurations.
3. Behavior on controlled cases matches the fair ablation specifications.
4. Fair ablation summary artifacts exist and are internally consistent.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from member_b.verifier import VerificationEngine
from member_b.calibration import load_temperature
from member_b.numeric_checks import check_numeric_conflict


ROOT = Path(__file__).resolve().parent.parent


def test_configurations_differ_strictly_by_numeric_rule():
    """Verify Configuration A and B share identical hyperparameters except enable_numeric_checks."""
    calib_artifact_path = ROOT / "member_b/results/calibration_canonical_artifact.json"
    assert calib_artifact_path.is_file(), "Calibration artifact missing"
    artifact = load_temperature(calib_artifact_path)
    temp = artifact.temperature

    eng_a = VerificationEngine(temperature=temp, enable_numeric_checks=False)
    eng_b = VerificationEngine(temperature=temp, enable_numeric_checks=True)

    assert eng_a.enable_numeric_checks is False
    assert eng_b.enable_numeric_checks is True

    # Verify identical architecture and thresholds
    assert eng_a.model_name == eng_b.model_name
    assert eng_a.temperature == eng_b.temperature
    assert eng_a.entailment_threshold == eng_b.entailment_threshold
    assert eng_a.contradiction_threshold == eng_b.contradiction_threshold
    assert eng_a.window_size == eng_b.window_size
    assert eng_a.overlap == eng_b.overlap


def test_controlled_numeric_contradiction_ablation_behavior():
    """Verify that on an explicit numeric mismatch, Config B detects conflict while Config A disables checking."""
    calib_artifact_path = ROOT / "member_b/results/calibration_canonical_artifact.json"
    artifact = load_temperature(calib_artifact_path)
    temp = artifact.temperature

    eng_a = VerificationEngine(temperature=temp, enable_numeric_checks=False)
    eng_b = VerificationEngine(temperature=temp, enable_numeric_checks=True)

    claim = "The non-refundable application processing fee is 1500 rupees."
    evidence = [
        {"chunk_id": "p1", "text": "Candidates must pay an application fee of 750 rupees through the gateway.", "source": "test.pdf", "page": 1}
    ]

    res_a = eng_a.verify_claim(claim, evidence)
    res_b = eng_b.verify_claim(claim, evidence)

    # Config A has numeric checks disabled
    assert res_a["numeric_check"]["numeric_conflict"] is False
    assert res_a["numeric_check"]["reason"] == "Numeric checks disabled"

    # Config B has numeric checks enabled and detects the fee mismatch (1500 vs 750)
    assert res_b["numeric_check"]["numeric_conflict"] is True
    assert "1500" in res_b["numeric_check"]["reason"] and "750" in res_b["numeric_check"]["reason"]
    assert res_b["action"] == "withhold"


def test_qualitative_claim_identical_behavior():
    """Verify that on qualitative non-numeric claims, both configurations produce identical results."""
    calib_artifact_path = ROOT / "member_b/results/calibration_canonical_artifact.json"
    artifact = load_temperature(calib_artifact_path)
    temp = artifact.temperature

    eng_a = VerificationEngine(temperature=temp, enable_numeric_checks=False)
    eng_b = VerificationEngine(temperature=temp, enable_numeric_checks=True)

    claim = "Applicants must hold an accredited bachelor degree with first class honours."
    evidence = [
        {"chunk_id": "p1", "text": "Eligibility clause: Candidates are required to hold an accredited bachelor degree with first class honours.", "source": "test.pdf", "page": 1}
    ]

    res_a = eng_a.verify_claim(claim, evidence)
    res_b = eng_b.verify_claim(claim, evidence)

    assert res_a["status"] == res_b["status"]
    assert res_a["action"] == res_b["action"]
    assert abs(res_a["confidence"] - res_b["confidence"]) < 1e-6


def test_fair_ablation_summary_artifact():
    """Verify the persisted fair ablation summary exists and contains valid metrics."""
    summary_path = ROOT / "member_b/results/fair_numeric_ablation_summary.csv"
    assert summary_path.is_file(), "Fair ablation summary CSV missing"

    import pandas as pd
    df = pd.read_csv(summary_path)
    assert len(df) >= 10
    assert "Metric" in df.columns
    assert "Config A (No Numeric Checks)" in df.columns
    assert "Config B (Numeric Checks Enabled)" in df.columns
    assert "Delta (B - A)" in df.columns


def test_controlled_cases_artifact():
    """Verify the controlled cases evaluation JSON exists and covers all 8 scenarios."""
    cases_path = ROOT / "member_b/results/controlled_cases_evaluation.json"
    assert cases_path.is_file(), "Controlled cases JSON missing"

    data = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(data) == 8
    case_ids = [c["case_id"] for c in data]
    assert "CASE_1_INCORRECT_DEADLINE" in case_ids
    assert "CASE_2_INCORRECT_FEE" in case_ids
    assert "CASE_3_COMPOUND_CLAIM" in case_ids
    assert "CASE_4_EQUIVALENT_DATES" in case_ids
    assert "CASE_5_UNRELATED_FIELDS" in case_ids
    assert "CASE_6_CONFLICTING_NOTICES" in case_ids
    assert "CASE_7_YEAR_NOT_A_FEE" in case_ids
    assert "CASE_8_NO_NUMERICAL_INFO" in case_ids

