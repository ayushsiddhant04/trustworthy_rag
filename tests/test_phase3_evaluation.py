"""Unit and integration tests for Phase 3: Scientific Evaluation Framework.

Covers:
- Multi-passage benchmark dataset parsing, grouping, JSON support, and label validation
- Temperature fitting, ECE, Brier score, and NLL calibration metrics
- Calibration artifact saving, loading, invalid file handling, and default fallback
- Controlled baseline experiments runner (5 configurations)
- False acceptance rate, coverage, and abstention rate metrics
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from member_b.benchmark import (
    _parse_benchmark_evidence,
    parse_benchmark_dataset,
    compute_evaluation_metrics,
    run_benchmark,
    LABELS,
)
from member_b.calibration import (
    compute_ece,
    compute_brier_score,
    compute_nll,
    fit_temperature,
    apply_temperature,
    save_temperature,
    load_temperature,
    CalibrationArtifact,
)
from member_b.experiments import ExperimentRunner, build_comparison_dataframe
from member_b.verifier import VerificationEngine


# ---------------------------------------------------------------------------
# Task 1: Multi-Passage Benchmark Support Tests
# ---------------------------------------------------------------------------

def test_parse_benchmark_evidence_json():
    json_data = json.dumps([
        {"chunk_id": "c1", "text": "Notice A says deadline 15 Nov.", "source": "Notice A", "page": 1},
        {"chunk_id": "c2", "text": "Notice B says deadline 20 Nov.", "source": "Notice B", "page": 2},
    ])
    passages = _parse_benchmark_evidence(json_data)
    assert len(passages) == 2
    assert passages[0]["chunk_id"] == "c1"
    assert passages[0]["source"] == "Notice A"
    assert passages[0]["page"] == 1
    assert "Notice A says" in passages[0]["text"]
    assert passages[1]["chunk_id"] == "c2"


def test_parse_benchmark_evidence_delimited():
    delim_data = "First relevant passage || Second relevant passage"
    passages = _parse_benchmark_evidence(delim_data)
    assert len(passages) == 2
    assert passages[0]["text"] == "First relevant passage"
    assert passages[1]["text"] == "Second relevant passage"


def test_parse_benchmark_evidence_single_text_and_empty():
    single = _parse_benchmark_evidence("Single passage text")
    assert len(single) == 1
    assert single[0]["text"] == "Single passage text"

    empty_nan = _parse_benchmark_evidence(float("nan"))
    assert empty_nan == []

    empty_str = _parse_benchmark_evidence("")
    assert empty_str == []


def test_parse_benchmark_evidence_malformed_json_fallback():
    malformed = "[{\"text\": \"broken json, missing bracket"
    passages = _parse_benchmark_evidence(malformed)
    assert len(passages) == 1
    assert passages[0]["text"] == malformed


def test_parse_benchmark_dataset_groups_by_example_id():
    df = pd.DataFrame([
        {"example_id": "ex_1", "claim": "Fee is 5000", "evidence": "Passage 1", "true_label": "supported"},
        {"example_id": "ex_1", "claim": "Fee is 5000", "evidence": "Passage 2", "true_label": "supported"},
        {"example_id": "ex_2", "claim": "Exam date 10 Dec", "evidence": "Passage 3", "true_label": "contradicted"},
    ])
    examples = parse_benchmark_dataset(df)
    assert len(examples) == 2
    assert examples[0]["example_id"] == "ex_1"
    assert len(examples[0]["evidence_passages"]) == 2
    assert examples[1]["example_id"] == "ex_2"
    assert len(examples[1]["evidence_passages"]) == 1


def test_parse_benchmark_dataset_validates_labels():
    df = pd.DataFrame([
        {"claim": "Some claim", "evidence": "Some text", "true_label": "invalid_label"}
    ])
    with pytest.raises(ValueError, match="Unknown true_label"):
        parse_benchmark_dataset(df)


# ---------------------------------------------------------------------------
# Task 2: Calibration Tests
# ---------------------------------------------------------------------------

def test_calibration_metrics_ece_and_brier():
    # Perfectly calibrated predictions: confident and 100% correct
    probs = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    labels = np.array([1, 0, 2])
    ece = compute_ece(probs, labels, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)
    brier = compute_brier_score(probs, labels)
    assert brier == pytest.approx(0.0, abs=1e-5)
    nll = compute_nll(probs, labels)
    assert nll < 1e-4

    # Poorly calibrated: high confidence in wrong class
    wrong_probs = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
    ])
    wrong_labels = np.array([1, 0])
    high_ece = compute_ece(wrong_probs, wrong_labels, n_bins=5)
    assert high_ece > 0.5


def test_fit_and_apply_temperature():
    np.random.seed(42)
    # Synthetic overconfident logits
    logits = np.array([
        [5.0, 1.0, 0.0],
        [0.5, 6.0, 1.0],
        [1.0, 0.0, 4.0],
        [3.0, 2.0, 1.0],
    ])
    labels = np.array([0, 1, 2, 0])
    temp = fit_temperature(logits, labels)
    assert temp > 0.0
    cal_probs = apply_temperature(logits, temp)
    assert cal_probs.shape == (4, 3)
    np.testing.assert_allclose(np.sum(cal_probs, axis=1), 1.0, rtol=1e-5)


def test_save_and_load_temperature_roundtrip(tmp_path):
    temp_val = 1.854
    meta = {"model_name": "test-cross-encoder", "ece_before": 0.22, "ece_after": 0.05}
    file_path = tmp_path / "temperature.json"

    save_temperature(temp_val, file_path, metadata=meta)
    loaded = load_temperature(file_path)

    assert isinstance(loaded, float)
    assert pytest.approx(float(loaded), abs=1e-4) == temp_val
    assert loaded["temperature"] == temp_val
    assert loaded["model_name"] == "test-cross-encoder"
    assert loaded["ece_before"] == 0.22


def test_load_temperature_error_handling(tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_temperature(tmp_path / "non_existent.json")

    # Invalid JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("invalid json {", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_temperature(bad_json)

    # Missing temperature field
    missing_temp = tmp_path / "missing.json"
    missing_temp.write_text(json.dumps({"key": 123}), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        load_temperature(missing_temp)

    # Negative temperature
    neg_temp = tmp_path / "neg.json"
    neg_temp.write_text(json.dumps({"temperature": -1.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_temperature(neg_temp)


def test_verification_engine_uncalibrated_vs_calibrated(tmp_path):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split())))
        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded text"

    class FakeNLI:
        def __init__(self, temperature=1.0):
            self.temperature = float(temperature)
            self.is_calibrated = (self.temperature != 1.0)
            self.model_name = "fake-model"

        def predict_pairs(self, pairs):
            return [{"prediction": "entailment", "scores": {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}, "logits": [0.0, 2.0, -1.0]}]

    # Uncalibrated engine
    uncal = VerificationEngine(nli=FakeNLI(temperature=1.0), tokenizer=FakeTokenizer())
    assert not uncal.is_calibrated
    assert uncal.temperature == 1.0

    # Calibrated engine
    cal_file = tmp_path / "cal.json"
    save_temperature(2.25, cal_file)
    cal_engine = VerificationEngine(calibration_file=cal_file, nli=FakeNLI(temperature=2.25), tokenizer=FakeTokenizer())
    assert cal_engine.is_calibrated
    assert cal_engine.temperature == 2.25


# ---------------------------------------------------------------------------
# Task 3 & 4: Evaluation Metrics & Controlled Baselines Tests
# ---------------------------------------------------------------------------

def test_evaluation_metrics_computation():
    y_true = ["supported", "contradicted", "insufficient", "conflicting"]
    y_pred = ["supported", "contradicted", "insufficient", "supported"]  # Conflicting falsely predicted as supported
    actions = ["state", "withhold", "withhold", "state"]

    metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
    assert metrics["n_examples"] == 4
    assert metrics["accuracy"] == 0.75
    # FAR: 1 out of 3 non-supported claims received 'state'
    assert metrics["false_acceptance_rate"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    # Coverage: 2 out of 4 claims received state or caveat
    assert metrics["coverage"] == 0.50
    # Abstention: 2 out of 4 claims received withhold
    assert metrics["abstention_rate"] == 0.50


def test_experiment_runner_mock_workflow():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split())))
        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded text"

    class FakeNLIForExperiment:
        def __init__(self):
            self.temperature = 1.0
            self.is_calibrated = False
            self.model_name = "mock"
            self.label_ids = {"contradiction": 0, "entailment": 1, "neutral": 2}

        def predict_pairs(self, pairs):
            results = []
            for premise, claim in pairs:
                if "deadline" in premise.lower() and "deadline" in claim.lower():
                    results.append({"prediction": "entailment", "scores": {"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05}, "logits": [0.0, 3.0, -1.0]})
                else:
                    results.append({"prediction": "neutral", "scores": {"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8}, "logits": [-1.0, -1.0, 2.0]})
            return results

    class FakeRelevanceChecker:
        def check(self, claim, evidence):
            sim = 0.85 if "deadline" in evidence.lower() else 0.1
            return {"similarity": sim, "is_relevant": sim >= 0.3}

    engine = VerificationEngine(nli=FakeNLIForExperiment(), tokenizer=FakeTokenizer())
    runner = ExperimentRunner(engine=engine, relevance_checker=FakeRelevanceChecker())

    dataset = pd.DataFrame([
        {
            "example_id": "test_1",
            "claim": "The deadline is Nov 15",
            "evidence": "Application deadline is Nov 15",
            "true_label": "supported"
        },
        {
            "example_id": "test_2",
            "claim": "The exam is in June",
            "evidence": "Library has books",
            "true_label": "insufficient"
        }
    ])

    results = runner.run_all_experiments(dataset)
    assert "baseline_1_raw_nli" in results
    assert "baseline_2_aggregation" in results
    assert "baseline_3_numeric" in results
    assert "baseline_4_ungated" in results
    assert "proposed_gated" in results

    df = build_comparison_dataframe(results)
    assert len(df) == 5
    assert "Configuration" in df.columns
    assert "Accuracy" in df.columns
    assert "False Acceptance Rate" in df.columns
