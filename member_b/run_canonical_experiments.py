"""Run canonical research experiments on held-out test split (Phase 6, Tasks 6 & 7).

Evaluates 5 standardized configurations on held-out test data (member_b/data/test_split.csv)
with zero document leakage. Computes 95% bootstrap confidence intervals at the document
group level. Calibrates temperature scaling strictly on member_b/data/calibration_split.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Enforce offline mode before any huggingface/transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import pandas as pd

# Add repo root and member_b to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "member_b"))

from member_b.verifier import VerificationEngine
from member_b.relevance import RelevanceChecker
from member_b.experiments import ExperimentRunner, build_comparison_dataframe
from member_b.calibration import (
    collect_logits_and_labels,
    calibrate_and_evaluate,
    apply_temperature,
    compute_ece,
    compute_brier_score,
    compute_nll,
    save_temperature,
)
from scipy.special import softmax


def main():
    print("=" * 70)
    print("PHASE 6: CANONICAL RESEARCH EXPERIMENTS & CALIBRATION")
    print("=" * 70)

    results_dir = ROOT / "member_b" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    calib_csv = ROOT / "member_b" / "data" / "calibration_split.csv"
    val_csv = ROOT / "member_b" / "data" / "validation_split.csv"
    test_csv = ROOT / "member_b" / "data" / "test_split.csv"

    print(f"Loading calibration split from: {calib_csv}")
    calib_df = pd.read_csv(calib_csv)
    print(f"Loading validation split from: {val_csv}")
    val_df = pd.read_csv(val_csv)
    print(f"Loading held-out test split from: {test_csv}")
    test_df = pd.read_csv(test_csv)

    print(f"Calibration examples: {len(calib_df)} (docs: {calib_df['document_id'].nunique()})")
    print(f"Validation examples: {len(val_df)} (docs: {val_df['document_id'].nunique()})")
    print(f"Held-out test examples: {len(test_df)} (docs: {test_df['document_id'].nunique()})")

    # 1. Initialize Verification Engine
    print("\n[1/3] Initializing VerificationEngine (DeBERTa model)...")
    engine = VerificationEngine()
    relevance_checker = RelevanceChecker()

    # 2. Temperature Calibration (Task 7)
    print("\n[2/3] Fitting Temperature Scaling strictly on calibration split...")
    calib_logits, calib_labels = collect_logits_and_labels(engine.nli, calib_df)
    print(f"Collected {len(calib_labels)} premise-hypothesis calibration pairs.")

    temp, calib_report = calibrate_and_evaluate(
        calib_logits, calib_labels, model_name=engine.nli.model_name
    )
    print(f"Fitted Optimal Temperature T = {temp:.4f}")
    print(f"  Calibration split ECE:   {calib_report['ece_uncalibrated']:.4f} -> {calib_report['ece_calibrated']:.4f}")
    print(f"  Calibration split Brier: {calib_report['brier_uncalibrated']:.4f} -> {calib_report['brier_calibrated']:.4f}")
    print(f"  Calibration split NLL:   {calib_report['nll_uncalibrated']:.4f} -> {calib_report['nll_calibrated']:.4f}")

    # Evaluate calibration transfer to held-out test split
    test_logits, test_labels = collect_logits_and_labels(engine.nli, test_df)
    print(f"Collected {len(test_labels)} premise-hypothesis pairs from held-out test split.")
    test_uncal_probs = softmax(test_logits, axis=1)
    test_cal_probs = apply_temperature(test_logits, temp)

    test_ece_before = compute_ece(test_uncal_probs, test_labels)
    test_ece_after = compute_ece(test_cal_probs, test_labels)
    test_brier_before = compute_brier_score(test_uncal_probs, test_labels)
    test_brier_after = compute_brier_score(test_cal_probs, test_labels)
    test_nll_before = compute_nll(test_uncal_probs, test_labels)
    test_nll_after = compute_nll(test_cal_probs, test_labels)

    print(f"Held-out Test Calibration Transfer:")
    print(f"  Test split ECE:   {test_ece_before:.4f} -> {test_ece_after:.4f}")
    print(f"  Test split Brier: {test_brier_before:.4f} -> {test_brier_after:.4f}")
    print(f"  Test split NLL:   {test_nll_before:.4f} -> {test_nll_after:.4f}")

    calib_full_artifact = {
        "temperature": temp,
        "is_calibrated": True,
        "calibration_set": {
            "n_samples": int(len(calib_labels)),
            "n_claims": int(len(calib_df)),
            "ece_uncalibrated": calib_report["ece_uncalibrated"],
            "ece_calibrated": calib_report["ece_calibrated"],
            "brier_uncalibrated": calib_report["brier_uncalibrated"],
            "brier_calibrated": calib_report["brier_calibrated"],
            "nll_uncalibrated": calib_report["nll_uncalibrated"],
            "nll_calibrated": calib_report["nll_calibrated"],
        },
        "held_out_test_set": {
            "n_samples": int(len(test_labels)),
            "n_claims": int(len(test_df)),
            "ece_uncalibrated": float(test_ece_before),
            "ece_calibrated": float(test_ece_after),
            "brier_uncalibrated": float(test_brier_before),
            "brier_calibrated": float(test_brier_after),
            "nll_uncalibrated": float(test_nll_before),
            "nll_calibrated": float(test_nll_after),
        },
    }
    save_temperature(
        temp,
        results_dir / "calibration_canonical_artifact.json",
        metadata=calib_full_artifact,
    )
    print(f"Saved calibration artifact to: {results_dir / 'calibration_canonical_artifact.json'}")

    # Set calibrated temperature on engine
    engine.temperature = temp

    # 3. Controlled Baseline & Proposed Experiments on Held-Out Test Set (Task 6)
    print("\n[3/3] Running 5 experimental configurations on held-out test split (N=120)...")
    runner = ExperimentRunner(engine=engine, relevance_checker=relevance_checker)
    all_results = runner.run_all_experiments(test_df)

    comparison_df = build_comparison_dataframe(all_results)
    print("\n" + "=" * 70)
    print("EXPERIMENTAL EVALUATION RESULTS (Held-Out Test Set, N=120 Claims):")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Save summary and full results
    comparison_df.to_csv(results_dir / "canonical_experiment_summary.csv", index=False)
    
    # Serialize results to json (converting numpy types)
    def sanitize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(results_dir / "canonical_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(sanitize(all_results), f, indent=2)

    print(f"\nSaved experiment summary to: {results_dir / 'canonical_experiment_summary.csv'}")
    print(f"Saved detailed results to: {results_dir / 'canonical_experiment_results.json'}")
    print("\nCanonical evaluation experiments successfully completed.")


if __name__ == "__main__":
    main()
