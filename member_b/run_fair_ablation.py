"""Run fair ablation experiment isolating numeric/date checking (Phase 6.2).

Compares two strictly identical configurations on held-out test split (N=120):
- Configuration A: Aggregation + Temperature Scaling + Decision Policy (NO Numeric Checks)
- Configuration B: Aggregation + Temperature Scaling + Decision Policy (WITH Numeric Checks)
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "member_b"))

from member_b.verifier import VerificationEngine
from member_b.benchmark import parse_benchmark_dataset, compute_evaluation_metrics, LABELS
from member_b.calibration import load_temperature
from member_b.experiments import compute_bootstrap_confidence_intervals
from member_b.numeric_checks import check_numeric_conflict


def run_fair_ablation():
    print("=" * 70)
    print("PHASE 6.2: FAIR NUMERIC/DATE ABLATION EXPERIMENT")
    print("=" * 70)

    calib_artifact_path = ROOT / "member_b/results/calibration_canonical_artifact.json"
    artifact = load_temperature(calib_artifact_path)
    cal_temp = artifact.temperature
    print(f"Loaded Calibrated Temperature: T = {cal_temp:.4f}")

    test_csv = ROOT / "member_b/data/test_split.csv"
    test_df = pd.read_csv(test_csv)
    examples = parse_benchmark_dataset(test_df)
    doc_ids = [ex.get("document_id") for ex in examples]
    print(f"Loaded Held-Out Test Set: N = {len(examples)} claims across {len(set(doc_ids))} documents.")

    # Shared DeBERTa NLI engine with identical calibrated temperature
    # Shared tokenizer and thresholds
    eng_a = VerificationEngine(temperature=cal_temp, enable_numeric_checks=False)
    eng_b = VerificationEngine(temperature=cal_temp, enable_numeric_checks=True)

    def evaluate_engine(engine: VerificationEngine, config_name: str) -> tuple[dict, list[dict]]:
        y_true, y_pred, actions, rows, latencies = [], [], [], [], []
        num_flags = []

        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            res = engine.verify_claim(claim, passages)
            latency = (time.perf_counter() - t0) * 1000.0

            y_true.append(truth)
            y_pred.append(res["status"])
            actions.append(res["action"])
            latencies.append(latency)

            num_res = res.get("numeric_check", {})
            num_flags.append(bool(num_res.get("numeric_conflict", False)))

            rows.append({
                "example_id": ex["example_id"],
                "document_id": ex.get("document_id"),
                "claim": claim,
                "true_label": truth,
                "predicted_label": res["status"],
                "action": res["action"],
                "confidence": float(res["confidence"]),
                "numeric_conflict": bool(num_res.get("numeric_conflict", False)),
                "numeric_reason": num_res.get("reason", ""),
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = config_name

        # Calculate False Caveat Rate (FCR):
        # Fraction of non-conflicting claims that received action == 'caveat'
        non_conflicting_total = sum(1 for yt in y_true if yt != "conflicting")
        false_caveats = sum(1 for yt, act in zip(y_true, actions) if yt != "conflicting" and act == "caveat")
        metrics["false_caveat_rate"] = float(false_caveats / non_conflicting_total) if non_conflicting_total > 0 else 0.0
        metrics["false_caveat_count"] = int(false_caveats)
        metrics["non_conflicting_total"] = int(non_conflicting_total)

        # State coverage vs overall coverage
        metrics["state_coverage"] = float(sum(1 for a in actions if a == "state") / len(actions))
        metrics["numeric_flags_count"] = int(sum(num_flags))

        return metrics, rows

    print("\nRunning Configuration A (Numeric Checks DISABLED)...")
    metrics_a, rows_a = evaluate_engine(eng_a, "Config A: Verifier (Numeric Checks DISABLED)")

    print("Running Configuration B (Numeric Checks ENABLED)...")
    metrics_b, rows_b = evaluate_engine(eng_b, "Config B: Verifier (Numeric Checks ENABLED)")

    # Analyze direct differences between Config A and Config B
    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)

    diff_status = df_a[df_a["predicted_label"] != df_b["predicted_label"]]
    diff_action = df_a[df_a["action"] != df_b["action"]]

    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY TABLE (FAIR NUMERIC ABLATION)")
    print("=" * 70)

    summary_rows = [
        {
            "Metric": "Status Accuracy (95% CI)",
            "Config A (No Numeric Checks)": f"{metrics_a['accuracy']:.3f} [{metrics_a['accuracy_ci'][0]:.3f}, {metrics_a['accuracy_ci'][1]:.3f}]",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['accuracy']:.3f} [{metrics_b['accuracy_ci'][0]:.3f}, {metrics_b['accuracy_ci'][1]:.3f}]",
            "Delta (B - A)": f"{metrics_b['accuracy'] - metrics_a['accuracy']:+.3f}",
        },
        {
            "Metric": "Macro F1 (95% CI)",
            "Config A (No Numeric Checks)": f"{metrics_a['macro_f1']:.3f} [{metrics_a['macro_f1_ci'][0]:.3f}, {metrics_a['macro_f1_ci'][1]:.3f}]",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['macro_f1']:.3f} [{metrics_b['macro_f1_ci'][0]:.3f}, {metrics_b['macro_f1_ci'][1]:.3f}]",
            "Delta (B - A)": f"{metrics_b['macro_f1'] - metrics_a['macro_f1']:+.3f}",
        },
        {
            "Metric": "Supported F1",
            "Config A (No Numeric Checks)": f"{metrics_a['per_class']['supported']['f1']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['per_class']['supported']['f1']:.3f}",
            "Delta (B - A)": f"{metrics_b['per_class']['supported']['f1'] - metrics_a['per_class']['supported']['f1']:+.3f}",
        },
        {
            "Metric": "Contradicted F1",
            "Config A (No Numeric Checks)": f"{metrics_a['per_class']['contradicted']['f1']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['per_class']['contradicted']['f1']:.3f}",
            "Delta (B - A)": f"{metrics_b['per_class']['contradicted']['f1'] - metrics_a['per_class']['contradicted']['f1']:+.3f}",
        },
        {
            "Metric": "Insufficient F1",
            "Config A (No Numeric Checks)": f"{metrics_a['per_class']['insufficient']['f1']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['per_class']['insufficient']['f1']:.3f}",
            "Delta (B - A)": f"{metrics_b['per_class']['insufficient']['f1'] - metrics_a['per_class']['insufficient']['f1']:+.3f}",
        },
        {
            "Metric": "Conflicting F1",
            "Config A (No Numeric Checks)": f"{metrics_a['per_class']['conflicting']['f1']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['per_class']['conflicting']['f1']:.3f}",
            "Delta (B - A)": f"{metrics_b['per_class']['conflicting']['f1'] - metrics_a['per_class']['conflicting']['f1']:+.3f}",
        },
        {
            "Metric": "False Acceptance Rate (FAR)",
            "Config A (No Numeric Checks)": f"{metrics_a['false_acceptance_rate']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['false_acceptance_rate']:.3f}",
            "Delta (B - A)": f"{metrics_b['false_acceptance_rate'] - metrics_a['false_acceptance_rate']:+.3f}",
        },
        {
            "Metric": "False Caveat Rate (FCR)",
            "Config A (No Numeric Checks)": f"{metrics_a['false_caveat_rate']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['false_caveat_rate']:.3f}",
            "Delta (B - A)": f"{metrics_b['false_caveat_rate'] - metrics_a['false_caveat_rate']:+.3f}",
        },
        {
            "Metric": "State Coverage",
            "Config A (No Numeric Checks)": f"{metrics_a['state_coverage']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['state_coverage']:.3f}",
            "Delta (B - A)": f"{metrics_b['state_coverage'] - metrics_a['state_coverage']:+.3f}",
        },
        {
            "Metric": "Overall Coverage",
            "Config A (No Numeric Checks)": f"{metrics_a['coverage']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['coverage']:.3f}",
            "Delta (B - A)": f"{metrics_b['coverage'] - metrics_a['coverage']:+.3f}",
        },
        {
            "Metric": "Abstention Rate",
            "Config A (No Numeric Checks)": f"{metrics_a['abstention_rate']:.3f}",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['abstention_rate']:.3f}",
            "Delta (B - A)": f"{metrics_b['abstention_rate'] - metrics_a['abstention_rate']:+.3f}",
        },
        {
            "Metric": "Avg Latency (ms)",
            "Config A (No Numeric Checks)": f"{metrics_a['avg_latency_ms']:.1f} ms",
            "Config B (Numeric Checks Enabled)": f"{metrics_b['avg_latency_ms']:.1f} ms",
            "Delta (B - A)": f"{metrics_b['avg_latency_ms'] - metrics_a['avg_latency_ms']:+.1f} ms",
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print(f"\nDiscrepancy Details:")
    print(f"  Claims with Status differences: {len(diff_status)}")
    print(f"  Claims with Action differences: {len(diff_action)}")

    if len(diff_status) > 0:
        print("\nClaims where Predicted Status differed:")
        for idx in diff_status.index:
            ra = df_a.loc[idx]
            rb = df_b.loc[idx]
            print(f"  [{ra['example_id']}] True: {ra['true_label']}")
            print(f"    Claim: '{ra['claim']}'")
            print(f"    Config A: status={ra['predicted_label']}, action={ra['action']}")
            print(f"    Config B: status={rb['predicted_label']}, action={rb['action']} (num_flag={rb['numeric_conflict']}, reason: {rb['numeric_reason']})")

    # Save results
    results_dir = ROOT / "member_b/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ablation_payload = {
        "config_a_no_numeric": metrics_a,
        "config_b_with_numeric": metrics_b,
        "delta": {
            "accuracy": metrics_b["accuracy"] - metrics_a["accuracy"],
            "macro_f1": metrics_b["macro_f1"] - metrics_a["macro_f1"],
            "false_acceptance_rate": metrics_b["false_acceptance_rate"] - metrics_a["false_acceptance_rate"],
            "false_caveat_rate": metrics_b["false_caveat_rate"] - metrics_a["false_caveat_rate"],
            "state_coverage": metrics_b["state_coverage"] - metrics_a["state_coverage"],
            "overall_coverage": metrics_b["coverage"] - metrics_a["coverage"],
            "abstention_rate": metrics_b["abstention_rate"] - metrics_a["abstention_rate"],
            "avg_latency_ms": metrics_b["avg_latency_ms"] - metrics_a["avg_latency_ms"],
        },
        "rows_a": rows_a,
        "rows_b": rows_b,
    }

    # Sanitize numpy
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

    out_json = results_dir / "fair_numeric_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sanitize(ablation_payload), f, indent=2)

    df_summary.to_csv(results_dir / "fair_numeric_ablation_summary.csv", index=False)
    print(f"\nSaved fair ablation results to {out_json}")
    print(f"Saved fair ablation summary to {results_dir / 'fair_numeric_ablation_summary.csv'}")


if __name__ == "__main__":
    run_fair_ablation()

