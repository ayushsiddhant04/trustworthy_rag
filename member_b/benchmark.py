"""Run a labelled claim-evidence benchmark and save metrics.

Supports single-passage rows (legacy schema) and multi-passage rows (new schema
with example_id, claim, evidence_passages / evidence, true_label).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)

try:
    from .verifier import VerificationEngine
except ImportError:
    from verifier import VerificationEngine

LABELS = ["supported", "contradicted", "insufficient", "conflicting"]


def _parse_benchmark_evidence(raw_evidence, default_source=None, default_page=None) -> list[dict]:
    """Parse benchmark evidence supporting JSON array, ' || ' delimiter, single text, or empty.
    
    Returns a list of dicts: [{"chunk_id": str, "text": str, "source": ..., "page": ...}]
    """
    if raw_evidence is None or (isinstance(raw_evidence, float) and pd.isna(raw_evidence)):
        return []
    
    if isinstance(raw_evidence, list):
        parsed_list = []
        for i, item in enumerate(raw_evidence):
            if isinstance(item, dict):
                p = dict(item)
                p.setdefault("chunk_id", f"passage_{i+1}")
                p.setdefault("source", default_source)
                p.setdefault("page", default_page)
                if "text" not in p:
                    p["text"] = str(p)
                if str(p["text"]).strip():
                    parsed_list.append(p)
            else:
                txt = str(item).strip()
                if txt:
                    parsed_list.append({
                        "chunk_id": f"passage_{i+1}",
                        "text": txt,
                        "source": default_source,
                        "page": default_page,
                    })
        return parsed_list

    text_val = str(raw_evidence).strip()
    if not text_val or text_val.lower() in ("nan", "none", "null", "[]"):
        return []

    # Try JSON parsing
    if text_val.startswith("[") and text_val.endswith("]"):
        try:
            parsed = json.loads(text_val)
            if isinstance(parsed, list):
                return _parse_benchmark_evidence(parsed, default_source=default_source, default_page=default_page)
        except Exception:
            # Fall back to raw text if malformed JSON
            pass

    # Try delimited string " || "
    if " || " in text_val:
        parts = text_val.split(" || ")
        return [
            {
                "chunk_id": f"passage_{i+1}",
                "text": p.strip(),
                "source": default_source,
                "page": default_page,
            }
            for i, p in enumerate(parts) if p.strip()
        ]

    return [{
        "chunk_id": "passage_1",
        "text": text_val,
        "source": default_source,
        "page": default_page,
    }]


def compute_evaluation_metrics(y_true: list[str], y_pred: list[str], actions: list[str] | None = None) -> dict:
    """Compute comprehensive evaluation metrics including accuracy, macro F1,
    per-class precision/recall/F1, confusion matrix, FAR, coverage, and abstention rate.
    """
    y_true_clean = [str(y).strip().lower() for y in y_true]
    y_pred_clean = [str(y).strip().lower() for y in y_pred]

    acc = float(accuracy_score(y_true_clean, y_pred_clean)) if y_true_clean else 0.0
    macro_f1 = float(f1_score(y_true_clean, y_pred_clean, labels=LABELS, average="macro", zero_division=0))
    macro_p = float(precision_score(y_true_clean, y_pred_clean, labels=LABELS, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_clean, y_pred_clean, labels=LABELS, average="macro", zero_division=0))

    matrix = confusion_matrix(y_true_clean, y_pred_clean, labels=LABELS)
    report = classification_report(y_true_clean, y_pred_clean, labels=LABELS, output_dict=True, zero_division=0)

    # Per-class metrics
    per_class = {}
    for label in LABELS:
        if label in report:
            per_class[label] = {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }

    # False Acceptance Rate (FAR):
    # Fraction of non-supported ground truth claims incorrectly accepted with action == 'state'
    # (or predicted as 'supported' if action not available)
    non_supported_count = sum(1 for yt in y_true_clean if yt != "supported")
    if actions is not None and len(actions) == len(y_true_clean):
        false_accepts = sum(
            1 for yt, act in zip(y_true_clean, actions)
            if yt != "supported" and str(act).strip().lower() == "state"
        )
    else:
        false_accepts = sum(
            1 for yt, yp in zip(y_true_clean, y_pred_clean)
            if yt != "supported" and yp == "supported"
        )
    far = float(false_accepts / non_supported_count) if non_supported_count > 0 else 0.0

    # Coverage: fraction of claims receiving 'state' or 'caveat'
    # Abstention rate: fraction of claims receiving 'withhold'
    n_total = len(y_true_clean)
    if actions is not None and n_total > 0:
        clean_actions = [str(a).strip().lower() for a in actions]
        covered = sum(1 for a in clean_actions if a in ("state", "caveat"))
        withheld = sum(1 for a in clean_actions if a == "withhold")
        coverage = float(covered / n_total)
        abstention = float(withheld / n_total)
    else:
        coverage = float(sum(1 for yp in y_pred_clean if yp in ("supported", "conflicting")) / n_total) if n_total else 0.0
        abstention = float(sum(1 for yp in y_pred_clean if yp in ("contradicted", "insufficient")) / n_total) if n_total else 0.0

    return {
        "n_examples": n_total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "false_acceptance_rate": far,
        "coverage": coverage,
        "abstention_rate": abstention,
    }


def parse_benchmark_dataset(df: pd.DataFrame) -> list[dict]:
    """Parse and normalize benchmark DataFrame into structured examples.
    
    Supports:
    1. Schema A: example_id, claim, evidence_passages (or evidence), true_label
    2. Schema B: claim, evidence, true_label
    Groups rows by example_id if multiple rows share the same example_id.
    """
    if df.empty:
        raise ValueError("Benchmark CSV has no rows")

    cols = {c.lower(): c for c in df.columns}
    
    # Identify claim column
    claim_col = cols.get("claim")
    if not claim_col:
        raise ValueError("CSV missing required 'claim' column")

    # Identify label column
    label_col = cols.get("true_status") or cols.get("true_label") or cols.get("label") or cols.get("ground_truth")
    if not label_col:
        raise ValueError("CSV missing required 'true_label' or 'true_status' column")

    # Identify evidence column
    evidence_col = cols.get("evidence_passages") or cols.get("evidence") or cols.get("passages")
    if not evidence_col:
        raise ValueError("CSV missing required 'evidence' or 'evidence_passages' column")

    example_id_col = cols.get("example_id") or cols.get("id")
    document_id_col = cols.get("document_id")
    source_col = cols.get("source") or cols.get("source_id")
    page_col = cols.get("page")

    grouped_examples = {}
    
    for idx, row in df.iterrows():
        raw_ex_id = str(row[example_id_col]).strip() if example_id_col else f"example_{idx+1}"
        if not raw_ex_id or raw_ex_id.lower() == "nan":
            raw_ex_id = f"example_{idx+1}"

        doc_id = str(row[document_id_col]).strip() if document_id_col and pd.notna(row[document_id_col]) else None
        claim_text = str(row[claim_col]).strip()
        true_label = str(row[label_col]).strip().lower()
        if true_label not in LABELS:
            raise ValueError(f"Unknown true_label {true_label!r} in row {idx+1}; allowed: {LABELS}")

        src = str(row[source_col]).strip() if source_col and pd.notna(row[source_col]) else None
        pg = int(row[page_col]) if page_col and pd.notna(row[page_col]) and str(row[page_col]).isdigit() else None

        passages = _parse_benchmark_evidence(row[evidence_col], default_source=src, default_page=pg)

        if raw_ex_id in grouped_examples:
            # Grouping multiple rows for the same example_id
            existing = grouped_examples[raw_ex_id]
            if existing["claim"] != claim_text:
                pass  # Keep first claim text or warn
            # Append new passages
            start_num = len(existing["evidence_passages"]) + 1
            for p_idx, p in enumerate(passages):
                p_copy = dict(p)
                p_copy["chunk_id"] = p_copy.get("chunk_id") or f"passage_{start_num + p_idx}"
                existing["evidence_passages"].append(p_copy)
        else:
            grouped_examples[raw_ex_id] = {
                "example_id": raw_ex_id,
                "document_id": doc_id,
                "claim": claim_text,
                "evidence_passages": passages,
                "true_label": true_label,
            }

    return list(grouped_examples.values())


def run_benchmark(csv_path_or_df, output_dir="results", engine=None):
    """Run benchmark evaluation on single or multi-passage dataset.
    
    csv_path_or_df: Path to CSV, file-like object, or pandas DataFrame.
    output_dir: Directory to save evaluation artifacts.
    engine: VerificationEngine instance (defaults to VerificationEngine()).
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df)

    examples = parse_benchmark_dataset(df)
    engine = engine or VerificationEngine()

    y_true = []
    y_pred = []
    actions = []
    output_rows = []
    latencies_ms = []

    for ex in examples:
        claim = ex["claim"]
        passages = ex["evidence_passages"]
        truth = ex["true_label"]

        t0 = time.perf_counter()
        result = engine.verify_claim(claim, passages)
        latency = (time.perf_counter() - t0) * 1000.0

        y_true.append(truth)
        y_pred.append(result["status"])
        actions.append(result["action"])
        latencies_ms.append(latency)

        # Serialized evidence representation for tabular display / CSV
        if not passages:
            evidence_summary = ""
        elif len(passages) == 1:
            evidence_summary = passages[0]["text"]
        else:
            evidence_summary = " || ".join(p["text"] for p in passages)

        output_rows.append({
            "example_id": ex["example_id"],
            "claim": claim,
            "evidence": evidence_summary,
            "n_passages": len(passages),
            "true_label": truth,
            "predicted_label": result["status"],
            "action": result["action"],
            "confidence": float(result["confidence"]),
            "reason": result["reason"],
            "latency_ms": round(latency, 2),
        })

    predictions = pd.DataFrame(output_rows)
    metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
    metrics["avg_latency_ms"] = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out_path / "predictions.csv", index=False)

        report_df = pd.DataFrame(metrics["per_class"]).transpose()
        report_df.to_csv(out_path / "classification_report.csv")

        matrix_df = pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS)
        matrix_df.to_csv(out_path / "confusion_matrix.csv")

        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV with claim,evidence,true_label columns")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_benchmark(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()
