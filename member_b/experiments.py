"""Controlled baseline and ablation experiments for the Evidence Review system.

Evaluates 5 standardized configurations on identical held-out datasets:
- Baseline 1: Raw NLI prediction (no windowing aggregation, no numeric rules)
- Baseline 2: NLI + Multi-passage aggregation (no numeric rules)
- Baseline 3: NLI + Numeric/date checks (direct pair check without full aggregation)
- Baseline 4: Complete Verifier ungated (windowing + aggregation + numeric checks)
- Proposed Variant: Complete Verifier with Relevance Gating
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

try:
    from .benchmark import parse_benchmark_dataset, compute_evaluation_metrics, LABELS
    from .verifier import VerificationEngine
    from .aggregation import aggregate_evidence
    from .numeric_checks import check_numeric_conflict
    from .decision_policy import decide
    from .relevance import RelevanceChecker
except ImportError:
    from benchmark import parse_benchmark_dataset, compute_evaluation_metrics, LABELS
    from verifier import VerificationEngine
    from aggregation import aggregate_evidence
    from numeric_checks import check_numeric_conflict
    from decision_policy import decide
    from relevance import RelevanceChecker


def _map_nli_to_label(nli_result: dict, entail_thresh: float = 0.5, contra_thresh: float = 0.5) -> tuple[str, float]:
    scores = nli_result.get("scores", {})
    e = scores.get("entailment", 0.0)
    c = scores.get("contradiction", 0.0)
    n = scores.get("neutral", 0.0)

    if e >= entail_thresh and e > c:
        return "supported", float(e)
    elif c >= contra_thresh and c > e:
        return "contradicted", float(c)
    else:
        return "insufficient", float(n)


class ExperimentRunner:
    """Orchestrates controlled baseline and ablation experiments."""

    def __init__(self, engine: VerificationEngine | None = None, relevance_checker: RelevanceChecker | None = None):
        self.engine = engine or VerificationEngine()
        self.relevance_checker = relevance_checker

    def run_baseline_1_raw_nli(self, examples: list[dict]) -> tuple[dict, list[dict]]:
        """Baseline 1: Raw NLI prediction directly on evidence text without windowing or numeric rules."""
        y_true, y_pred, actions, rows, latencies = [], [], [], [], []
        
        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            if not passages:
                status, conf = "insufficient", 0.0
            else:
                combined_text = " ".join(p["text"] for p in passages)
                pair_preds = self.engine.nli.predict_pairs([(combined_text, claim)])
                if pair_preds:
                    status, conf = _map_nli_to_label(
                        pair_preds[0],
                        self.engine.entailment_threshold,
                        self.engine.contradiction_threshold,
                    )
                else:
                    status, conf = "insufficient", 0.0
            latency = (time.perf_counter() - t0) * 1000.0

            action = "state" if status == "supported" else ("caveat" if status == "conflicting" else "withhold")
            y_true.append(truth)
            y_pred.append(status)
            actions.append(action)
            latencies.append(latency)
            rows.append({
                "example_id": ex["example_id"],
                "claim": claim,
                "true_label": truth,
                "predicted_label": status,
                "action": action,
                "confidence": conf,
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        doc_ids = [ex.get("document_id") for ex in examples]
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = "Baseline 1: Raw NLI"
        return metrics, rows

    def run_baseline_2_aggregation_only(self, examples: list[dict]) -> tuple[dict, list[dict]]:
        """Baseline 2: Windowing and multi-passage evidence aggregation without numeric/date checks."""
        y_true, y_pred, actions, rows, latencies = [], [], [], [], []

        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            if not passages:
                status, conf, action = "insufficient", 0.0, "withhold"
            else:
                # Run through verifier logic up to aggregation, skipping numeric checks
                all_pairs = []
                pair_meta = []
                for p_idx, p in enumerate(passages):
                    text = p.get("text", "").strip()
                    if not text:
                        continue
                    windows = self.engine.tokenizer and self.engine.window_size
                    # Use verifier windowing
                    from member_b.windowing import split_into_windows
                    wins = split_into_windows(
                        text, self.engine.tokenizer,
                        **({"window_size": self.engine.window_size} if self.engine.window_size else {}),
                        **({"overlap": self.engine.overlap} if self.engine.overlap is not None else {})
                    )
                    for w in wins:
                        all_pairs.append((w["text"], claim))
                        pair_meta.append((p_idx, w))

                pair_preds = self.engine.nli.predict_pairs(all_pairs)
                per_passage = []
                for p_idx, p in enumerate(passages):
                    p_rows = [pred for pred, (m_idx, _) in zip(pair_preds, pair_meta) if m_idx == p_idx]
                    best_e = max((r["scores"]["entailment"] for r in p_rows), default=0.0)
                    best_c = max((r["scores"]["contradiction"] for r in p_rows), default=0.0)
                    best_n = max((r["scores"]["neutral"] for r in p_rows), default=0.0)
                    per_passage.append({
                        "chunk_id": p.get("chunk_id", f"p_{p_idx}"),
                        "max_entailment": best_e,
                        "max_contradiction": best_c,
                        "max_neutral": best_n,
                    })

                agg = aggregate_evidence(
                    per_passage,
                    entail_threshold=self.engine.entailment_threshold,
                    contradiction_threshold=self.engine.contradiction_threshold,
                )
                status = agg["status"]
                conf = float(agg["confidence"])
                dec = decide(status, conf, numeric_conflict=False)
                action = dec["action"]
            latency = (time.perf_counter() - t0) * 1000.0

            y_true.append(truth)
            y_pred.append(status)
            actions.append(action)
            latencies.append(latency)
            rows.append({
                "example_id": ex["example_id"],
                "claim": claim,
                "true_label": truth,
                "predicted_label": status,
                "action": action,
                "confidence": conf,
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        doc_ids = [ex.get("document_id") for ex in examples]
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = "Baseline 2: NLI + Aggregation (No Numeric Rules)"
        return metrics, rows

    def run_baseline_3_numeric_only(self, examples: list[dict]) -> tuple[dict, list[dict]]:
        """Baseline 3: NLI pair check combined with numeric/date conflict detection."""
        y_true, y_pred, actions, rows, latencies = [], [], [], [], []

        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            if not passages:
                status, conf, action = "insufficient", 0.0, "withhold"
            else:
                combined_text = " ".join(p["text"] for p in passages)
                pair_preds = self.engine.nli.predict_pairs([(combined_text, claim)])
                if pair_preds:
                    status, conf = _map_nli_to_label(
                        pair_preds[0],
                        self.engine.entailment_threshold,
                        self.engine.contradiction_threshold,
                    )
                else:
                    status, conf = "insufficient", 0.0
                
                # Apply numeric check directly
                num_check = check_numeric_conflict(claim, passages)
                if num_check["numeric_conflict"]:
                    if status == "supported":
                        status = "contradicted"
                dec = decide(status, conf, numeric_conflict=num_check["numeric_conflict"])
                action = dec["action"]
            latency = (time.perf_counter() - t0) * 1000.0

            y_true.append(truth)
            y_pred.append(status)
            actions.append(action)
            latencies.append(latency)
            rows.append({
                "example_id": ex["example_id"],
                "claim": claim,
                "true_label": truth,
                "predicted_label": status,
                "action": action,
                "confidence": conf,
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        doc_ids = [ex.get("document_id") for ex in examples]
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = "Baseline 3: NLI + Numeric Checks (No Aggregation)"
        return metrics, rows

    def run_baseline_4_complete_ungated(self, examples: list[dict]) -> tuple[dict, list[dict]]:
        """Baseline 4: Complete verifier without relevance gating (standard VerificationEngine)."""
        y_true, y_pred, actions, rows, latencies = [], [], [], [], []

        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            res = self.engine.verify_claim(claim, passages)
            latency = (time.perf_counter() - t0) * 1000.0

            y_true.append(truth)
            y_pred.append(res["status"])
            actions.append(res["action"])
            latencies.append(latency)
            rows.append({
                "example_id": ex["example_id"],
                "claim": claim,
                "true_label": truth,
                "predicted_label": res["status"],
                "action": res["action"],
                "confidence": float(res["confidence"]),
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        doc_ids = [ex.get("document_id") for ex in examples]
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = "Baseline 4: Complete Verifier (Ungated)"
        return metrics, rows

    def run_proposed_relevance_gated(self, examples: list[dict], threshold: float = 0.3) -> tuple[dict, list[dict]]:
        """Proposed Variant: Complete verifier with relevance gating."""
        if self.relevance_checker is None:
            self.relevance_checker = RelevanceChecker()

        y_true, y_pred, actions, rows, latencies = [], [], [], [], []

        for ex in examples:
            claim = ex["claim"]
            passages = ex["evidence_passages"]
            truth = ex["true_label"]

            t0 = time.perf_counter()
            # Filter passages with relevance checker
            gated_passages = []
            for p in passages:
                text = p.get("text", "")
                rel = self.relevance_checker.check(claim, text)
                if rel["similarity"] >= threshold:
                    p_copy = dict(p)
                    p_copy["relevance_similarity"] = rel["similarity"]
                    gated_passages.append(p_copy)
            
            # If all passages filtered out, pass empty list to verifier safely
            res = self.engine.verify_claim(claim, gated_passages)
            latency = (time.perf_counter() - t0) * 1000.0

            y_true.append(truth)
            y_pred.append(res["status"])
            actions.append(res["action"])
            latencies.append(latency)
            rows.append({
                "example_id": ex["example_id"],
                "claim": claim,
                "true_label": truth,
                "predicted_label": res["status"],
                "action": res["action"],
                "confidence": float(res["confidence"]),
                "n_passages_before_gate": len(passages),
                "n_passages_after_gate": len(gated_passages),
                "latency_ms": latency,
            })

        metrics = compute_evaluation_metrics(y_true, y_pred, actions=actions)
        doc_ids = [ex.get("document_id") for ex in examples]
        metrics.update(compute_bootstrap_confidence_intervals(y_true, y_pred, doc_ids=doc_ids))
        metrics["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        metrics["config_name"] = "Proposed: Verifier + Relevance Gating"
        return metrics, rows

    def run_all_experiments(self, dataset_path_or_df) -> dict[str, dict]:
        """Run all 5 configurations on identical dataset and return consolidated comparison."""
        if isinstance(dataset_path_or_df, pd.DataFrame):
            df = dataset_path_or_df.copy()
        else:
            df = pd.read_csv(dataset_path_or_df)

        examples = parse_benchmark_dataset(df)

        b1_metrics, _ = self.run_baseline_1_raw_nli(examples)
        b2_metrics, _ = self.run_baseline_2_aggregation_only(examples)
        b3_metrics, _ = self.run_baseline_3_numeric_only(examples)
        b4_metrics, _ = self.run_baseline_4_complete_ungated(examples)
        prop_metrics, _ = self.run_proposed_relevance_gated(examples)

        all_results = {
            "baseline_1_raw_nli": b1_metrics,
            "baseline_2_aggregation": b2_metrics,
            "baseline_3_numeric": b3_metrics,
            "baseline_4_ungated": b4_metrics,
            "proposed_gated": prop_metrics,
        }
        return all_results


from sklearn.metrics import accuracy_score, f1_score


def compute_bootstrap_confidence_intervals(y_true: list[str], y_pred: list[str],
                                           doc_ids: list[str] | None = None,
                                           n_bootstraps: int = 500,
                                           ci: float = 95.0,
                                           seed: int = 42) -> dict:
    """Compute bootstrap confidence intervals at document-cluster level."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return {}

    unique_docs = sorted(list(set(doc_ids))) if doc_ids is not None and any(doc_ids) else None
    accs, f1s = [], []

    for _ in range(n_bootstraps):
        if unique_docs:
            sampled_docs = rng.choice(unique_docs, size=len(unique_docs), replace=True)
            indices = [i for i, d in enumerate(doc_ids) if d in sampled_docs]
            if not indices:
                continue
        else:
            indices = rng.choice(n, size=n, replace=True)

        b_true = [y_true[i] for i in indices]
        b_pred = [y_pred[i] for i in indices]

        accs.append(accuracy_score(b_true, b_pred))
        f1s.append(f1_score(b_true, b_pred, labels=LABELS, average="macro", zero_division=0))

    alpha = (100.0 - ci) / 2.0
    return {
        "accuracy_ci": (round(float(np.percentile(accs, alpha)), 3), round(float(np.percentile(accs, 100.0 - alpha)), 3)),
        "macro_f1_ci": (round(float(np.percentile(f1s, alpha)), 3), round(float(np.percentile(f1s, 100.0 - alpha)), 3)),
    }


def build_comparison_dataframe(experiment_results: dict[str, dict]) -> pd.DataFrame:
    """Format multiple experiment summaries into a standardized comparison DataFrame."""
    rows = []
    for key, metrics in experiment_results.items():
        acc_ci = metrics.get("accuracy_ci")
        f1_ci = metrics.get("macro_f1_ci")
        acc_ci_str = f" [{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]" if acc_ci else ""
        f1_ci_str = f" [{f1_ci[0]:.2f}, {f1_ci[1]:.2f}]" if f1_ci else ""

        rows.append({
            "Configuration": metrics.get("config_name", key),
            "Accuracy": f"{round(metrics['accuracy'], 3):.3f}{acc_ci_str}",
            "Macro F1": f"{round(metrics['macro_f1'], 3):.3f}{f1_ci_str}",
            "Macro Precision": round(metrics.get("macro_precision", 0.0), 3),
            "Macro Recall": round(metrics.get("macro_recall", 0.0), 3),
            "False Acceptance Rate": round(metrics.get("false_acceptance_rate", 0.0), 3),
            "Coverage": round(metrics.get("coverage", 0.0), 3),
            "Abstention Rate": round(metrics.get("abstention_rate", 0.0), 3),
            "Avg Latency (ms)": round(metrics.get("avg_latency_ms", 0.0), 1),
        })
    return pd.DataFrame(rows)
