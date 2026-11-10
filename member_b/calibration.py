"""Temperature scaling and calibration metrics for multiclass NLI logits.

Evaluates calibration using Expected Calibration Error (ECE), Brier score,
and negative log-likelihood (NLL). Supports saving and loading calibration
artifacts with comprehensive metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


LABEL_TO_CLASS = {
    "contradiction": 0,
    "contradicted": 0,
    "entailment": 1,
    "supported": 1,
    "neutral": 2,
    "insufficient": 2,
}


class CalibrationArtifact(float):
    """Float representing temperature with attached metadata for backward compatibility."""
    def __new__(cls, temperature: float, metadata: dict | None = None):
        val = super().__new__(cls, float(temperature))
        val.temperature = float(temperature)
        val.metadata = dict(metadata or {})
        return val

    @property
    def is_calibrated(self) -> bool:
        return bool(self.metadata.get("is_calibrated", self.temperature != 1.0))

    def __getitem__(self, key):
        if key == "temperature":
            return self.temperature
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "temperature":
            return self.temperature
        return self.metadata.get(key, default)

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "metadata": self.metadata}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for predicted probabilities.
    
    probs: array [n_samples, n_classes] of probabilities (or confidence scores).
    labels: array [n_samples] of integer ground truth class IDs.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(probs) == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score: mean squared difference between predicted
    probabilities and one-hot true labels.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(probs) == 0:
        return 0.0

    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), labels] = 1.0

    # Mean squared error across classes and samples
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Negative Log-Likelihood (cross-entropy loss)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(probs) == 0:
        return 0.0

    clipped_probs = np.clip(probs, eps, 1.0)
    correct_class_probs = clipped_probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(correct_class_probs)))


def fit_temperature(logits: Sequence[Sequence[float]] | np.ndarray, label_ids: Sequence[int] | np.ndarray) -> float:
    """Fit one scalar temperature using a separate calibration set via NLL minimization.

    logits: array [n_examples, 3] of raw model logits.
    label_ids: integer class IDs in the model's output-column order.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(label_ids, dtype=int)
    if logits.ndim != 2 or logits.shape[1] != 3 or len(labels) != len(logits):
        raise ValueError("Expected logits [n, 3] and n integer labels")

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        log_probs = log_softmax(logits / temperature, axis=1)
        return float(-np.mean(log_probs[np.arange(len(labels)), labels]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(logits: Sequence[Sequence[float]] | np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to raw logits and return calibrated probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def collect_logits_and_labels(nli_engine, calibration_data) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw uncalibrated logits and integer labels for a calibration dataset.
    
    calibration_data can be:
    - Sequence of dicts: [{"premise": str, "hypothesis": str, "label": int|str}]
      or [{"evidence": str, "claim": str, "true_label": str}]
    - Sequence of tuples: (premise, hypothesis, label)
    - pandas DataFrame with claim, evidence, true_label
    """
    import pandas as pd
    if isinstance(calibration_data, pd.DataFrame):
        records = calibration_data.to_dict(orient="records")
    else:
        records = list(calibration_data)

    pairs = []
    labels = []

    for item in records:
        if isinstance(item, (tuple, list)):
            premise, hypothesis, raw_label = item[0], item[1], item[2]
        elif isinstance(item, dict):
            hypothesis = item.get("claim") or item.get("hypothesis") or ""
            raw_passages = item.get("evidence_passages")
            if raw_passages:
                if isinstance(raw_passages, str):
                    try:
                        raw_passages = json.loads(raw_passages)
                    except Exception:
                        raw_passages = []
                for p in raw_passages:
                    p_text = p.get("text", "")
                    p_rel = p.get("relationship") or item.get("true_label") or item.get("true_status")
                    if p_text and hypothesis and p_rel:
                        p_rel_str = str(p_rel).strip().lower()
                        if p_rel_str in ("entails", "supported"):
                            p_idx = LABEL_TO_CLASS["entailment"]
                        elif p_rel_str in ("contradicts", "contradicted"):
                            p_idx = LABEL_TO_CLASS["contradiction"]
                        elif p_rel_str in ("neutral", "irrelevant", "insufficient"):
                            p_idx = LABEL_TO_CLASS["neutral"]
                        else:
                            continue
                        pairs.append((str(p_text).strip(), str(hypothesis).strip()))
                        labels.append(p_idx)
                continue

            premise = item.get("evidence") or item.get("premise") or item.get("text") or ""
            raw_label = item.get("true_label") or item.get("true_status") or item.get("label")
        else:
            continue

        if not str(premise).strip() or not str(hypothesis).strip():
            continue

        if isinstance(raw_label, int):
            label_idx = raw_label
        else:
            label_str = str(raw_label).strip().lower()
            if label_str not in LABEL_TO_CLASS:
                continue
            label_idx = LABEL_TO_CLASS[label_str]

        pairs.append((str(premise).strip(), str(hypothesis).strip()))
        labels.append(label_idx)

    if not pairs:
        return np.empty((0, 3)), np.empty((0,), dtype=int)

    # Use nli_engine.model directly to obtain raw uncalibrated logits
    logits = nli_engine.model.predict(
        pairs,
        convert_to_numpy=True,
        apply_softmax=False,
        show_progress_bar=False,
    )
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    
    return logits, np.asarray(labels, dtype=int)


def calibrate_and_evaluate(logits: np.ndarray, labels: np.ndarray, model_name: str = "") -> tuple[float, dict]:
    """Fit temperature on logits and return (temperature, calibration_report)."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    uncal_probs = softmax(logits, axis=1)
    ece_before = compute_ece(uncal_probs, labels)
    brier_before = compute_brier_score(uncal_probs, labels)
    nll_before = compute_nll(uncal_probs, labels)

    temp = fit_temperature(logits, labels)

    cal_probs = apply_temperature(logits, temp)
    ece_after = compute_ece(cal_probs, labels)
    brier_after = compute_brier_score(cal_probs, labels)
    nll_after = compute_nll(cal_probs, labels)

    metadata = {
        "model_name": model_name,
        "n_samples": int(len(labels)),
        "ece_uncalibrated": float(ece_before),
        "ece_calibrated": float(ece_after),
        "brier_uncalibrated": float(brier_before),
        "brier_calibrated": float(brier_after),
        "nll_uncalibrated": float(nll_before),
        "nll_calibrated": float(nll_after),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    return temp, metadata


def save_temperature(temperature: float, path: str | Path = "temperature.json", metadata: dict | None = None) -> None:
    """Save temperature and metadata to a JSON file."""
    data = {
        "temperature": float(temperature),
        "is_calibrated": float(temperature) != 1.0,
        "metadata": metadata or {},
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_temperature(path: str | Path = "temperature.json") -> CalibrationArtifact:
    """Load temperature and metadata from a JSON file.
    
    Returns a CalibrationArtifact (float subclass with .metadata and .temperature).
    Raises FileNotFoundError or ValueError if invalid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON in calibration file {file_path}: {exc}") from exc

    if not isinstance(data, dict) or "temperature" not in data:
        raise ValueError(f"Calibration file must contain a 'temperature' field: {file_path}")

    try:
        temp = float(data["temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid temperature value {data.get('temperature')!r} in {file_path}") from exc

    if temp <= 0:
        raise ValueError(f"Temperature must be positive, got {temp}")

    metadata = data.get("metadata", {})
    return CalibrationArtifact(temp, metadata=metadata)
