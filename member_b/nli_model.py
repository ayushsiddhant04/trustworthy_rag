"""Load the pretrained NLI cross-encoder and return logits/probabilities.

Set engine.temperature to a value fitted on a separate labelled calibration set.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sentence_transformers import CrossEncoder

try:
    from .config import MODEL_NAME, MAX_LENGTH, KNOWN_MODEL_LABEL_ORDER
except ImportError:  # also allow importing when member_b itself is on sys.path
    from config import MODEL_NAME, MAX_LENGTH, KNOWN_MODEL_LABEL_ORDER


class NLIEngine:
    LABELS = ("contradiction", "entailment", "neutral")

    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH, device=None, temperature: float = 1.0):
        kwargs = {"max_length": max_length}
        if device is not None:
            kwargs["device"] = device
        self.model_name = model_name
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.is_calibrated = bool(getattr(temperature, "is_calibrated", self.temperature != 1.0))
        self.calibration_metadata = getattr(temperature, "metadata", {})
        self.model = CrossEncoder(model_name, **kwargs)
        self.label_ids = self._read_label_ids()

    def _read_label_ids(self):
        raw = getattr(self.model.model.config, "id2label", {}) or {}
        normalized = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            name = str(value).strip().lower().replace(" ", "_")
            if name.startswith("label_"):
                continue
            normalized[name] = idx

        # Some checkpoints expose generic LABEL_0 names. For the exact model
        # named here, use the published model-card order as a documented fallback.
        if all(label in normalized for label in self.LABELS):
            return {label: normalized[label] for label in self.LABELS}
        if self.model_name == MODEL_NAME and len(KNOWN_MODEL_LABEL_ORDER) == 3:
            return {label: idx for idx, label in KNOWN_MODEL_LABEL_ORDER.items()}
        raise ValueError(
            "Could not map NLI labels from model config. Inspect config.id2label "
            "and set an explicit, documented label mapping before continuing."
        )

    def predict_pairs(self, pairs):
        """pairs: sequence of (evidence/premise, claim/hypothesis) tuples."""
        if not pairs:
            return []
        logits = self.model.predict(
            list(pairs),
            convert_to_numpy=True,
            apply_softmax=False,
            show_progress_bar=False,
        )
        logits = np.asarray(logits, dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if logits.shape[1] != 3:
            raise ValueError(f"Expected 3 NLI logits, received shape {logits.shape}")

        probs = softmax(logits / self.temperature, axis=1)
        results = []
        for row_logits, row_probs in zip(logits, probs):
            scores = {
                label: float(row_probs[class_id])
                for label, class_id in self.label_ids.items()
            }
            predicted = max(scores, key=scores.get)
            results.append({
                "prediction": predicted,
                "scores": scores,
                "logits": [float(x) for x in row_logits],
            })
        return results
