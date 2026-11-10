"""Configuration for Member 2's verification engine."""
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 512
WINDOW_SIZE = 240          # evidence tokens; leaves room for the claim
WINDOW_OVERLAP = 50
DEFAULT_ENTAILMENT_THRESHOLD = 0.60  # pilot starting point; tune on validation data
DEFAULT_CONTRADICTION_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.80

# The official model card documents this output order for this specific model.
# nli_model.py first tries to read the model's id2label mapping.
KNOWN_MODEL_LABEL_ORDER = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}
