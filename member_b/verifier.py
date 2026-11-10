"""End-to-end claim-level verifier used by Members A and C."""
try:
    from .config import MODEL_NAME, DEFAULT_ENTAILMENT_THRESHOLD, DEFAULT_CONTRADICTION_THRESHOLD
    from .nli_model import NLIEngine
    from .windowing import load_tokenizer, split_into_windows
    from .aggregation import aggregate_evidence
    from .numeric_checks import check_numeric_conflict
    from .decision_policy import decide
    from .calibration import load_temperature
except ImportError:  # allow notebook imports with /content/member_b on sys.path
    from config import MODEL_NAME, DEFAULT_ENTAILMENT_THRESHOLD, DEFAULT_CONTRADICTION_THRESHOLD
    from nli_model import NLIEngine
    from windowing import load_tokenizer, split_into_windows
    from aggregation import aggregate_evidence
    from numeric_checks import check_numeric_conflict
    from decision_policy import decide
    from calibration import load_temperature


class VerificationEngine:
    def __init__(self, model_name=MODEL_NAME, nli=None, tokenizer=None,
                 entailment_threshold=DEFAULT_ENTAILMENT_THRESHOLD,
                 contradiction_threshold=DEFAULT_CONTRADICTION_THRESHOLD,
                 window_size=None, overlap=None,
                 temperature=None, calibration_file=None,
                 enable_numeric_checks: bool = True):
        self.model_name = model_name
        self.enable_numeric_checks = bool(enable_numeric_checks)
        if calibration_file is not None and temperature is None:
            temperature = load_temperature(calibration_file)
        if nli is not None:
            self.nli = nli
        elif temperature is not None:
            self.nli = NLIEngine(model_name=model_name, temperature=temperature)
        else:
            self.nli = NLIEngine(model_name=model_name)
        self.tokenizer = tokenizer or load_tokenizer(model_name)
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.window_size = window_size
        self.overlap = overlap

    @property
    def is_calibrated(self) -> bool:
        return getattr(self.nli, "is_calibrated", False)

    @property
    def temperature(self) -> float:
        return getattr(self.nli, "temperature", 1.0)

    @temperature.setter
    def temperature(self, value: float) -> None:
        if self.nli is not None:
            self.nli.temperature = float(value)
            self.nli.is_calibrated = float(value) != 1.0

    def verify_claim(self, claim, evidence_passages):
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        if isinstance(evidence_passages, str):
            evidence_passages = [{"text": evidence_passages}]
        evidence_passages = evidence_passages or []

        normalized = []
        all_pairs = []
        pair_metadata = []
        for idx, item in enumerate(evidence_passages):
            if isinstance(item, str):
                item = {"text": item}
            text = (item.get("text") or "").strip()
            if not text:
                continue
            chunk_id = str(item.get("chunk_id", f"chunk_{idx+1}"))
            windows = split_into_windows(
                text,
                self.tokenizer,
                **({"window_size": self.window_size} if self.window_size else {}),
                **({"overlap": self.overlap} if self.overlap is not None else {}),
            )
            passage = {
                "chunk_id": chunk_id,
                "page": item.get("page"),
                "source": item.get("source"),
                "text": text,
                "windows": windows,
            }
            normalized.append(passage)
            for window in windows:
                all_pairs.append((window["text"], claim))  # premise first, hypothesis second
                pair_metadata.append((len(normalized) - 1, window))

        pair_predictions = self.nli.predict_pairs(all_pairs)
        per_passage = []
        all_window_results = []
        for passage_index, passage in enumerate(normalized):
            rows = []
            for pred, (meta_passage_index, window) in zip(pair_predictions, pair_metadata):
                if meta_passage_index == passage_index:
                    row = {
                        "window": window,
                        "scores": pred["scores"],
                        "logits": pred["logits"],
                    }
                    rows.append(row)
                    all_window_results.append({
                        "chunk_id": passage["chunk_id"],
                        "page": passage["page"],
                        "source": passage["source"],
                        "window_text": window["text"],
                        "scores": pred["scores"],
                    })
            best_entail = max(rows, key=lambda r: r["scores"]["entailment"], default=None)
            best_contra = max(rows, key=lambda r: r["scores"]["contradiction"], default=None)
            best_neutral = max(rows, key=lambda r: r["scores"]["neutral"], default=None)
            per_passage.append({
                "chunk_id": passage["chunk_id"],
                "page": passage["page"],
                "source": passage["source"],
                "text": passage["text"],
                "max_entailment": best_entail["scores"]["entailment"] if best_entail else 0.0,
                "max_contradiction": best_contra["scores"]["contradiction"] if best_contra else 0.0,
                "max_neutral": best_neutral["scores"]["neutral"] if best_neutral else 0.0,
                "best_entailment_window": best_entail["window"]["text"] if best_entail else None,
                "best_contradiction_window": best_contra["window"]["text"] if best_contra else None,
            })

        aggregate = aggregate_evidence(
            per_passage,
            entail_threshold=self.entailment_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )
        if getattr(self, "enable_numeric_checks", True):
            numeric = check_numeric_conflict(claim, normalized)
        else:
            numeric = {
                "numeric_conflict": False,
                "reason": "Numeric checks disabled",
                "claim_values": [],
                "evidence_values": [],
                "matched_fields": [],
                "evidence_chunk_id": None,
            }
        status = aggregate["status"]
        # A matched contextual value mismatch is evidence against the exact claim.
        # Keep a pre-rule status for audit. If other passages support it too, retain conflict.
        status_before_numeric_rule = status
        if numeric["numeric_conflict"]:
            if status == "supported":
                status = "contradicted"
            elif status == "conflicting":
                status = "conflicting"

        action = decide(status, aggregate["confidence"], numeric["numeric_conflict"])
        relevant_evidence = []
        for p in per_passage:
            if p["max_entailment"] >= self.entailment_threshold:
                relevant_evidence.append({
                    "chunk_id": p["chunk_id"], "page": p["page"], "source": p["source"],
                    "relation": "entailment", "text": p["best_entailment_window"],
                    "score": p["max_entailment"],
                })
            if p["max_contradiction"] >= self.contradiction_threshold:
                relevant_evidence.append({
                    "chunk_id": p["chunk_id"], "page": p["page"], "source": p["source"],
                    "relation": "contradiction", "text": p["best_contradiction_window"],
                    "score": p["max_contradiction"],
                })

        if status == "conflicting" and numeric["numeric_conflict"]:
            reason = f"{action['reason']} {numeric['reason']}".strip()
        else:
            reason = numeric["reason"] if numeric["numeric_conflict"] else action["reason"]
        return {
            "claim": claim,
            "status": status,
            "status_before_numeric_rule": status_before_numeric_rule,
            "confidence": float(aggregate["confidence"]),
            "action": action["action"],
            "reason": reason,
            "numeric_check": numeric,
            "supporting_passages": aggregate["supporting_passages"],
            "contradicting_passages": aggregate["contradicting_passages"],
            "evidence": relevant_evidence,
            "passage_scores": per_passage,
            "window_scores": all_window_results,
        }

    def verify_claims(self, claims, evidence_passages):
        results = []
        for idx, claim_item in enumerate(claims):
            if isinstance(claim_item, str):
                claim_id, claim_text = f"claim_{idx+1}", claim_item
            else:
                claim_id = claim_item.get("claim_id", f"claim_{idx+1}")
                claim_text = claim_item.get("text", "")
            result = self.verify_claim(claim_text, evidence_passages)
            result["claim_id"] = claim_id
            results.append(result)
        return results
