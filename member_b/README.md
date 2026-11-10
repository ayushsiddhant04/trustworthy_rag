# Member B — Verification Engine

## Google Colab quick start
1. Open the supplied `Member2_Verification_Engine_Colab.ipynb` in Google Colab.
2. Select **Runtime → Run all**. The first run downloads the NLI model and can take a few minutes.
3. The notebook creates `member_b/`, writes each Python file, runs a sample verification, executes tests, and creates `member_b_starter.zip`.

## Main public interface

```python
from member_b.verifier import VerificationEngine
engine = VerificationEngine()
result = engine.verify_claim(
    "The application deadline is 20 November.",
    [{"chunk_id": "chunk_1", "text": "The application deadline is 15 November.", "page": 3, "source": "notice.pdf"}]
)
```

For the group pipeline, pass a list of claims and evidence to `verify_claims`.

## Files
- `nli_model.py`: pretrained NLI cross-encoder and class score conversion
- `windowing.py`: token-aware overlapping evidence windows
- `aggregation.py`: supported / contradicted / insufficient / conflicting aggregation
- `numeric_checks.py`: conservative, context-cue-based number/date mismatch heuristic
- `decision_policy.py`: state / caveat / withhold decisions
- `calibration.py`: temperature scaling helpers
- `verifier.py`: end-to-end claim verification
- `benchmark.py`: labelled CSV benchmark and metrics

## Important research cautions
- The initial thresholds in `config.py` are placeholders. Choose them on a validation set, never on the final test set.
- A 60-row benchmark is a sanity check, not sufficient evidence for a broad generalization claim.
- FEVER's NOT ENOUGH INFO examples do not come with ordinary gold supporting evidence. Create and manually audit candidate evidence pairs for the insufficient class.
- Numeric checks are heuristics. A mismatch is a warning requiring careful interpretation, not a universal proof of contradiction.
- Pairwise NLI scores are not automatically calibrated claim-level confidence. Calibrate using a separate labelled set and audit claim-level behaviour.
