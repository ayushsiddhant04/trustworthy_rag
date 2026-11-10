# Integration contract: Member A pipeline → Member B verifier → Member C UI

## Ownership
- `src/` is the reconstructed Member A ingestion/retrieval/generation/claim/pipeline implementation.
- `member_b/` preserves Member B's submitted verification engine and benchmark modules.
- `app.py` and `src/verification_adapter.py` are Member C's application/integration layer.

## Pipeline entry point

```python
from src.pipeline import run
result = run(question, doc, answer=None, mode="generate", top_k=3)
```

`doc` can be a local path, bytes, an upload-like object with `.name` and `.read()`, or an already-ingested dictionary.

## Canonical per-claim input

```json
{
  "claim_id": "C1",
  "claim": "The application deadline is 20 November 2026.",
  "evidence": [
    {
      "chunk_id": "chunk_abc123",
      "text": "The last date for submitting the application is 15 November 2026.",
      "page": 1,
      "source": "application_notice.pdf",
      "score": 0.91,
      "paragraph": null
    }
  ]
}
```

For backward compatibility, `passages` (text-only) and `passage_refs` (metadata) are also returned.

## Member B call

The adapter calls `VerificationEngine.verify_claim(claim, evidence)` once per claim. This is deliberate: Member A retrieval is run separately for each claim so each claim gets its own top evidence passages.

```python
from src.verification_adapter import verify_pipeline_output
verified = verify_pipeline_output(result)  # loads member_b lazily
```

The returned dictionary adds `verification_results` and `overall_verdict` without mutating the pipeline input.

## Safe composer
- `action=state` AND `status=supported`: state the claim.
- `action=caveat`: include with explicit uncertainty language.
- `action=withhold`: exclude from the verified answer.
- If no claim survives, return an explicit insufficient-evidence message.

## Important interface decisions
- Status values: `supported`, `contradicted`, `insufficient`, `conflicting`.
- Action values: `state`, `caveat`, `withhold`.
- Confidence is the verifier's current score, not a calibrated probability until Member B completes calibration.
- PDF page numbers are 1-based. TXT and DOCX use paragraph/source references where page number is unavailable.
