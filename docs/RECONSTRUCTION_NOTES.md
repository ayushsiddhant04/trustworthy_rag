# Reconstruction notes and known boundaries

## Why this is a reconstruction

The uploaded Member A archives contained README/configuration/setup material but not the `src/` backend implementation. This project therefore rebuilds Member A's layer from the published requirements and function descriptions. It must not be represented as Member A's recovered original code.

Member B's `.py` modules, tests, benchmark inputs and result artifacts were copied from the supplied Member B submission. A package initializer was added so the modules can be imported as `member_b.*`.

## Implemented by this reconstruction

- PDF/TXT/DOCX text ingestion; empty PDF pages are skipped with warnings and all-text-empty PDFs raise a clear scanned-PDF error.
- 120-word chunks with 30-word overlap (configurable), retaining source, PDF page, paragraph and chunk IDs.
- TF-IDF retrieval by default; optional all-MiniLM-L6-v2 hybrid retrieval with corpus/query embedding caching and TF-IDF fallback.
- Manual answer, explicitly pasted known-bad demo answer, optional grounded Anthropic generation, and an offline extractive answer fallback.
- Sentence-splitting claim fallback and optional LLM JSON claim decomposition with safe JSON parsing.
- Per-claim top-k retrieval and a unified evidence object compatible with Member B's `verify_claim(claim, evidence_passages)`.
- Thread-safe TTL/LRU cache for chunk and LLM intermediates.
- Member B adapter, safe composer and Streamlit application.

## Decisions that need team sign-off

1. Chunk overlap: set to 30 words by default because A's `.env.example` specified 30, while one README example mentioned 20. This is configurable.
2. Member A's docs split passage text and refs into separate arrays. This implementation adds a canonical `evidence` array of dicts and keeps `passages`/`passage_refs` for compatibility.
3. Each claim is independently retrieved and passed to Member B. This improves evidence alignment compared with sending one shared passage list to every claim.
4. `demo_bad` requires the operator to paste an incorrect answer; the system does not fabricate a wrong answer automatically.
5. External LLM calls and embedding retrieval are opt-in. This preserves the stated privacy/offline default.

## Member B behaviour to review

- The submitted verifier reports `status="conflicting"` when passages support and contradict a claim. However, when its numeric/date heuristic also reports a mismatch, `decision_policy.decide()` takes precedence and sets `action="withhold"`. This is the current submitted policy and has been preserved. The team should explicitly decide whether conflicting multi-notice cases should instead be `caveat` while showing both passages.
- The submitted `relevance.py` exists, but `VerificationEngine.verify_claim()` does not invoke `RelevanceChecker`. Its reported relevance-gated pilot results cannot be reproduced from the package alone without the notebook/gating implementation.
- The submitted thresholds are preliminary; its calibration helper is not connected to a persisted calibration workflow by default.
- The submitted benchmark CSV contract evaluates one evidence text per row. Multi-passage conflict tests require a separate benchmark extension.

## Verification performed in this runtime

- Core/reconstructed project, evaluation, and hardening tests: 65 passed.
- Member B verifier helper, numeric checks, relevance, and windowing tests: 14 passed.
- Total test coverage: 79 passed, 0 failed, 0 skipped.
- Real model inference validated: `cross-encoder/nli-deberta-v3-base` running locally in offline mode (`HF_HUB_OFFLINE=1`).
- Streamlit application validated across all four navigation sections: Document Review, Research Evaluation, Demo Centre, and Methodology.
- Date normalization, context-aware semantic field comparisons, and atomic factual claim extraction validated end-to-end.
