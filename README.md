---
title: Evidence Review
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Evidence Review — Document Intelligence & Claim Verification

Evidence Review is an institutional document intelligence and automated claim verification system. It pairs document ingestion and hybrid retrieval with natural language inference (NLI) cross-encoders, multi-passage evidence aggregation, deterministic date/numeric conflict detection, temperature-scaled calibration, and conservative safe answer composition.

---

## 1. Modular Architecture

The application is structured into three cooperating modules:

- **Member A (`src/`)**: Multi-format document ingestion (PDF, DOCX, TXT), sectioning and sliding-window chunking, hybrid sparse (BM25) + dense (MiniLM) retrieval, extractive answer generation with offline abstention, and factual claim decomposition.
- **Member B (`member_b/`)**: Token windowing, DeBERTa-v3 NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`), multi-passage evidence pooling, deterministic numeric and date conflict detection, temperature-scaling calibration, and empirical benchmarking.
- **Member C (`src/ui/`, `app.py`, `src/safe_composer.py`)**: Professional light-theme UI, conservative safe answer composition (`state` / `caveat` / `withhold`), evidence provenance cards, research evaluation dashboard, and interactive demo centre.

### Module Map

```
trustworthy_rag_rebuild/
├── app.py                         # Master Streamlit application entrypoint & router
├── src/
│   ├── ingestion.py               # PDF/DOCX/TXT parsers with 25 MB & 100-page limits
│   ├── chunking.py                # Provenance-preserving sliding window chunker
│   ├── retrieval.py               # Hybrid BM25 + dense embedding retrieval
│   ├── answer_generation.py       # Extractive fallback with explicit no-evidence status
│   ├── claim_extraction.py        # Factual propositional claim extractor
│   ├── pipeline.py                # End-to-end ingestion-to-claim orchestration
│   ├── verification_adapter.py    # Bridge between Member A and Member B verifier
│   ├── safe_composer.py           # Conservative decision policy & answer composer
│   └── ui/                        # Modular Streamlit views & design system
│       ├── styles.py              # Design system tokens and CSS rules
│       ├── components.py          # Badges, citations, empty states, audit exporter
│       ├── view_review.py         # Section 1: Document Review
│       ├── view_evaluation.py     # Section 2: Research Evaluation
│       ├── view_demo.py           # Section 3: Demo Centre
│       └── view_methodology.py    # Section 4: Methodology & Limitations
├── member_b/
│   ├── nli_model.py               # DeBERTa-v3 cross-encoder wrapper & temperature scaling
│   ├── windowing.py               # 128-token sliding window tokenizer
│   ├── aggregation.py             # Multi-passage evidence pooling logic
│   ├── numeric_checks.py          # Contextual numeric and date discrepancy parser
│   ├── decision_policy.py         # 4-status to 3-action mapping rules
│   ├── calibration.py             # ECE, Brier, NLL metrics & temperature fitting
│   ├── benchmark.py               # Multi-passage benchmark evaluator
│   ├── experiments.py             # 5-baseline controlled experiment runner
│   ├── relevance.py               # Sentence-Transformers cosine similarity gate
│   ├── data/                      # Segregated calibration & multi-passage evaluation sets
│   └── results/                   # Benchmark artifacts & smoke test records
└── tests/                         # Automated unit & integration tests (92 tests total)
```

---

## 2. Installation & Environment Setup

Python 3.12 (specifically 3.12.10) is recommended.

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install core application dependencies
python -m pip install -r requirements.txt

# 3. Install verification & NLI dependencies
python -m pip install -r requirements-verification.txt
```

### Pre-cached Model Weights
The repository supports air-gapped / offline execution by pre-caching the HuggingFace model checkpoints:
- Cross-encoder: `cross-encoder/nli-deberta-v3-base`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

To run in offline mode:
```bash
set HF_HUB_OFFLINE=1
```

---

## 3. Running the Application

Launch the Streamlit interface:

```bash
run_app.bat
# Or directly:
.venv\Scripts\python.exe -m streamlit run app.py
```

The application is organized into four distinct sections:
1. **Document Review**: Ingest native documents, generate/audit answers, extract claims, execute NLI verification, inspect evidence provenance, and export audit trails.
2. **Research Evaluation**: Evaluate accuracy, FAR, coverage, and latency across multi-passage datasets, run 5 controlled baseline experiments, view $4 \times 4$ confusion matrices, and review temperature-scaling calibration reports.
3. **Demo Centre**: Test three synthetic scenarios showcasing numeric conflict detection, missing evidence abstention, and conflicting circular arbitration.
4. **Methodology**: Detailed architectural specifications, decision policy matrices, and documented scientific boundaries (no OCR, confidence vs. probability distinction, retrieval recall ceilings).

---

## 4. Running the Test Harness

Execute all **92 passing tests** across both suites:

```bash
run_tests.bat
```

To run the suites individually:

```bash
# Core pipeline, integration, research validation, and numeric ablation tests (78 tests)
.venv\Scripts\python.exe -m pytest tests -q

# Member B verifier, numeric checks, and relevance tests (14 tests)
set PYTHONPATH=member_b
.venv\Scripts\python.exe -m pytest member_b\tests -q
```

---

## 5. Verification Policy & Decision Matrix

| Verification Status | NLI & Evidence Condition | Safe Policy Action | Composed Answer Behavior |
|---|---|---|---|
| **Supported** | Max Entailment $\ge 0.50$ and Contradiction $< 0.50$; zero numeric mismatches. | **State** | Asserted normally with exact page/paragraph provenance citations. |
| **Contradicted** | Max Contradiction $\ge 0.50$ or contextual date/numeric mismatch. | **Withhold** | Suppressed entirely with an explicit contradiction reason provided. |
| **Conflicting** | Multiple authentic passages directly disagree on a fact or date. | **Caveat** | Reported with an explicit caveat alerting the user to the institutional disagreement. |
| **Insufficient** | Neither entailment nor contradiction meets confidence thresholds. | **Withhold** | Withheld due to lack of ground truth support. |

---

## 6. Research & Empirical Integrity

- **Strict Data Partitioning**: Demonstration files in `data/demo/` are strictly isolated from benchmarks. Calibration data (`member_b/data/calibration_split.csv`) is fitted independently and never leaks into held-out test sets (`member_b/data/test_split.csv`).
- **Uncalibrated Confidence vs. Calibrated Probabilities**: Softmax confidence scores are only treated as calibrated probabilities when temperature scaling ($T = 2.6888$) is loaded from `member_b/results/calibration_canonical_artifact.json`.
- **Zero Hallucination Guarantee**: If retrieved evidence is missing or below threshold, the extractive pipeline issues an explicit `no_evidence` status and abstains from manufacturing ungrounded claims.
- **Fair Numeric Ablation**: Isolates pure date/numeric rules from confidence thresholding, establishing exact parity across identical aggregation and decision policies.
