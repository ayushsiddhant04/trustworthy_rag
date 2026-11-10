# Evidence Review: System Architecture, Research Foundations & Project Overview

**Project Name**: Evidence Review — Document Intelligence & Automated Claim Verification  
**Domain**: Natural Language Processing, Document Intelligence, Natural Language Inference (NLI)  
**Primary Framework**: Python 3.12, PyTorch, Hugging Face Transformers, Streamlit  
**Model**: `cross-encoder/nli-deberta-v3-base` (DeBERTa-v3 cross-encoder)  

---

## 1. Executive Summary & Problem Statement

Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) architectures are increasingly deployed in high-stakes institutional settings such as academic admissions, corporate policy enforcement, legal compliance, and procurement audits. However, standard RAG systems suffer from critical reliability hazards:
1. **Semantic Masking of Quantitative Errors**: Neural embeddings and generative models often confuse subtle numerical changes (e.g., *750 rupees* vs. *1500 rupees* or *15 November* vs. *30 November*) because the surrounding linguistic context is identical, leading to undetected hallucinations.
2. **Compound Sentence Conflation**: When an AI-generated answer asserts multiple facts within a single sentence (e.g., *"The registration fee is 500 rupees and the deadline is 30 November"*), an error in one fact invalidates the entire response or, worse, passes unnoticed if one component is supported.
3. **Cross-Document Conflicts**: Institutional information is rarely static; amended circulars, overlapping directives, and conflicting departmental notices coexist in active repositories. Standard single-passage retrieval models cannot arbitrate or communicate mutual institutional contradictions to users.
4. **Overconfident Probability Spikes**: Deep neural cross-encoders typically exhibit extreme overconfidence (>0.99 softmax scores), failing to reflect true epistemic uncertainty.

**Evidence Review** addresses these challenges through a modular, verifiable pipeline pairing multi-format document ingestion and hybrid retrieval with atomic claim extraction, sliding-window NLI cross-encoders, multi-passage evidence aggregation, deterministic date/numeric field checks, temperature-scaled calibration, and conservative safe answer composition.

---

## 2. Why Document-Grounded AI Answers Can Be Unreliable

Even when an AI model is restricted to retrieved document excerpts, three fundamental failure modes persist:
- **Retrieval Bias & Distractor Noise**: Top-$k$ similarity metrics often retrieve irrelevant background passages that contain matching keywords but lack factual relevance, diluting verifier attention.
- **Single-Source Inadequacy**: Standard verification benchmarks evaluate claims against a single contiguous text excerpt. In authentic institutional workflows, establishing truth or identifying contradictions requires synthesizing evidence across multiple pages or separate notices.
- **Unconditional Generation**: Standard generative RAG systems are forced to output a fluent answer even when evidence is completely absent or partially contradictory, fabricating plausible bridges between disconnected statements.

---

## 3. Existing Approaches and Addressed Limitations

| Existing Approach | Operational Mechanism | Fundamental Limitation Addressed by Evidence Review |
|---|---|---|
| **Vanilla RAG (LLM-only)** | Prompts an LLM with retrieved context to generate answers directly. | Lacks claim-level auditability; LLM cannot be deterministically verified against hallucinations. |
| **Cosine Embedding Match** | Measures semantic vector distance between query and text. | Symmetric similarity cannot distinguish logical *entailment* from *contradiction* or *neutrality*. |
| **Single-Pair NLI** | Evaluates a single concatenated passage against the whole answer. | Fails on compound sentences; completely blind to cross-document institutional conflicts. |
| **Heuristic Fact-Checking** | Regular expression matching or entity overlap. | Fails on paraphrasing, passive voice, synonymy, and complex grammatical transformations. |

---

## 4. Proposed Solution & Architecture

Evidence Review introduces a three-tiered defense:
1. **Decomposition**: Complex candidate answers are split across finite-verb boundaries into atomic, single-proposition claims.
2. **Multi-Passage Windowed NLI with Calibration**: Each atomic claim is evaluated across sliding windows (240 tokens with 50-token overlap) over all retrieved passages using a DeBERTa-v3 cross-encoder calibrated via temperature scaling ($T = 2.6888$).
3. **Conservative Decision Policy**: A four-status classification (`supported`, `contradicted`, `insufficient`, `conflicting`) is mapped into three safe user actions:
   - `state`: Assert claim with exact page and paragraph citations (requires verification confidence $\ge 0.80$).
   - `caveat`: Alert user to institutional conflict across legitimate notices.
   - `withhold`: Completely suppress contradicted or unverified assertions.

---

## 5. System Architecture

```
[Institutional Documents (PDF / DOCX / TXT)]
                     │
                     ▼
             [Member A: Ingestion]
                     │ (Section & Sliding Window Chunking, 120 words / 30 overlap)
                     ▼
          [Member A: Hybrid Retrieval] ────► [BM25 Sparse + MiniLM Dense]
                     │
                     ▼
        [Candidate Answer Generation] ────► [Extractive or User/LLM Answer]
                     │
                     ▼
       [Member A: Claim Extraction] ───► [Finite-Verb Atomic Splitter]
                     │
            (Atomic Claims + Candidate Passages)
                     │
                     ▼
        [Member B: Verification Engine]
         ├── Relevance Gating (TF-IDF Cosine Filter, Optional)
         ├── 240-Token Evidence Windowing (50-token overlap)
         ├── DeBERTa-v3 NLI Cross-Encoder (Offline air-gapped cache)
         ├── Temperature Scaling Calibration (T = 2.6888)
         ├── Multi-Passage Evidence Aggregation (Entailment / Contradiction Pooling)
         └── Context-Aware Numeric & Date Checks (5 Disjoint Semantic Fields)
                     │
              (Status & Action Metadata)
                     │
                     ▼
         [Member C: Safe Answer Composer]
         ├── 5-State Application Verdict Hierarchy:
         │   ├── answer (supported only)
         │   ├── answer_with_caveats (supported + caveated)
         │   ├── partial_answer (supported + withheld)
         │   ├── partial_answer_with_caveats (supported + caveated + withheld)
         │   └── abstain (all withheld)
         └── Provenance & Audit Exporter (RFC 4180 CSV / JSON)
                     │
                     ▼
       [Member C: Streamlit Web Interface]
```

---

## 6. Modular Division of Labor & Team Contributions

### Member A: Document Ingestion, Retrieval & Claim Orchestration (`src/`)
- Multi-format document parsers for PDF (PyMuPDF), DOCX (python-docx), and TXT with strict file size ($\le 25\text{ MB}$) and page ($\le 100\text{ pages}$) safeguards.
- Sliding-window chunker preserving full document provenance (source filename, page number, paragraph index, and unique chunk IDs).
- Hybrid retrieval pairing lexical BM25 sparse search with optional dense embedding retrieval (`all-MiniLM-L6-v2`).
- Extractive answer generation with explicit `no_evidence` abstention metadata when retrieval falls below threshold.
- Atomic claim extraction (`split_compound_clause`) identifying independent factual clauses using finite-verb checks across coordinating conjunctions.

### Member B: Verification Engine, Calibration & Benchmarking (`member_b/`)
- Token sliding-window segmenter (240 tokens, 50-token overlap) adapting long passages to DeBERTa-v3-base's 512-token context window.
- Multi-passage evidence pooling (`aggregate_evidence`) tracking independent passage entailment and contradiction scores to arbitrate mutual contradictions.
- Context-aware numeric/date verification (`numeric_checks.py`) isolating 5 disjoint semantic fields (date, fee, percentage, duration, count) with calendar year protection.
- Temperature scaling calibration (`calibration.py`) optimizing multiclass cross-entropy log-loss ($T = 2.6888$).
- Research benchmarking suite (`experiments.py`, `benchmark.py`) computing document-clustered bootstrap confidence intervals (95% CI).

### Member C: User Interface, Safe Composition & Integration (`src/ui/`, `app.py`, `src/safe_composer.py`)
- Conservative safe answer composition synthesizing final answers strictly from verified claims, appending caveats for multi-source disputes, and cleanly excising unverified assertions.
- Professional light-theme Streamlit interface across four dedicated navigation sections:
  1. *Document Review*: Full ingestion, extraction, verification, and audit trail export.
  2. *Research Evaluation*: Live benchmark runner, confusion matrices, controlled baseline comparisons, fair numeric ablation, and calibration analysis.
  3. *Demo Centre*: Interactive showcase of three synthetic scenarios.
  4. *Methodology*: Complete technical specifications, decision matrices, and scientific boundaries.

---

## 7. Technology Stack

- **Core Runtime**: Python 3.12.10
- **Natural Language Inference**: `cross-encoder/nli-deberta-v3-base` (DeBERTa-v3 cross-encoder)
- **Dense Retrieval**: `sentence-transformers/all-MiniLM-L6-v2`
- **Document Processing**: PyMuPDF (`fitz` 1.25.3), `python-docx` (1.1.2)
- **Lexical Retrieval & ML**: Scikit-Learn 1.6.1, NumPy 2.2.3, SciPy 1.15.2
- **User Interface**: Streamlit 1.42.2
- **Testing & Quality Assurance**: PyTest 9.2.0 (92 passing tests)

---

## 8. Experimental Methodology

- **Canonical Evaluation Partitioning**: Grouped strictly at the document level across 15 synthetic institutional document domains (180 total claims).
  - *Calibration Split*: 36 claims (46 passage pairs) across 3 domains.
  - *Validation Split*: 24 claims across 2 domains.
  - *Held-Out Test Split*: 120 claims (150 passage pairs) across 10 unseen domains (balanced: 30 supported, 30 contradicted, 30 insufficient, 30 conflicting).
  - *Zero Document Leakage*: $\text{Docs}(\text{Calib}) \cap \text{Docs}(\text{Val}) \cap \text{Docs}(\text{Test}) = \emptyset$.
- **Statistical Resampling**: 500-fold bootstrap resampling clustered at the document group level ($K = 10$ held-out clusters) to produce 95% confidence intervals.

---

## 9. Key Measured Findings (Held-Out Test Split, $N=120$)

1. **Multi-Passage Aggregation (RQ1)**:
   - Raw single-passage NLI fails completely on conflicting evidence ($F_1 = 0.000$, $\text{FAR} = 22.2\%$), misclassifying 20 of 30 conflicting claims as supported.
   - Multi-passage aggregation enables 4-class conflict detection ($F_1 = 0.800$, Precision = 100.0%) and lifts overall accuracy from **0.725 to 0.892** (+16.7 percentage points) and Macro F1 from **0.626 to 0.889**.
2. **Fair Numeric/Date Ablation (RQ2)**:
   - When all pipeline components (windowing, aggregation, temperature scaling, decision policy) are held identical, Configuration A (no numeric checks) and Configuration B (with numeric checks) yield identical macro metrics ($0.892$ accuracy, $0.111$ FAR) on broad test data.
   - The primary contribution of numeric checks is **deterministic prevention of semantic masking** on targeted near-miss numerical perturbations and compound claims, as verified on 8 controlled scenarios.
3. **Probability Calibration (RQ4)**:
   - Calibrated temperature scaling ($T = 2.6888$) reduced negative log-likelihood (NLL) by **43.8%** on calibration ($0.7926 \rightarrow 0.4455$) and **19.2%** out-of-domain on held-out test data ($0.3623 \rightarrow 0.2927$), successfully mitigating DeBERTa's overconfidence spikes.
4. **Relevance Gating Trade-Off (RQ3)**:
   - Lexical TF-IDF gating increases precision on contradicted claims ($0.91 \rightarrow 0.98$), but risks pruning paraphrased contrasting passages, reducing conflicting recall from $66.7\%$ to $53.3\%$. Confirmed as an optional feature.

---

## 10. Documented Limitations & Future Scope

1. **Synthetic Document Scope**: The evaluation suite evaluates synthetic institutional texts emulating university circulars. It does not evaluate handwritten text, scanned OCR images, clinical medical records, or financial tables.
2. **Small-Cluster Bootstrap Sample**: With $K = 10$ held-out document clusters, bootstrap confidence intervals are approximate cluster-robust estimates rather than asymptotic guarantees (which typically require $K \ge 30-50$).
3. **Multi-Hop Arithmetic Reasoning**: The system validates explicit numerical values within matched semantic fields; it does not perform multi-step arithmetic (e.g., computing net fees by summing items across tables).
4. **Temporal Document Supersession**: Resolving whether Notice B nullifies Notice A requires document publication timestamps, which NLI models cannot deduce from semantics alone.

