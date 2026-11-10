# 10-Slide Competition Presentation Deck Outline

**Presentation Title**: Evidence Review — Document Intelligence & Automated Claim Verification  
**Target Duration**: 10–12 Minutes (Competition Presentation) or 3 Minutes (Pitch Format)  
**Presenter**: Student Engineering Team / System Architect  
**Track**: Applied Artificial Intelligence, Document Intelligence & Trustworthy NLP  

---

### Slide 1: Title & Executive Vision
- **Header**: Evidence Review: Document Intelligence & Automated Claim Verification
- **Subtitle**: Preventing Quantitative Hallucinations and Multi-Notice Conflicts in Institutional RAG Systems
- **Team**: Engineering Team (Members A, B, and C)
- **Visuals**: Product logo, hero screenshot of the Streamlit Document Review interface with citation badges and evidence cards.
- **Key Message**: Transforming brittle generative question-answering into auditable, accountable, and conservative institutional intelligence.

---

### Slide 2: The Core Problem in Institutional RAG
- **Header**: Why Document-Grounded AI Fails in High-Stakes Environments
- **Core Pain Points**:
  - *Semantic Masking*: Neural models overlook critical quantitative errors (*750 rupees* vs *1500 rupees*) because grammatical context is identical.
  - *Compound Conflation*: Multiple facts in one sentence prevent partial error detection.
  - *Institutional Disagreements*: Coexisting circulars and amended policies cannot be arbitrated by single-passage similarity.
  - *Extreme Overconfidence*: Deep networks output >0.99 confidence on hallucinated predictions.
- **Visuals**: A side-by-side callout showing an AI asserting an incorrect admissions deadline alongside a correct fee, with a red alert highlighting the undetected error.

---

### Slide 3: Existing Approaches & Why They Fall Short
- **Header**: Limitations of Existing Solutions
- **Comparison Table**:
  - *Vanilla LLM Prompting*: Fluent, but hallucinations are stochastic and uncalibrated.
  - *Cosine Vector Similarity*: Symmetric; cannot distinguish logical contradiction from entailment.
  - *Single-Pair NLI (FEVER-style)*: Fails on compound sentences; blind to multi-notice conflicts.
  - *Heuristic Keyword Matching*: Brittle to paraphrase, active/passive voice, and administrative synonyms.
- **Visual**: A 2x2 matrix: Semantic Depth vs. Verification Auditability, placing Evidence Review in the top-right quadrant.

---

### Slide 4: Proposed System Architecture
- **Header**: End-to-End Three-Module Pipeline
- **Diagram**: Flowchart from input documents to final safe answer:
  - *Member A*: Ingestion (PDF/DOCX/TXT) $\rightarrow$ Sliding Chunking (120w/30w) $\rightarrow$ Hybrid Retrieval (BM25 + MiniLM) $\rightarrow$ Atomic Claim Decomposition.
  - *Member B*: Token Windowing (240/50) $\rightarrow$ DeBERTa-v3 Cross-Encoder $\rightarrow$ Multi-Passage Pooling $\rightarrow$ Contextual Numeric Engine.
  - *Member C*: Temperature Calibration $\rightarrow$ Decision Policy $\rightarrow$ Conservative Safe Answer Composition.
- **Visual**: High-level architectural flowchart highlighting modular separation and air-gapped offline capability.

---

### Slide 5: Document Ingestion, Retrieval & Chunking
- **Header**: Provenance-Preserving Ingestion & Hybrid Retrieval
- **Key Specifications**:
  - *Multi-Format Parsers*: PyMuPDF for PDF, `python-docx` for Word, plain text with file-size ($\le 25\text{ MB}$) and page ($\le 100\text{ p}$) validation.
  - *Provenance Retention*: Chunks retain source document, page number, paragraph index, and unique UUID chunk IDs.
  - *Hybrid Sparse + Dense*: Lexical BM25 ensures exact code/date matching; dense MiniLM embeddings capture semantic paraphrasing.
  - *Explicit Abstention*: If retrieved evidence is below relevance threshold, the pipeline sets `no_evidence` status rather than manufacturing ungrounded text.
- **Visual**: Diagram showing a PDF page sliced into overlapping chunks, tracking page and paragraph tags through the retrieval index.

---

### Slide 6: Atomic Claim Decomposition & Numeric Verification
- **Header**: Atomic Proposition Splitting & Disjoint Semantic Fields
- **Breakthrough Mechanisms**:
  - *Atomic Extraction*: Finite-verb coordinating conjunction splitting separates compound answers:
    - Candidate: *"The fee is 500 rupees and the deadline is 30 November."*
    - Claim 1: *"The fee is 500 rupees."* (Evaluated independently)
    - Claim 2: *"The deadline is 30 November."* (Evaluated independently)
  - *Context-Aware Numeric Parser*: Categorizes values into 5 disjoint fields (date, fee, percentage, duration, count). Calendar years are consumed by date parsers, preventing false alarms on numbers like "2026".
- **Visual**: Screenshot of Scenario 1 output from Demo Centre showing Claim 1 contradicted and Claim 2 supported.

---

### Slide 7: Decision Policy & Safe Answer Composition
- **Header**: From Verification Status to Operational User Action
- **Taxonomy Table**:

| Status | Verification Condition | Action | Application Verdict | User-Facing Behavior |
|---|---|---|---|---|
| **Supported** | Max Entailment $\ge 0.50$, zero numeric conflict | `state` | `answer` | Asserted with page/paragraph citation chips. |
| **Contradicted** | Max Contradiction $\ge 0.50$ or numeric mismatch | `withhold` | `abstain` | Suppressed with explicit contradiction reasoning. |
| **Conflicting** | Entailment from Doc A + Contradiction from Doc B | `caveat` | `answer_with_caveats` | States institutional disagreement side-by-side. |
| **Insufficient** | Neutral evidence; lacks grounding | `withhold` | `abstain` | Conservative abstention to prevent liability. |

- **Mixed Verdicts**: `partial_answer` preserves verified assertions while silently suppressing unverified claims.

---

### Slide 8: Empirical Methodology & Benchmark Results
- **Header**: Scientifically Sound Evaluation on Held-Out Test Data
- **Experimental Rigor**:
  - Canonical dataset of 180 claims across 15 synthetic institutional domains; held-out test split of 120 claims across 10 unseen domains.
  - Strict zero document leakage; 500-fold document-clustered bootstrap (95% CI).
- **Benchmark Results Table**:
  - *Baseline 1 (Raw NLI)*: Accuracy = 0.725 [0.71, 0.74], Macro F1 = 0.626, Conflicting F1 = 0.000, FAR = 0.222.
  - *Baseline 2 (Aggregation Only)*: Accuracy = **0.892 [0.88, 0.91]**, Macro F1 = **0.889**, Conflicting F1 = **0.800**, FAR = 0.111.
  - *Fair Ablation (Config B with Numeric)*: Accuracy = 0.892, FAR = 0.111, Latency = 96.8 ms.
- **Visual**: Bar chart comparing Raw NLI vs. Multi-Passage Verifier across Accuracy, Macro F1, and False Acceptance Rate.

---

### Slide 9: Temperature Calibration & Disentangled Attribution
- **Header**: Calibration Generalization & Architectural Attribution
- **Calibration Findings ($T = 2.6888$)**:
  - Calibration Split: NLL reduced by **43.8%** ($0.7926 \rightarrow 0.4455$), ECE reduced by 6.7%.
  - Held-Out Test Split: NLL reduced by **19.2%** ($0.3623 \rightarrow 0.2927$), successfully mitigating probability spikes.
  - *Nuanced Reporting*: Test ECE slightly increases ($0.068 \rightarrow 0.081$) due to probability smoothing over confident bins.
- **Attribution Breakdown**:
  - *Aggregation*: Drives conflict detection ($F_1 = 0.800$) and lifts macro accuracy (+16.7%).
  - *Decision Policy*: Calibrated 0.80 threshold demotes borderline assertions to caveats.
  - *Numeric Checks*: Eliminates semantic masking on targeted quantitative near-misses.
- **Visual**: Reliability diagram (ECE bin calibration plot) showing calibrated vs. uncalibrated probabilities.

---

### Slide 10: Limitations, Future Work & Conclusion
- **Header**: Scientific Boundaries & Future Trajectory
- **Documented Limitations**:
  - Evaluated on synthetic institutional policy texts; does not cover OCR noise or clinical records.
  - Small-cluster bootstrap ($K=10$ document clusters) represents approximate cluster-robust inference.
  - Validates explicit numbers; does not perform multi-hop tabular arithmetic.
- **Future Roadmap**:
  - Layout-aware multimodal table parsers (Table Transformer).
  - Automated temporal document supersession resolution using document metadata.
- **Conclusion**: Evidence Review replaces blind RAG trust with mathematically grounded, multi-passage, calibrated verification.
- **Closing**: *"Accountable AI for Institutional Governance."* Thank you!

