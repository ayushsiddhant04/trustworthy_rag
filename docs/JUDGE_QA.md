# Technical Defense Guide: 35 Core Judge Questions & Model Answers

**Document Purpose**: Comprehensive technical preparation for competition evaluation panels, viva voce examinations, and judge Q&A sessions.

---

### 1. Architectural & Theoretical Foundations

#### Q1: Why use Natural Language Inference (NLI) rather than semantic vector similarity (e.g., cosine similarity)?
**Answer**: Cosine similarity is a symmetric geometric metric: $\text{sim}(A, B) = \text{sim}(B, A)$. It measures topical relatedness rather than directional truth. For example, *"The deadline is 15 November"* and *"The deadline is NOT 15 November"* have near-identical vector representations (>0.92 cosine similarity) because they share identical lexical contexts, yet they are direct logical opposites. NLI models directional logical entailment, distinguishing whether a premise $P$ guarantees, refutes, or is neutral to a hypothesis $H$.

#### Q2: Why choose DeBERTa-v3 as the cross-encoder backbone?
**Answer**: `cross-encoder/nli-deberta-v3-base` uses disentangled attention (representing token content and relative position in separate vectors) and an enhanced masked language modeling objective. Unlike bi-encoders (which encode premise and hypothesis into separate vectors before comparing), a cross-encoder passes the premise and hypothesis concatenated into joint self-attention layers, allowing every word in the claim to cross-attend to every word in the document passage. This yields substantially higher precision on subtle semantic contradictions.

#### Q3: In our architecture, what constitutes the "premise" and the "hypothesis"?
**Answer**: The **premise** is a retrieved document chunk (or sliding window excerpt) from the official institutional text. The **hypothesis** is an extracted atomic claim derived from the candidate answer. The cross-encoder evaluates: *Does the official document excerpt (premise) logically entail, contradict, or remain neutral toward the candidate claim (hypothesis)?*

#### Q4: How does evidence windowing work?
**Answer**: DeBERTa-v3-base has a hard maximum context length of 512 tokens. A long document passage often exceeds this limit or dilutes attention across irrelevant sentences. Our windowing module (`member_b/windowing.py`) segments evidence text into overlapping sub-windows of 240 tokens with 50-token overlap, ensuring the cross-encoder evaluates focused, high-density premise spans against the claim.

#### Q5: Why are overlapping windows necessary?
**Answer**: Factual propositions often span paragraph or sentence boundaries. A rigid chunk boundary might cut a key condition (e.g. *"provided that..."*) in half, placing the premise in window 1 and the qualification in window 2. A 50-token overlap ensures boundary-spanning factual clauses appear together in at least one window.

#### Q6: Why not use a large language model (LLM) as the only verifier?
**Answer**: While LLMs can be prompted to verify facts, they suffer from three engineering drawbacks in production verification: (1) **Non-determinism and stochastic hallucinations**, (2) **Substantially higher inference latency and financial cost** per claim, and (3) **Lack of calibrated, continuous confidence scores**. A dedicated cross-encoder provides deterministic, reproducible logits that can be directly calibrated via temperature scaling and audited mathematically.

---

### 2. Multi-Passage Aggregation & Cross-Document Conflicts

#### Q7: What does multi-passage evidence aggregation do?
**Answer**: Rather than assuming a single passage contains all relevant information, `aggregate_evidence` evaluates all retrieved passages associated with a claim independently. It extracts the maximum entailment, contradiction, and neutral scores for each passage, and then pools these signals across the document set to determine whether the collective evidence uniformly supports, uniformly contradicts, or presents an institutional conflict.

#### Q8: How is conflicting evidence formally identified?
**Answer**: A claim is classified as `conflicting` when at least one authentic passage exhibits strong entailment ($\max E \ge \tau_e$) while another authentic passage exhibits strong contradiction ($\max C \ge \tau_c$). This occurs frequently in institutional governance when revised notices, amended policies, or departmental exceptions contradict general circulars.

#### Q9: What happens when official documents directly conflict?
**Answer**: Rather than randomly choosing one passage or outputting a single hallucinated consensus, our safe composer issues an `answer_with_caveats` (or `partial_answer_with_caveats`). It explicitly presents both sides to the user, citing both sources: *"Notice A specifies 15 November, but Notice B specifies 20 November."*

---

### 3. Numeric & Date Verification Engine

#### Q10: How does the context-aware numeric and date checking engine work?
**Answer**: The module (`member_b/numeric_checks.py`) parses numbers, percentages, durations, currency fees, and dates into structured typed records. It matches values between claim and evidence **only when they belong to the same semantic field** (e.g., fee compared against fee; deadline compared against deadline). If matching semantic contexts contain incompatible numbers or dates, it flags a deterministic numeric conflict.

#### Q11: How are equivalent date representations handled?
**Answer**: Dates are normalized into ISO-8601 calendar dates (`YYYY-MM-DD`). The parser normalizes formats such as *"15 November 2026"*, *"November 15, 2026"*, and *"15/11/2026"* into identical date objects, preventing spurious mismatch warnings when different date formats express the same day.

#### Q12: How do you prevent calendar years from being treated as fees or counts?
**Answer**: The date parser consumes four-digit numbers between 1900 and 2099 as date components before numeric value extraction runs. Once a four-digit number is tagged as a calendar year, it is masked so the currency/fee parser never treats *"2026"* as a fee of 2026 rupees.

#### Q13: In the fair ablation experiment, why did numeric checking show no change in macro accuracy (0.892 vs 0.892)?
**Answer**: On broad out-of-domain evaluation text, DeBERTa-v3's multi-passage window aggregation already achieves 0.952 contradicted F1 because general contradictions in the test set contain substantial semantic divergence that cross-encoder self-attention easily flags. The primary value of the numeric rule is **targeted defense against semantic near-misses and compound claim traps** (e.g., changing only the fee amount while keeping all other words identical), as proven in our 8 controlled scenarios.

---

### 4. Claim Decomposition & Safe Answer Composition

#### Q14: Why is atomic claim extraction necessary?
**Answer**: Candidate answers generated by LLMs or humans frequently combine multiple assertions into compound sentences (e.g., *"The deadline is 30 November and the fee is 500 rupees"*). If verified as a monolithic unit, the truth of the fee statement can semantically mask the falsehood of the deadline statement. Atomic extraction decomposes compound clauses across finite-verb coordinating conjunctions, allowing each proposition to be verified, cited, and acted upon independently.

#### Q15: Why must a verification system be able to abstain?
**Answer**: In high-stakes domains (legal, medical, admissions), stating a false fact is far more dangerous than acknowledging incomplete information. An abstention mechanism prevents administrative liability by guaranteeing that if evidence is absent or inconclusive, the system outputs an explicit `abstain` or `partial_answer` rather than manufacturing ungrounded filler text.

#### Q16: What is the 5-state application verdict hierarchy?
**Answer**:
1. `answer`: All extracted claims are verified as `supported` and stated with citations.
2. `answer_with_caveats`: Supported claims accompanied by authentic institutional conflicts (caveats).
3. `partial_answer`: Supported claims are stated, while unverified/contradicted claims are suppressed without caveats.
4. `partial_answer_with_caveats`: Mix of stated supported claims, caveated conflicts, and suppressed contradictions.
5. `abstain`: All claims are unverified or contradicted; complete safe abstention.

---

### 5. Retrieval & Preprocessing Pipeline

#### Q17: How does document ingestion handle complex multi-format inputs?
**Answer**: `src/ingestion.py` implements dedicated extractors for PDF (PyMuPDF / `fitz`), DOCX (`python-docx`), and plain text. It validates file sizes ($\le 25\text{ MB}$) and page counts ($\le 100\text{ pages}$), skips blank pages with logged warnings, and raises an explicit `ScannedPdfError` if a PDF contains only scanned raster images without an extractable text layer.

#### Q18: What is the difference between BM25 sparse retrieval and dense embedding retrieval?
**Answer**: BM25 sparse retrieval matches exact lexical tokens using term frequency and inverse document frequency. It excels at finding exact IDs, regulation numbers, and specific dates. Dense embedding retrieval (`all-MiniLM-L6-v2`) maps text into a 384-dimensional semantic space, capturing conceptual similarity and paraphrasing. Our system uses BM25 by default for deterministic local deployment, with an optional hybrid mode combining both scores.

#### Q19: What is relevance gating, and what was its measured effect?
**Answer**: Relevance gating (`RelevanceChecker`) applies an optional TF-IDF cosine similarity threshold ($\theta = 0.30$) to filter out irrelevant distractor passages before NLI inference. In our evaluation, it increased precision on contradicted claims ($0.91 \rightarrow 0.98$), but lexical filtering pruned paraphrased contrasting evidence in 4 conflicting examples, reducing conflicting recall from $66.7\%$ to $53.3\%$. It is therefore architected as an optional, configurable module.

---

### 6. Calibration, Metrics & Statistical Methodology

#### Q20: How is temperature scaling fitted, and what problem does it solve?
**Answer**: Modern neural networks output overconfident probability distributions. Temperature scaling divides raw uncalibrated logits $z$ by a learned scalar $T > 0$ before applying softmax: $\hat{p} = \text{softmax}(z / T)$. The temperature $T = 2.6888$ was fitted strictly on a separate calibration dataset ($N = 46$ pairs from 3 independent documents) by minimizing Negative Log-Likelihood (NLL) via bounded scalar optimization.

#### Q21: What is the operational meaning of the 0.80 high-confidence threshold?
**Answer**: Softmax confidence represents the model's posterior probability of entailment. The 0.80 threshold enforces an asymmetric safety margin: even if a claim is classified as `supported`, it is only stated as settled fact if calibrated confidence meets or exceeds 0.80. If confidence falls between 0.50 and 0.79, the decision policy demotes the action to `caveat`, preventing marginal assertions from being presented as unconditional truth.

#### Q22: What is False Acceptance Rate (FAR), and what is its denominator?
**Answer**: False Acceptance Rate measures the proportion of non-supported claims (contradicted, insufficient, or conflicting) that the system erroneously presents to the user as settled fact ($a_i = \text{'state'}$):
$$\text{FAR} = \frac{\sum_{y_i \ne \text{supported}} \mathbb{I}(a_i = \text{'state'})}{\sum \mathbb{I}(y_i \ne \text{supported})} = \frac{\text{False 'state' Actions}}{90 \text{ non-supported claims}}$$
In our held-out test set ($N=120$), exactly 30 claims are supported and 90 are non-supported, making the denominator strictly 90.

#### Q23: Why report Macro F1 rather than Micro F1 or simple Accuracy?
**Answer**: In 4-class verification with balanced categories (30 per class), Macro F1 computes the unweighted arithmetic mean of F1 scores across all four classes: $\frac{1}{4}(F_{1,\text{sup}} + F_{1,\text{con}} + F_{1,\text{ins}} + F_{1,\text{cnf}})$. This prevents high performance on easy entailments from masking catastrophic failures on conflicting or insufficient evidence.

#### Q24: How was data leakage prevented across dataset splits?
**Answer**: Dataset partitioning was enforced strictly at the **document group level**:
- Calibration ($N=36$ claims, 3 documents)
- Validation ($N=24$ claims, 2 documents)
- Held-Out Test ($N=120$ claims, 10 documents)
No document appears in more than one split. There is exactly 0% claim text overlap, 0% evidence passage overlap, and complete quarantine from demonstration documents (`data/demo/`).

#### Q25: Why are the evaluation documents synthetic?
**Answer**: Genuine institutional documents from universities or corporations often lack dense, controlled instances of multi-source contradictions, compound date perturbations, and explicit passage-level entailment tags. Synthetic generation allows us to construct an independently verifiable, ground-truth benchmark with balanced classes and zero ambiguity. We transparently disclose that all 15 domains are synthetic administrative policy texts.

#### Q26: What is the limitation of the document-group bootstrap sample?
**Answer**: Our test set contains $K = 10$ unique document clusters. In statistical resampling theory, asymptotic cluster-robust inference typically requires $K \ge 30-50$ clusters. With $K = 10$, intra-document claim correlations are properly accounted for, but the resulting 95% confidence intervals should be interpreted as approximate cluster-robust estimates rather than exact asymptotic intervals.

---

### 7. Scalability, Deployment & Edge Cases

#### Q27: How would this architecture scale to large document repositories (e.g., 50,000 pages)?
**Answer**: The two-stage architecture scales linearly:
1. **Retrieval Stage**: BM25 inverted indexes or dense vector databases (e.g. FAISS/Qdrant) narrow down 50,000 pages to top-$k$ relevant chunks ($k \approx 5-10$) in sub-50 milliseconds.
2. **Verification Stage**: Cross-encoder verification is applied **only to the top-$k$ retrieved chunks**, keeping inference complexity constant regardless of total repository size.

#### Q28: What happens if an applicant submits a claim with no numbers or dates?
**Answer**: The numeric check returns `numeric_conflict = False` with zero false alarms. Verification proceeds purely through DeBERTa-v3 cross-encoder self-attention and multi-passage aggregation, as verified in Controlled Case 8.

#### Q29: What happens if candidate answer text contains complex table structures?
**Answer**: Plain-text extractors can lose tabular column alignments. While our current system handles narrative circulars and schedules reliably, multi-column tables with complex header nesting require layout-aware PDF parsers (e.g. Table Transformer), which is documented as a future enhancement.

#### Q30: How does the system handle subtle temporal supersession (Notice B supersedes Notice A)?
**Answer**: When Notice B alters a rule without explicitly naming Notice A, the system flags the evidence as `conflicting` and issues an `answer_with_caveats` displaying both notices. True automatic supersession resolution requires document publication timestamp metadata, which NLI models cannot deduce from text alone.

#### Q31: Can Evidence Review run completely air-gapped without internet access?
**Answer**: Yes. All model weights (`cross-encoder/nli-deberta-v3-base` and `sentence-transformers/all-MiniLM-L6-v2`) are stored in the local cache. The application runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, requiring zero outbound network traffic.

#### Q32: How is latency managed during live user interactions?
**Answer**: Verification latency averages **96.8 ms per claim** on standard CPU hardware. By segmenting passages into focused 240-token windows and batching pair predictions, full multi-passage verification executes in under 200 ms for typical 2-claim answers.

#### Q33: How does the system handle claims with multiple contradictory numbers (e.g. 500 fee vs 750 fee)?
**Answer**: The contextual numeric checker identifies the fee field, compares values within that field, and triggers a deterministic contradiction flag, overriding neutral semantic scores and forcing the action to `withhold`.

#### Q34: What is the difference between a "partial answer" and an "answer with caveats"?
**Answer**: A **partial answer** occurs when an answer contains multiple assertions, some of which are verified while others lack evidence or are contradicted. The verified assertions are stated, and the unverified ones are silently omitted. An **answer with caveats** occurs when legitimate documents disagree on a single assertion; the user is explicitly informed of the institutional conflict.

#### Q35: Does Evidence Review claim state-of-the-art superiority over academic benchmarks like FEVER or HoVer?
**Answer**: No. We make no unsubstantiated claims of SOTA superiority. Evidence Review is specifically engineered to solve document-grounded institutional claim verification, multi-passage institutional conflict arbitration, and compound sentence decomposition on policy texts.

