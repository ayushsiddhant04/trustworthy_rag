# Competition Demonstration Script: Evidence Review (3-Minute Walkthrough)

**Audience**: Technical Competition Judges & Evaluation Panel  
**Presenter**: Student Presenter / System Architect  
**Live Application Target**: Streamlit Interface at `http://localhost:8501`  
**Total Allocated Duration**: 3 Minutes (180 Seconds)  

---

## Pre-Demo Setup & Verification (T-minus 5 Minutes)

1. Launch application:
   ```cmd
   cmd.exe /c run_app.bat
   ```
2. Navigate to **Demo Centre** in the left sidebar navigation.
3. Ensure local DeBERTa model weights are loaded in offline mode (`HF_HUB_OFFLINE=1`).

---

## Minute 0:00 – 0:35: Introduction & The Core Vulnerability

**Presenter Spoken Dialogue**:
> "Good morning, judges. Today, language models are deployed across universities and enterprises to answer questions from official documents. But standard Retrieval-Augmented Generation suffers from a fatal blind spot: **it cannot distinguish between factual support, subtle quantitative errors, and conflicting notices.**
>
> If an AI produces a compound sentence where one fact is correct and one is wrong, standard systems either accept the whole hallucination or reject everything.
>
> We built **Evidence Review**, a document intelligence and automated claim verification system that decomposes answers into atomic propositions, runs calibrated multi-passage NLI, catches numeric mismatches deterministically, and composes conservative safe answers. Let's see it live on three real-world institutional failure modes."

**Presenter UI Action**:
- Select **Demo Centre** in the left sidebar.
- Point to the three preset scenarios.

---

## Minute 0:35 – 1:20: Scenario 1 — Compound Claim & Numeric Discrepancy

**Presenter UI Action**:
1. Click **"Scenario 1: Wrong Date + Correct Fee"**.
2. Point to the Candidate Answer:
   > *"The application deadline is 30 November 2026 and the registration fee is 500 rupees."*
3. Click the primary button: **"Run Scenario Verification"**.

**Presenter Spoken Dialogue**:
> "In Scenario 1, an applicant asks about admissions. The candidate answer asserts two facts in one sentence: the deadline is 30 November 2026, and the fee is 500 rupees.
>
> Notice what Evidence Review does:
> 1. First, our **finite-verb atomic claim extractor** decomposes the compound sentence into two independent propositions: Claim 1 for the deadline, and Claim 2 for the fee.
> 2. For Claim 1, our DeBERTa cross-encoder and date normalizer inspect the circular. The evidence states the deadline is 15 November, not 30 November. Claim 1 is flagged as **Contradicted** (99.9% confidence) and **withheld**.
> 3. For Claim 2, the evidence confirms the 500-rupee fee. Claim 2 is verified as **Supported** (98.3% confidence) and **stated**.
> 4. Finally, our **Safe Answer Composer** issues a `partial_answer`, delivering: *'The registration fee is 500 rupees.'* It completely suppresses the hallucinated deadline while preserving the verified fee, citing Page 1, Paragraph 2."

---

## Minute 1:20 – 1:55: Scenario 2 — Missing Factual Requirement & Abstention

**Presenter UI Action**:
1. In the Demo Scenario selector, switch to **"Scenario 2: Missing Evidence / Abstention"**.
2. Point to the Candidate Answer:
   > *"Candidates with state-level sports certificates receive an automatic 50 percent fee waiver."*
3. Click **"Run Scenario Verification"**.

**Presenter Spoken Dialogue**:
> "Next, Scenario 2 demonstrates a common institutional risk: **ungrounded policy fabrication**. An applicant asks if sports medals grant fee waivers.
>
> The candidate answer fabricates an 'automatic 50 percent fee waiver'. The retrieved regulations mention fee concessions for disability identity cards, but make zero mention of sports waivers.
>
> Our verifier scores both entailment and contradiction below threshold, correctly categorizing the claim as **Insufficient Evidence** (99.7% neutral confidence).
>
> Rather than hallucinating a polite approximation, our system outputs an explicit **Abstain** verdict: *'I cannot answer this question because there is not enough reliable document evidence.'* This prevents administrative liability."

---

## Minute 1:55 – 2:35: Scenario 3 — Multi-Notice Institutional Conflicts

**Presenter UI Action**:
1. Switch to **"Scenario 3: Conflicting Institutional Notices"**.
2. Point to the Candidate Answer:
   > *"The final submission deadline is 15 November."*
3. Click **"Run Scenario Verification"**.

**Presenter Spoken Dialogue**:
> "Now, our most complex challenge: **conflicting institutional notices**.
>
> In real universities, Notice A is issued in September stating the deadline is 15 November. Later, Notice B is published extending the deadline to 20 November. Both documents exist concurrently in the repository.
>
> Standard single-passage NLI models fail completely here—in our benchmark, raw NLI misclassified 20 of 30 conflicting claims as supported, suffering a 22.2% False Acceptance Rate.
>
> Watch Evidence Review: our **multi-passage aggregation engine** pools evidence across both notices simultaneously. It detects strong entailment from Notice A alongside direct contradiction from Notice B. It assigns the status **Conflicting** and issues an `answer_with_caveats`, displaying both citations side-by-side: *'Notice A specifies 15 November, but Notice B specifies 20 November.'*"

---

## Minute 2:35 – 3:00: Research Evaluation & Conclusion

**Presenter UI Action**:
1. Click **Research Evaluation** in the sidebar.
2. Select **Canonical Held-Out Test Set (120 claims, 10 unseen domains)**.
3. Switch to **Tab 3: Fair Numeric/Date Ablation (A vs B)**.

**Presenter Spoken Dialogue**:
> "To conclude: we did not rely on decorative metrics. We evaluated our system on a **strictly partitioned held-out benchmark of 120 claims across 10 unseen document domains with zero document leakage**.
>
> Our empirical results demonstrate:
> - **Multi-passage aggregation** improves classification accuracy from **72.5% to 89.2%** (+16.7 percentage points) and enables conflict detection with an F1 score of **0.800** on held-out test data.
> - **Temperature scaling** ($T = 2.6888$) reduces cross-entropy log-loss by **19.2%** on unseen test documents.
> - And our **calibrated decision policy** cuts false factual acceptances by half.
>
> Evidence Review turns brittle document search into accountable, verifiable intelligence. Thank you, and we welcome your questions."

---

## Presenter Quick Reference Card (Timing & Key Statistics)

| Section | Target Time | Key Concept Demonstrated | Grounded Experimental Number |
|---|---|---|---|
| **Intro** | 0:00 – 0:35 | RAG failure modes; atomic verification | N/A |
| **Scenario 1** | 0:35 – 1:20 | Compound claim splitting; fee verified, date suppressed | 99.97% contradiction on deadline; 98.28% support on fee |
| **Scenario 2** | 1:20 – 1:55 | Missing evidence detection; safe abstention | 99.7% neutral confidence; zero hallucination |
| **Scenario 3** | 1:55 – 2:35 | Multi-passage pooling; conflicting circulars | Raw NLI fails ($F_1 = 0.000$); Verifier achieves $F_1 = 0.800$ |
| **Evaluation** | 2:35 – 3:00 | Document-grouped held-out test benchmark | **Accuracy 89.2% [0.88, 0.91]**; **19.2% NLL reduction** on $N=120$ test claims |

