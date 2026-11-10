# Research Annotation Guidelines: Document Intelligence & Claim Verification

**Protocol Version**: 2.0.0  
**Project**: Evidence Review  
**Date**: November 2026  

---

## 1. Objective and Scope

These guidelines establish standardized, reproducible criteria for annotating factual claims against retrieved document evidence. The taxonomy addresses single-passage and multi-passage institutional documents, enforcing clear distinctions between ungrounded claims, direct contradictions, and genuine multi-source institutional conflict.

---

## 2. Core Four-Class Verification Taxonomy

Every evaluation pair comprises an atomic factual hypothesis ($C$) and a set of retrieved document evidence passages ($E = \{p_1, p_2, \dots, p_k\}$). Annotators assign exactly one of four mutually exclusive claim-level ground-truth status labels:

### 2.1 SUPPORTED
- **Operational Definition**: The available evidence passages directly entail and establish the claim. All factual elements (quantities, dates, entities, modalities, and scope) stated in the claim are substantiated by the text.
- **Decision Rule**: At least one authentic passage entails the claim ($p_i \models C$), and zero authentic passages contradict it, and zero numeric/date mismatches exist.
- **Example**:
  - *Claim*: "Undergraduate degree completion requires earning at least 160 academic credits."
  - *Evidence*: "Curriculum degree rules state that undergraduate degree completion requires earning at least 160 academic credits."
  - *Label*: `SUPPORTED` (Action: `STATE`).

### 2.2 CONTRADICTED
- **Operational Definition**: The available evidence passages directly refute or conflict with the factual assertions of the claim within a single coherent institutional source context.
- **Decision Rule**: The evidence establishes an incompatible factual reality ($p_i \models \neg C$), or asserts a conflicting date/numeric value for the same entity or cue word within a unified authority.
- **Example**:
  - *Claim*: "The semester tuition fee for postgraduate science programs is 45000 rupees."
  - *Evidence*: "The semester tuition fee for postgraduate science programs is 25000 rupees."
  - *Label*: `CONTRADICTED` (Action: `WITHHOLD`).

### 2.3 INSUFFICIENT
- **Operational Definition**: The available evidence neither substantiates nor refutes the claim. The document is silent, neutral, or discusses related background without verifying the specific proposition.
- **Decision Rule**: Zero passages entail the claim, and zero passages directly contradict the claim. Includes lexical overlap without logical entailment, and unmentioned exemptions.
- **Example**:
  - *Claim*: "National athletics medalists receive a 50 percent waiver on application fees."
  - *Evidence*: "Payment schedule: Candidates must pay a non-refundable application processing fee of 750 rupees. Concessions are available for disability card holders."
  - *Label*: `INSUFFICIENT` (Action: `WITHHOLD`).

### 2.4 CONFLICTING
- **Operational Definition**: Multiple authentic, authoritative document passages disagree about the factual status of the claim.
- **Decision Rule**: The evidence set contains at least one authentic passage that supports the claim ($p_a \models C$) AND at least one authentic passage that contradicts the claim ($p_b \models \neg C$).
- **Crucial Distinction (Numeric Error vs. Multi-Notice Conflict)**:
  - *Single-Source Error*: A candidate answer hallucinates "30 November" when the circular states "15 November". Here, the document is unified; the claim is **CONTRADICTED** and must be **WITHHELD**.
  - *Multi-Source Conflict*: Circular A (issued in August) states the deadline is "15 November", while Revised Circular B (issued in October) states the deadline is "20 November". Here, both passages are authentic; the claim is **CONFLICTING** and must be **CAVEATED** to transparently present both notices to the user.
- **Example**:
  - *Claim*: "The full paper submission deadline is 20 October 2026."
  - *Passage 1 (Call for Papers)*: "The full paper submission deadline is 20 October 2026."
  - *Passage 2 (Extension Addendum)*: "The paper submission deadline has been extended to 05 November 2026."
  - *Label*: `CONFLICTING` (Action: `CAVEAT`).

---

## 3. Passage-Level Relationship Annotation

For multi-passage benchmarks, annotators label each individual passage $p_i$ with its passage-level relationship to claim $C$:

| Passage Relationship | Criterion | Typical Claim Aggregation Effect |
| :--- | :--- | :--- |
| `entails` | Passage directly establishes the proposition. | Contributes to `supported` or `conflicting`. |
| `contradicts` | Passage directly refutes or provides incompatible facts. | Contributes to `contradicted` or `conflicting`. |
| `neutral` | Passage discusses the topic but is non-committal. | Contributes to `insufficient`. |
| `irrelevant` | Passage has minimal topical relevance to the claim. | Filtered by semantic gating or neutral pooling. |

---

## 4. Quality Control & Adjudication Protocol

### 4.1 Double Annotation Requirement
- A minimum of **20% of all evaluation examples** must be independently double-annotated by two trained annotators without cross-communication.
- Annotators evaluate randomized claims without seeing experimental system predictions or model scores.

### 4.2 Inter-Annotator Agreement Calculation
- Where multiple annotations exist, calculate Cohen's Kappa ($\kappa$) for two raters or Fleiss' Kappa for three or more raters:
  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
- If $\kappa < 0.75$, the guidelines must be reviewed for ambiguous edge cases before proceeding.
- *Scientific Integrity Policy*: Real inter-annotator agreement scores are only computed when multiple human annotations are formally collected. Agreement scores must never be simulated or fabricated.

### 4.3 Disagreement Adjudication Procedure
When Annotator 1 and Annotator 2 assign differing status labels:
1. **Passage Scrutiny**: Verify whether both annotators considered all supplied evidence passages.
2. **Authority Check**: For potential `conflicting` claims, verify whether both contradictory passages are from authentic institutional sources or whether one was an ungrounded candidate string.
3. **Adjudicator Ruling**: A designated Senior Domain Specialist reviews the conflicting rationale and assigns the final canonical ground-truth label, logging the decision in `annotation_notes`.

