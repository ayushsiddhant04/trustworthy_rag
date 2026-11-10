# Research Experiment Reproduction Guide

This document details the exact protocols, commands, parameters, and mathematical definitions necessary to reproduce the empirical evaluation of the **Evidence Review** system from scratch.

---

## 1. Environment & Hardware Specifications

- **Operating System**: Windows 11 / Server x64
- **Python Version**: Python 3.12.10 (virtual environment: `.venv`)
- **PyTorch**: 2.6.0+cpu
- **Transformers**: 4.49.0
- **Sentence-Transformers**: 3.4.1
- **Scikit-Learn**: 1.6.1
- **Hardware Profile**: CPU inference (AMD/Intel x86_64, 8+ threads, 16GB+ RAM)
- **Model Checkpoint**: `cross-encoder/nli-deberta-v3-base` (local offline cache)

### Strict Offline Requirement

Before executing any script, enforce offline Hugging Face execution to prevent unauthorized network requests and ensure exact local determinism:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = "."
```

---

## 2. Reproduction Commands

To reproduce temperature calibration, baseline evaluations, and confidence interval computations in a single reproducible run:

```powershell
# In PowerShell:
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; .venv\Scripts\python.exe member_b/run_canonical_experiments.py
```

To run the automated verification suite validating all empirical constraints and split isolation:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_phase6_research_validation.py -v
```

To run the full project regression test harness (87 tests):

```cmd
cmd.exe /c run_tests.bat
```

---

## 3. Dataset Partitions & Inputs

All experiments evaluate identical held-out examples with zero document leakage:

- **Calibration Split** (`member_b/data/calibration_split.csv`):
  - $N = 36$ claims, 46 passage-claim pairs across 3 independent document domains (`doc_admissions_circular_2026`, `doc_attendance_regulations`, `doc_fellowship_guidelines`).
  - Used exclusively for fitting temperature scaling parameter $T$.
- **Validation Split** (`member_b/data/validation_split.csv`):
  - $N = 24$ claims across 2 document domains (`doc_exam_schedule_circulars`, `doc_library_policies`).
- **Held-Out Test Split** (`member_b/data/test_split.csv`):
  - $N = 120$ claims across 10 unseen document domains (`doc_hostel_regulations`, `doc_research_grant_awards`, `doc_symposium_call_for_papers`, `doc_tuition_fee_structure`, `doc_degree_curriculum_rules`, `doc_laboratory_safety_manual`, `doc_travel_reimbursement_rules`, `doc_sports_quota_ordinance`, `doc_disciplinary_code`, `doc_medical_leave_policy`).
  - Balanced class distribution: 30 `supported`, 30 `contradicted`, 30 `insufficient`, 30 `conflicting`.

---

## 4. Evaluated Configurations

1. **Baseline 1: Raw NLI (`run_baseline_1_raw_nli`)**:
   - Concatenates all evidence passages into a single string.
   - Executes standard NLI pair classification: $(P_{\text{combined}}, H_{\text{claim}})$.
   - Maps predicted probabilities directly to labels without windowing, cross-passage aggregation, or numeric rules.
   - Action policy: `state` if `supported`, `caveat` if `conflicting`, `withhold` otherwise (bypasses confidence gating).
2. **Baseline 2: Aggregation Only (`run_baseline_2_aggregation_only`)**:
   - Tokenizes passages into 240-token windows with 50-token overlap.
   - Computes window-level NLI scores across all passages.
   - Evaluates multi-passage conflict aggregation (`aggregate_evidence`) using thresholds $\tau_e = 0.60, \tau_c = 0.60$.
   - Does NOT apply numeric or date heuristics (`numeric_conflict = False`).
3. **Baseline 3: Numeric Checks Only (`run_baseline_3_numeric_only`)**:
   - Executes raw concatenated NLI pair classification.
   - Directly applies `check_numeric_conflict(claim, passages)` on semantic fields (date, fee, percentage, duration, count).
   - If a numeric conflict is detected on a `supported` claim, status is overridden to `contradicted`.
   - Passes status, confidence, and numeric flag to `decide(status, conf, numeric_conflict)`.
4. **Baseline 4: Complete Verifier Ungated (`run_baseline_4_complete_ungated`)**:
   - Executes the complete `VerificationEngine`: token windowing, multi-passage aggregation, and context-aware numeric checks.
   - Relevance gating is disabled.
5. **Proposed Variant: Verifier + Relevance Gating (`run_proposed_relevance_gated`)**:
   - Pre-filters evidence passages using `RelevanceChecker` (TF-IDF cosine similarity with threshold $\theta = 0.30$).
   - Passes filtered passages to `VerificationEngine`.

---

## 5. Metric Definitions & Formulas

Let $N = 120$ be the total test claims, $y_i \in \{\text{supported}, \text{contradicted}, \text{insufficient}, \text{conflicting}\}$ be the ground-truth status, $\hat{y}_i$ be the predicted status, and $a_i \in \{\text{state}, \text{caveat}, \text{withhold}\}$ be the decision policy action.

### 1. Overall Accuracy
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i = \hat{y}_i) = \frac{\text{Trace}(M)}{\sum M}$$
where $M$ is the $4 \times 4$ confusion matrix.

### 2. Macro F1 Score
$$\text{Macro } F_1 = \frac{1}{4} \sum_{c \in \text{Classes}} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
where $P_c = \frac{M_{c,c}}{\sum_{r} M_{r,c}}$ and $R_c = \frac{M_{c,c}}{\sum_{k} M_{c,k}}$ with zero-division handling set to 0.

### 3. False Acceptance Rate (FAR)
$$\text{FAR} = \frac{\sum_{i: y_i \ne \text{supported}} \mathbb{I}(a_i = \text{'state'})}{\sum_{i=1}^{N} \mathbb{I}(y_i \ne \text{supported})} = \frac{\text{False 'state' Actions}}{90}$$
*Critical Note*: FAR measures operational deployment hazard. A false acceptance occurs when any non-supported claim (contradicted, insufficient, or conflicting) is presented to the user as settled factual truth ($a_i = \text{'state'}$).

### 4. Coverage and Abstention Rate
$$\text{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(a_i \in \{\text{'state'}, \text{'caveat'}\}), \quad \text{Abstention} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(a_i = \text{'withhold'})$$
By definition, $\text{Coverage} + \text{Abstention} = 1.000$.

### 5. Document-Cluster Bootstrap Confidence Intervals
Resampling is performed over the $K = 10$ unique document clusters:
1. Sample $K = 10$ document IDs with replacement using random seed $S = 42$.
2. Gather all test claims associated with sampled document clusters ($B = 500$ iterations).
3. Compute metric $\theta^{(b)}$ for each bootstrap sample.
4. Set $95\%$ CI as $[\text{Percentile}_{2.5\%}, \text{Percentile}_{97.5\%}]$.

---

## 6. Deep Dive Investigation: Baseline 1 vs. Baseline 3 (FAR vs. Accuracy)

### Observed Metric Paradox
In the summary table:
- **Baseline 1 (Raw NLI)**: Accuracy = 0.725, Macro F1 = 0.626, **FAR = 0.222**
- **Baseline 3 (NLI + Numeric)**: Accuracy = 0.725, Macro F1 = 0.626, **FAR = 0.022**

Why did Baseline 3 reduce FAR by $90\%$ while classification accuracy and macro F1 remained identical?

### Investigation & Root Cause Analysis

1. **Separation of Status Classification from Decision Action**:
   - `Accuracy` and `Macro F1` are evaluated strictly against $\hat{y}_i$ (`predicted_label` / `status`).
   - `FAR` is evaluated strictly against $a_i$ (`action`), defined by the operational decision policy.
2. **Action Assignment Differences**:
   - **In Baseline 1 (`run_baseline_1_raw_nli`)**:
     ```python
     action = "state" if status == "supported" else ("caveat" if status == "conflicting" else "withhold")
     ```
     Baseline 1 mapped every `supported` classification directly to `state` without checking verification confidence. Of the 30 conflicting claims in the test set, raw NLI misclassified 20 as `supported`. All 20 were unconditionally assigned `action = 'state'`, yielding $\text{FAR} = 20 / 90 = 0.2222$.
   - **In Baseline 3 (`run_baseline_3_numeric_only`)**:
     ```python
     dec = decide(status, conf, numeric_conflict=num_check["numeric_conflict"])
     action = dec["action"]
     ```
     Baseline 3 routes predictions through `decision_policy.decide()`, which enforces the `HIGH_CONFIDENCE_THRESHOLD = 0.80` rule:
     ```python
     if status == "supported":
         if confidence >= high_confidence_threshold: # 0.80
             return {"action": "state"}
         return {"action": "caveat"}
     ```
3. **The Role of Temperature Scaling ($T = 2.6888$)**:
   - Under calibrated temperature scaling, DeBERTa's overconfident probability spikes (>0.99) were softened into the $0.65 - 0.78$ range for 18 of the 20 misclassified conflicting claims.
   - Because their calibrated confidence was below $0.80$, `decide()` assigned `action = 'caveat'` rather than `state`.
   - As a result, false `state` actions dropped from 20 down to 2, yielding $\text{FAR} = 2 / 90 = 0.0222$.
4. **Why Status Accuracy Did Not Change**:
   - For those 18 claims, `status` remained `supported` (since the numeric heuristic found no single-source numeric contradiction within the prompt text).
   - Because `accuracy` and `macro_f1` are computed strictly on `status` vs `true_status`, accuracy remained exactly $87 / 120 = 0.725$, and macro F1 remained $0.6264$.

This demonstrates the crucial distinction between **semantic classification** and **operational risk mitigation**: the calibrated decision policy prevents catastrophic overconfidence in user-facing assertions even when the underlying semantic classifier is imperfect.

---

## 7. Complete Reproduced Metrics Table

| Metric | Baseline 1 (Raw NLI) | Baseline 2 (Agg) | Baseline 3 (Numeric) | Baseline 4 (Ungated) | Proposed (Gated) |
|---|---|---|---|---|---|
| **Accuracy** | 0.725 | 0.892 | 0.725 | 0.892 | 0.875 |
| **Accuracy 95% CI** | [0.708, 0.740] | [0.875, 0.906] | [0.708, 0.740] | [0.875, 0.906] | [0.845, 0.905] |
| **Macro F1** | 0.626 | 0.889 | 0.626 | 0.889 | 0.868 |
| **Macro F1 95% CI** | [0.608, 0.643] | [0.869, 0.905] | [0.608, 0.643] | [0.869, 0.905] | [0.833, 0.901] |
| **Macro Precision** | 0.558 | 0.915 | 0.558 | 0.915 | 0.912 |
| **Macro Recall** | 0.725 | 0.892 | 0.725 | 0.892 | 0.875 |
| **False Acceptance Rate** | 0.222 | 0.111 | 0.022 | 0.111 | 0.133 |
| **Coverage** | 0.417 | 0.500 | 0.417 | 0.500 | 0.500 |
| **Abstention Rate** | 0.583 | 0.500 | 0.583 | 0.500 | 0.500 |
| **Mean Latency (ms)** | 91.4 ms | 96.4 ms | 91.3 ms | 97.7 ms | 106.7 ms |

