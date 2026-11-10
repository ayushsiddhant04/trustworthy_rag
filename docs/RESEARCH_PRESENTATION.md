# Scientific Evaluation & Empirical Findings Presentation

**Document Purpose**: Scientific presentation notes for competition judges, technical reviewers, and defense panels.  
**Primary Checkpoint**: `cross-encoder/nli-deberta-v3-base`  
**Evaluation Set**: Canonical Held-Out Test Split ($N = 120$ claims, 150 passage pairs across 10 unseen document domains)  
**Calibration Artifact**: Temperature $T = 2.6888$ fitted on $N = 46$ calibration pairs across 3 independent document domains  

---

## 1. Dataset Architecture & Scientific Disclosures

### Synthetic Document Disclosure
The canonical benchmark consists of **15 document domains** (180 total claims). All 15 domains are **synthetic administrative policy texts** programmatically generated using structured grammar templates to emulate institutional circulars, regulations, handbooks, and ordinances.
- **Scientific Integrity Guarantee**: None of the documents are scraped from live institutional websites or represent genuine third-party legal documents.
- **Purpose**: Enables controlled, reproducible benchmarking of claim verification algorithms with exact ground-truth passage annotations.

### Strict Document-Grouped Partitioning
To prevent data leakage, dataset splitting is enforced strictly at the **document group level**:
- **Calibration Split** (`member_b/data/calibration_split.csv`):
  - $N = 36$ claims (46 passage pairs) across 3 independent domains (`doc_admissions_circular_2026`, `doc_attendance_regulations`, `doc_fellowship_guidelines`).
  - Used exclusively for fitting temperature scaling parameter $T$.
- **Validation Split** (`member_b/data/validation_split.csv`):
  - $N = 24$ claims (30 passage pairs) across 2 domains (`doc_exam_schedule_circulars`, `doc_library_policies`).
- **Held-Out Test Split** (`member_b/data/test_split.csv`):
  - $N = 120$ claims (150 passage pairs) across 10 unseen domains.
  - Exactly balanced across 4 target classes: 30 `supported`, 30 `contradicted`, 30 `insufficient`, 30 `conflicting`.
- **Zero Document Leakage**: $\text{Docs}(\text{Calib}) \cap \text{Docs}(\text{Val}) \cap \text{Docs}(\text{Test}) = \emptyset$. Exact text claim overlap: 0. Exact evidence passage overlap: 0. Demo file overlap: 0.

---

## 2. Five Critical Scientific Distinctions

In presenting our findings, we adhere strictly to scientific honesty and avoid exaggerating contributions:

1. **Multi-Passage Aggregation Improvement**:
   - The reported accuracy increase from **0.725 to 0.892** (+16.7 percentage points) and Macro F1 increase from **0.626 to 0.889** is demonstrated on the **120-claim synthetic held-out test split**. It reflects the architectural necessity of cross-passage pooling when evidence is distributed across divergent notices.
2. **Fair Numeric/Date Ablation Metric Equivalence**:
   - When all other components (windowing, aggregation, temperature scaling, decision policy) are held identical, Configuration A (no numeric checks) and Configuration B (with numeric checks) yield **identical status accuracy (0.892) and FAR (0.111)** on this general held-out test set.
   - The primary value of the rule-based numeric engine is **deterministic defense against semantic masking on targeted near-miss perturbations and compound claims**, rather than broad macro-metric lifting.
3. **Disentangling Decision Policy from Numeric Rules**:
   - In our historical audit, Baseline 3 reported an apparent FAR reduction from 0.222 to 0.022. Our Phase 6.1 audit established that this reduction was driven by **temperature scaling ($T=2.6888$) and confidence thresholding ($0.80$) moving 18 misclassified claims from `state` to `caveat`**, NOT purely by numeric checks. The decision policy and numeric rules must not be conflated.
4. **Nuanced Temperature Scaling Generalization**:
   - Temperature scaling **improved held-out test Negative Log-Likelihood (NLL) by 19.2%** ($0.3623 \rightarrow 0.2927$), successfully mitigating DeBERTa's extreme probability spikes.
   - However, **Brier score remained essentially flat (+1.59%) and Expected Calibration Error (ECE) slightly worsened ($0.068 \rightarrow 0.081$)** on held-out test data. Probability smoothing flattened extreme confidences into intermediate bins, which slightly increased bin-level error. Calibration must not be described as uniformly improved across all metrics.
5. **Relevance Gating Performance Trade-Off**:
   - Relevance gating is an **optional, configurable module**. While it increased precision on contradicted claims ($0.91 \rightarrow 0.968$), lexical TF-IDF filtering at threshold $\theta = 0.30$ pruned subtle paraphrased contrasting passages, reducing conflicting-class recall from $66.7\%$ to $53.3\%$ and overall accuracy from $0.892$ to $0.875$.

---

## 3. Comprehensive Experimental Comparison Table

All configurations evaluated on the identical held-out test set ($N=120$ claims across 10 unseen document domains).

### Table 1: Complete Configuration Comparison on Held-Out Test Set ($N=120$)

| Metric | Baseline 1 (Raw NLI) | Baseline 2 (Agg Only) | Baseline 3 (Num Only) | Baseline 4 (Complete Ungated) | Proposed (Relevance Gated) | Config A (Fair Ablation: No Num) | Config B (Fair Ablation: With Num) |
|---|---|---|---|---|---|---|---|
| **Overall Accuracy** | 0.725 | 0.892 | 0.725 | **0.892** | 0.875 | **0.892** | **0.892** |
| **Accuracy 95% CI** | [0.708, 0.740] | [0.875, 0.906] | [0.708, 0.740] | [0.875, 0.906] | [0.845, 0.905] | [0.875, 0.906] | [0.875, 0.906] |
| **Macro F1** | 0.626 | 0.889 | 0.626 | **0.889** | 0.868 | **0.889** | **0.889** |
| **Macro F1 95% CI** | [0.608, 0.643] | [0.869, 0.905] | [0.608, 0.643] | [0.869, 0.905] | [0.833, 0.901] | [0.869, 0.905] | [0.869, 0.905] |
| **Supported F1** | 0.750 | 0.857 | 0.750 | **0.857** | 0.811 | **0.857** | **0.857** |
| **Contradicted F1** | 0.938 | 0.952 | 0.938 | 0.952 | **0.984** | 0.952 | 0.952 |
| **Insufficient F1** | 0.818 | 0.947 | 0.818 | 0.947 | **0.983** | 0.947 | 0.947 |
| **Conflicting F1** | 0.000 | **0.800** | 0.000 | **0.800** | 0.696 | **0.800** | **0.800** |
| **False Acceptance Rate** | 0.222 | 0.111 | 0.022* | 0.111 | 0.133 | 0.111 | 0.111 |
| **False Caveat Rate** | 0.000 | 0.111 | 0.289 | 0.111 | 0.156 | 0.111 | 0.111 |
| **State Coverage** | 0.417 | 0.250 | 0.183 | 0.250 | 0.225 | 0.250 | 0.250 |
| **Overall Coverage** | 0.417 | 0.500 | 0.417 | 0.500 | 0.500 | 0.500 | 0.500 |
| **Abstention Rate** | 0.583 | 0.500 | 0.583 | 0.500 | 0.500 | 0.500 | 0.500 |
| **Mean Latency (ms)** | 91.4 ms | 96.4 ms | 91.3 ms | 97.7 ms | 106.7 ms | 99.3 ms | 96.8 ms |

*\*Historical artifact explained in Section 2, Point 3.*

---

## 4. Evaluated Metric Formulations & Denominators

- **Status Accuracy**: $\frac{\sum_{i=1}^{120} \mathbb{I}(y_i = \hat{y}_i)}{120}$. Evaluates agreement between predicted 4-class status and ground truth.
- **Macro F1**: Unweighted arithmetic mean of F1 scores across the 4 classes: Supported ($N=30$), Contradicted ($N=30$), Insufficient ($N=30$), Conflicting ($N=30$).
- **False Acceptance Rate (FAR)**: $\frac{\text{Number of Non-Supported Claims Assigned 'state' Action}}{90 \text{ Non-Supported Claims}}$.
- **False Caveat Rate (FCR)**: $\frac{\text{Number of Non-Conflicting Claims Assigned 'caveat' Action}}{90 \text{ Non-Conflicting Claims}}$.
- **State Coverage**: $\frac{\text{Number of Claims Assigned 'state' Action}}{120}$.
- **Overall Coverage**: $\frac{\text{Number of Claims Assigned 'state' or 'caveat' Action}}{120}$.
- **Abstention Rate**: $\frac{\text{Number of Claims Assigned 'withhold' Action}}{120}$. Note: $\text{Overall Coverage} + \text{Abstention Rate} = 1.000$.

---

## 5. Statistical Methodology & Boundary Limitations

1. **Document-Cluster Bootstrap**:
   - Resampling is executed over the $K = 10$ unique document clusters using 500 resamples and fixed random seed $S = 42$.
   - **Methodological Limitation**: Econometric cluster bootstrap literature recommends $K \ge 30-50$ clusters for exact asymptotic coverage. With $K = 10$ clusters, our 95% confidence intervals are approximate cluster-robust estimates.
2. **Domain Boundaries**:
   - Evaluated strictly on structured synthetic institutional governance documents.
   - Findings do not claim generalization to open-web factual queries, conversational dialogue, OCR noise, or multi-hop tabular math.

