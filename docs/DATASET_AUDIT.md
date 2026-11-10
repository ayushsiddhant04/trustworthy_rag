# Dataset Audit and Empirical Provenance Report

**Project**: Evidence Review — Document Intelligence & Claim Verification  
**Phase**: Phase 6 — Research Dataset Expansion & Scientific Validation  
**Date**: November 2026  
**Status**: Completed Audit

---

## 1. Executive Summary

This audit establishes a rigorous inventory of all evaluation data, ground-truth labels, and empirical artifacts across the repository. Prior to this audit, benchmark claims were often evaluated on repetitive single-source snippets or distorted variants of a very small set of underlying claims.

### Key Inventory Findings

| Dataset File | File Location | Rows | Unique Claims | Evidence Passages | Class Distribution | Document Independence |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Multi-Passage Benchmark** | `member_b/data/benchmark_multipassage.csv` | 12 | 12 | 24 | Supported: 4<br>Contradicted: 3<br>Insufficient: 3<br>Conflicting: 2 | 12 synthetic academic circulars |
| **Calibration Set** | `member_b/data/calibration_set.csv` | 12 | 12 | 12 | Supported: 4<br>Contradicted: 4<br>Insufficient: 4<br>Conflicting: 0 | Single-passage factual statements |
| **Evidence Shift 30** | `member_b/evidence_shift_30.csv` | 30 | **6** | 30 | Insufficient: 16<br>Contradicted: 10<br>Supported: 4<br>Conflicting: 0 | **6 underlying claims** evaluated across 5 perturbation conditions |
| **Benchmark Template** | `member_b/benchmark_template.csv` | 6 | 6 | 6 | Supported: 2<br>Contradicted: 2<br>Insufficient: 2<br>Conflicting: 0 | 6 base template claims |
| **Demo Documents** | `data/demo/*.txt` | 3 docs | 3 claims | 7 paragraphs | Contradicted: 1<br>Insufficient: 1<br>Conflicting: 1 | 3 interactive demo files |

---

## 2. Granular Dataset Inspection

### 2.1 Multi-Passage Benchmark (`member_b/data/benchmark_multipassage.csv`)
- **Structure**: Extended multi-passage schema containing `example_id`, `claim`, `evidence_passages` (JSON string array), and `true_label`.
- **Passage Statistics**:
  - Total claims: 12
  - Total passages: 24 (average 2.0 passages per claim; range 1 to 3)
  - Passage sources: `fellowship_circular.pdf`, `grad_handbook.pdf`, `ieee_guidelines.pdf`, `orientation_notice_a.pdf`, `orientation_notice_b.pdf`, `grant_award.pdf`, `attendance_regulations.pdf`, `colloquium_schedule.pdf`, `project_guidelines_v1.pdf`, `project_guidelines_v2.pdf`, `robotics_inventory.pdf`, `hostel_rules.pdf`, `gym_brochure.pdf`, `acm_program.pdf`.
- **Strengths**: Successfully tests multi-passage pooling and genuine institutional conflicts (`mp_04`, `mp_08`).
- **Deficiencies**: Small sample size ($N=12$) limits statistical significance; only 2 conflicting examples.

### 2.2 Calibration Set (`member_b/data/calibration_set.csv`)
- **Structure**: Single-passage pairs (`example_id`, `claim`, `evidence`, `true_label`).
- **Content**: General knowledge and policy facts (e.g. Earth's orbit, water boiling point, conference fees, library hours).
- **Deficiencies**:
  - Sample size is strictly $N=12$.
  - Lacks representation of the `conflicting` class (0 examples).
  - Fitting temperature scaling on 12 pairs ($T=3.6815$) exhibits high sample variance.

### 2.3 Evidence Shift Benchmark (`member_b/evidence_shift_30.csv`)
- **Structure**: 30 rows evaluating robustness under controlled evidence distortion conditions:
  - Condition 1: Direct authentic evidence
  - Condition 2: Numeric/date perturbation
  - Condition 3: Topically irrelevant text
  - Condition 4: Lexical overlap without entailment
  - Condition 5: Incomplete / partial passage
- **Critical Audit Finding**:
  - **The 30 rows represent only 6 unique claims**, each evaluated 5 times under varying conditions.
  - While scientifically valuable as a controlled stress test, reporting "Accuracy on 30 test examples" creates an illusion of sample size if treated as 30 independent evaluation queries.
  - The conflicting class is entirely absent (0/30).

---

## 3. Data Leakage and Overlap Analysis

1. **Demo Document Leakage**:
   - The 3 demonstration documents in `data/demo/`:
     - `application_notice.txt`
     - `attendance_policy.txt`
     - `conflicting_notices.txt`
   - Content and numeric figures from these demo files were borrowed to create the 6 base claims in `evidence_shift_30.csv` and `benchmark_template.csv`.
   - *Requirement for Phase 6*: Demonstration files must be strictly quarantined from the research evaluation splits. Evaluation metrics must never be reported on demo documents.
2. **Train / Calibration / Test Leakage**:
   - Claims in `member_b/data/calibration_set.csv` do not share claim strings with `member_b/data/benchmark_multipassage.csv`.
   - However, because both calibration and evaluation sets are synthetic, independent document partitioning was not previously enforced.

---

## 4. Identified Gaps & Objectives for Phase 6 Expansion

1. **Sample Size Deficiency**:
   The entire existing repository contains only 30 unique evaluation claims across all files. To defend empirical findings with bootstrap confidence intervals, the evaluation set must be expanded to **150–200 independently labelled examples**.
2. **Severe Underrepresentation of Institutional Conflicts**:
   Only 2 conflicting examples exist in the multi-passage benchmark, and 0 in evidence shift or calibration sets. The expanded dataset must include **at least 30 genuine conflicting examples** with authentic multi-notice provenance.
3. **Canonical Unified Schema**:
   Existing datasets mix single-passage CSVs with multi-passage JSON strings. Phase 6 must establish a single canonical schema supporting document provenance, passage relationships (`entails`, `contradicts`, `neutral`, `irrelevant`), and atomic claim tracking.
4. **Grouped Document Splitting**:
   Splits must group by `document_id` to guarantee zero passage leakage between calibration, validation, and held-out testing.

