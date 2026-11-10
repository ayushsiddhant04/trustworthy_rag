"""Methodology View — Architectural Reference, Decision Policy, and System Limitations."""
from __future__ import annotations

import streamlit as st
from .components import render_page_header


def render_methodology() -> None:
    """Render the comprehensive technical methodology and architecture documentation."""
    render_page_header(
        eyebrow="Technical Documentation & System Principles",
        title="Methodology & Architecture",
        subtitle="End-to-end specification of document intelligence, hybrid retrieval, cross-encoder NLI verification, decision policy, and scientific limitations.",
    )

    tab_arch, tab_policy, tab_limits = st.tabs([
        "1. Pipeline Architecture",
        "2. Decision Policy & Status Matrix",
        "3. Scientific Limitations & Boundaries",
    ])

    with tab_arch:
        st.markdown("### End-to-End System Architecture")
        st.markdown(
            """
            The **Evidence Review** architecture separates document processing, claim extraction, 
            natural language inference, and safe answer composition into isolated, auditable phases:
            """
        )

        with st.container(border=True):
            st.markdown("#### Phase A: Document Ingestion, Provenance & Retrieval (Member A)")
            st.markdown(
                """
                1. **Multi-Format Ingestion**: Supports text-based PDF (via PyMuPDF), Word DOCX (python-docx), and plain text. Ingestion extracts text with hierarchical structural metadata preserving exact page and paragraph indices.
                2. **Sectioning & Sliding Window Chunking**: Text is segmented into chunks of 220 words with 40-word overlap, recording original page and paragraph provenance tags.
                3. **Hybrid Sparse + Dense Retrieval**: Combines BM25 token matching with optional dense embedding cosine similarity (MiniLM) using reciprocal rank fusion to retrieve top-$k$ relevant evidence passages.
                4. **Extractive Answer Generation & Offline Fallback**: Under offline settings, extracts high-relevance sentences from retrieved chunks. If no evidence meets the threshold, returns an explicit `no_evidence` status to prevent ungrounded claim hallucination.
                5. **Claim Extraction**: Decomposes answers into discrete verifiable propositional claims, stripping hedges and non-factual discourse markers.
                """
            )

        with st.container(border=True):
            st.markdown("#### Phase B: Cross-Encoder NLI & Evidence Aggregation (Member B)")
            st.markdown(
                """
                1. **Token Windowing with Sliding Overlap**: To prevent long-context truncation, long passages are broken into 128-token windows with 32-token overlap using DeBERTa tokenizers.
                2. **DeBERTa-v3 NLI Cross-Encoder**: Evaluates (Premise, Hypothesis) pairs using `cross-encoder/nli-deberta-v3-base`. Produces 3-class logits for *Contradiction*, *Entailment*, and *Neutral*.
                3. **Multi-Passage Evidence Pooling**: Computes the maximum entailment and contradiction scores across all passage windows to form a claim-level score profile.
                4. **Deterministic Numeric & Date Conflict Detection**: Parses dates, monetary amounts, percentages, and cardinal integers from claims and evidence. Detects value mismatches in identical contexts.
                5. **Temperature Scaling Calibration**: Scales raw logits via scalar $T$ fitted on a separate calibration dataset via negative log-likelihood minimization.
                """
            )

        with st.container(border=True):
            st.markdown("#### Phase C: Conservative Decision Policy & Safe Composition (Member C)")
            st.markdown(
                """
                1. **Conservative Action Assignment**: Maps aggregated evidence status to one of three safe actions: `state`, `caveat`, or `withhold`.
                2. **Safe Answer Composer**: Replaces or caveats statements in the final user-facing answer based on verification outcomes. Withheld claims are suppressed with explicit reasons.
                3. **Audit Trail Generation**: Produces line-by-line provenance records linking claims to page and paragraph citations for compliance review.
                """
            )

    with tab_policy:
        st.markdown("### Verification Statuses & Action Matrix")
        st.markdown(
            """
            The verification engine classifies each claim into one of four empirical statuses, 
            which directly determines the safe composition policy:
            """
        )

        matrix_data = [
            {
                "Verification Status": "Supported",
                "NLI Criteria": "Entailment score >= 0.50 and Contradiction < 0.50 with zero numeric conflicts.",
                "Safe Policy Action": "State",
                "User-Facing Treatment": "Asserted normally in the composed answer with exact page/paragraph citations.",
            },
            {
                "Verification Status": "Contradicted",
                "NLI Criteria": "Contradiction score >= 0.50 or contextual date/number mismatch.",
                "Safe Policy Action": "Withhold",
                "User-Facing Treatment": "Suppressed entirely from the answer. Explanatory notice citing contradiction is provided.",
            },
            {
                "Verification Status": "Conflicting",
                "NLI Criteria": "Evidence contains passages that support and passages that contradict the claim.",
                "Safe Policy Action": "Caveat",
                "User-Facing Treatment": "Reported with an explicit caveat explaining that available documents or notices disagree.",
            },
            {
                "Verification Status": "Insufficient",
                "NLI Criteria": "Neither entailment nor contradiction exceeds confidence thresholds.",
                "Safe Policy Action": "Withhold",
                "User-Facing Treatment": "Withheld from final text due to lack of ground truth evidence.",
            },
        ]
        st.dataframe(matrix_data, use_container_width=True, hide_index=True)

        st.markdown("#### Numerical & Date Precedence Rule")
        st.markdown(
            """
            - **Isolated Numeric Mismatch**: If an evidence passage discusses the same subject but specifies a different number or date (e.g. fee is ₹1,500 vs claim stating ₹5,000), the status is `contradicted` and the action is `withhold`.
            - **Genuine Multi-Notice Conflict**: If two separate authentic notices disagree on a date (e.g. Notice A says 15 Nov, Notice B says 20 Nov), the status remains `conflicting` and the action is `caveat`, preserving both passages and alerting the user to the institutional ambiguity.
            """
        )

    with tab_limits:
        st.markdown("### Documented Scientific Limitations & Operational Boundaries")
        st.markdown(
            """
            The Evidence Review system is designed for high auditability and conservative risk mitigation. 
            It is **not** an infallible general fact-checking oracle. Users and evaluators must account for the following boundaries:
            """
        )

        with st.container(border=True):
            st.markdown("##### 1. No Optical Character Recognition (OCR)")
            st.write(
                "The system processes native text streams. Scanned raster images, flattened image PDFs without text layers, "
                "or photographed forms cannot be parsed. The ingestion layer issues an explicit clear error rather than attempting "
                "hallucinatory reconstruction."
            )

        with st.container(border=True):
            st.markdown("##### 2. Uncalibrated Confidence vs. Calibrated Probabilities")
            st.write(
                "Raw softmax outputs from deep neural networks are frequently overconfident. Unless temperature scaling "
                "is explicitly fitted and enabled from a separate calibration artifact, scores must be interpreted as ranking "
                "confidences, never as true Bayesian posterior probabilities."
            )

        with st.container(border=True):
            st.markdown("##### 3. Retrieval Recall Dependence")
            st.write(
                "The verifier operates strictly upon the passages delivered by the retrieval pipeline. If relevant evidence "
                "is missed during ingestion or chunk ranking, the verifier will accurately report 'insufficient evidence' relative "
                "to its context, even if the fact exists elsewhere in an unretrieved portion of a massive document."
            )

        with st.container(border=True):
            st.markdown("##### 4. Internal Document Fidelity vs. External Ground Truth")
            st.write(
                "The system verifies whether a claim is faithfully grounded in the supplied document text. It does not verify "
                "whether the document itself is truthful with respect to the real world. A claim supported by a fraudulent or outdated "
                "document will be classified as supported relative to that document."
            )

        with st.container(border=True):
            st.markdown("##### 5. Syntactic and Semantic Edge Cases")
            st.write(
                "Highly nuanced legal qualifications, complex conditional clauses ('provided that unless X occurs then Y'), "
                "or double negatives may challenge standard NLI cross-encoders. When in doubt, the system errs toward abstention (withhold)."
            )

