"""Document Review View — Document Ingestion, Claim Extraction, and Verification."""
from __future__ import annotations

import html
from pathlib import Path
import streamlit as st

from src.config import SETTINGS
from src.pipeline import run as run_pipeline
from src.safe_composer import compose_safe_answer
from src.verification_adapter import verify_pipeline_output
from .components import (
    render_page_header,
    render_empty_state,
    status_badge_html,
    action_badge_html,
    format_citation,
    generate_audit_csv,
)


def render_document_review(get_verifier_fn) -> None:
    """Render the primary Document Review workspace."""
    render_page_header(
        eyebrow="Document Intelligence & Claim Verification",
        title="Document Review",
        subtitle="Ingest complex regulatory or institutional documents, extract factual claims, and verify groundedness using NLI cross-encoders.",
    )

    # 1. Document Upload & Configuration
    with st.container(border=True):
        st.markdown("**1. Document Ingestion & Query**")
        max_upload_mb = SETTINGS.max_upload_size_bytes / (1024 * 1024)
        max_upload_mb_str = f"{max_upload_mb:.0f}" if max_upload_mb.is_integer() else f"{max_upload_mb:.1f}"
        uploaded = st.file_uploader(
            "Select a document (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            help=f"Text-based documents up to {max_upload_mb_str} MB. Scanned image PDFs without text layers will be reported clearly.",
        )

        oversized = False
        current_sig = f"{uploaded.name}_{uploaded.size}" if uploaded else None
        if current_sig != st.session_state.get("active_doc_signature"):
            st.session_state["active_doc_signature"] = current_sig
            st.session_state.pop("pipeline_output", None)
            st.session_state.pop("verified_output", None)

        if uploaded is not None:
            size_mb = uploaded.size / (1024 * 1024)
            if uploaded.size > SETTINGS.max_upload_size_bytes:
                oversized = True
                st.error(f"Uploaded file ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_upload_mb_str} MB.")
            else:
                st.caption(f"Loaded: `{uploaded.name}` ({size_mb:.2f} MB)")

        question = st.text_input(
            "Query or Investigation Question",
            placeholder="e.g., What is the application deadline and fee requirement?",
            help="The question used to retrieve relevant evidence passages from the document.",
        )

        mode_label = st.radio(
            "Answer Input Mode",
            ["Generate from document", "Paste an answer to audit", "Known-bad answer demonstration"],
            horizontal=True,
            help="'Generate from document' uses extractive retrieval. 'Paste an answer' lets you audit external model answers without configuring an external LLM.",
        )
        mode_map = {
            "Generate from document": "generate",
            "Paste an answer to audit": "manual",
            "Known-bad answer demonstration": "demo_bad",
        }
        mode = mode_map[mode_label]

        supplied_answer = None
        if mode != "generate":
            prompt_label = "Paste the answer to be decomposed and audited" if mode == "manual" else "Paste the deliberately faulty / ungrounded answer"
            supplied_answer = st.text_area(
                prompt_label,
                height=110,
                placeholder="Enter factual statements to verify...",
            )

        submit_disabled = (not uploaded) or oversized or (not question.strip())
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            submitted = st.button("Prepare Answer & Claims", type="primary", disabled=submit_disabled, use_container_width=True)

    if submitted:
        if mode != "generate" and not (supplied_answer or "").strip():
            st.error("Please supply an answer text before proceeding.")
        else:
            with st.spinner("Processing document, executing hybrid retrieval, and extracting claims..."):
                try:
                    output = run_pipeline(question, uploaded, answer=supplied_answer, mode=mode)
                    st.session_state["pipeline_output"] = output
                    st.session_state.pop("verified_output", None)
                    st.success(f"Prepared {len(output.get('claims', []))} claim(s) from `{output.get('source', 'document')}`.")
                except Exception as exc:
                    st.error(f"Pipeline execution failed: {exc}")

    output = st.session_state.get("pipeline_output")
    if not output:
        render_empty_state(
            title="No Active Document Loaded",
            description="Upload a document above and enter a question to extract claims and initiate verification.",
            hint="Supports PDF, TXT, and DOCX formats up to 25 MB. You can also run pre-configured scenarios in the Demo Centre.",
        )
        return

    # 2. Pipeline Decomposition View
    st.markdown("---")
    st.markdown("**2. Extracted Claims & Retrieved Context**")

    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.markdown("**Raw Answer**")
        st.container(border=True).write(output.get("raw_answer", ""))
        st.caption(f"Source: `{output.get('source')}` | Chunks Indexed: {output.get('metadata', {}).get('chunk_count', 0)}")

    with right_col:
        st.markdown("**Extracted Factual Claims**")
        claims_list = output.get("claims", [])
        if claims_list:
            for c in claims_list:
                st.markdown(f"**`{c['claim_id']}`** · {c['claim']}")
        else:
            if output.get("answer_status") == "no_evidence":
                st.info("The document contains no evidence to answer this query. The pipeline abstains from fabricating claims.")
            else:
                st.warning("No discrete factual claims were extracted from the answer.")

    with st.expander(f"Retrieved Evidence Chunks ({len(output.get('retrieved_passages', []))})", expanded=False):
        for p in output.get("retrieved_passages", []):
            citation = format_citation(p)
            st.markdown(f"<span class='er-citation-tag'>{html.escape(citation)}</span> · **Score {p.get('score', 0.0):.3f}**", unsafe_allow_html=True)
            st.markdown(f"<div class='er-evidence-box'>{html.escape(p.get('text', ''))}</div>", unsafe_allow_html=True)

    warnings = output.get("warnings", [])
    if warnings:
        with st.expander(f"Pipeline Diagnostic Notices ({len(warnings)})", expanded=False):
            for w in warnings:
                st.caption(f"• {w}")

    # 3. Verification Execution Controls
    st.markdown("---")
    st.markdown("**3. Claim Verification**")
    
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        enable_relevance = st.checkbox(
            "Apply Relevance Gating (MiniLM pre-filter)",
            value=SETTINGS.enable_relevance_gate,
            help="Filters candidate passages with semantic similarity < 0.30 before cross-encoder scoring.",
        )
    with ctrl_col2:
        use_calibration = st.checkbox(
            "Enable Temperature-Scaled Calibration (member_b/calibration_artifact.json)",
            value=False,
            help="Applies fitted temperature scaling to raw cross-encoder logits.",
        )

    verify_btn = st.button(
        "Verify Claims with Member B",
        type="primary",
        disabled=not output.get("verification_inputs"),
        use_container_width=False,
    )

    if verify_btn:
        with st.spinner("Executing DeBERTa-v3 cross-encoder inference and numeric conflict checks..."):
            try:
                cal_path = "member_b/calibration_artifact.json" if use_calibration else None
                engine = get_verifier_fn(calibration_file=cal_path)
                verified = verify_pipeline_output(
                    output,
                    engine=engine,
                    enable_relevance_gate=enable_relevance,
                )
                st.session_state["verified_output"] = verified
                st.success("Claim verification completed successfully.")
            except Exception as exc:
                st.error(f"Verification execution failed: {exc}")

    verified = st.session_state.get("verified_output")
    if not verified:
        return

    results = verified.get("verification_results", [])
    safe = compose_safe_answer(results)

    # 4. Verification Summary
    st.markdown("---")
    st.markdown("**4. Verification Findings & Trust Summary**")

    counts = {
        "supported": sum(1 for r in results if r.get("status") == "supported"),
        "contradicted": sum(1 for r in results if r.get("status") == "contradicted"),
        "conflicting": sum(1 for r in results if r.get("status") == "conflicting"),
        "insufficient": sum(1 for r in results if r.get("status") == "insufficient"),
    }

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Supported", counts["supported"])
    m2.metric("Contradicted", counts["contradicted"])
    m3.metric("Conflicting", counts["conflicting"])
    m4.metric("Insufficient", counts["insufficient"])
    m5.metric("Safe Policy Verdict", safe["verdict"].replace("_", " ").title())

    # 5. Claim-Level Findings Cards
    st.markdown("**Detailed Claim Findings**")
    for item in results:
        status = item.get("status", "insufficient")
        action = item.get("action", "withhold")
        conf = float(item.get("confidence", 0.0))
        claim_id = item.get("claim_id", "Claim")

        with st.container(border=True):
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown(
                    f"**`{claim_id}`** · {status_badge_html(status)} &nbsp; {action_badge_html(action)}",
                    unsafe_allow_html=True,
                )
            with header_col2:
                conf_label = "Calibrated Probability" if verified.get("is_calibrated") else "Confidence Score"
                st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#64748b;'>{conf_label}: <strong>{conf:.3f}</strong></div>", unsafe_allow_html=True)

            st.markdown(f"**Claim:** {item.get('claim', '')}")
            if item.get("reason"):
                st.caption(f"Analysis: {item.get('reason')}")

            # Decisive evidence windows
            evidence_rows = item.get("evidence") or []
            if evidence_rows:
                st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#475569; margin-top:6px;'>Grounding Evidence:</div>", unsafe_allow_html=True)
                for ev in evidence_rows:
                    cite = format_citation(ev)
                    src = ev.get("source") or "Document"
                    rel = ev.get("relation", "evidence").title()
                    box_class = "er-evidence-box-contradict" if rel.lower() == "contradiction" else "er-evidence-box"
                    st.markdown(
                        f"""
                        <div class='{box_class}'>
                          <span class='er-citation-tag'>{html.escape(cite)}</span> &nbsp;
                          <strong>{html.escape(src)}</strong> · <em>{html.escape(rel)}</em><br>
                          {html.escape(ev.get('text', ''))}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No decisive NLI evidence window met the threshold. Candidate passages are preserved for inspection.")
                for cand in item.get("candidate_evidence", [])[:2]:
                    cite = format_citation(cand)
                    st.markdown(f"<span class='er-citation-tag'>{html.escape(cite)}</span> · <em>Candidate passage</em>", unsafe_allow_html=True)
                    st.markdown(f"<div class='er-evidence-box'>{html.escape(cand.get('text', ''))}</div>", unsafe_allow_html=True)

            filtered_rows = item.get("filtered_evidence") or []
            if filtered_rows:
                with st.expander(f"Passages Filtered by Relevance Gate ({len(filtered_rows)})", expanded=False):
                    for fe in filtered_rows:
                        f_cite = format_citation(fe)
                        st.caption(f"{f_cite} · {fe.get('source') or 'Source'}")
                        st.write(fe.get("text", ""))

    # 6. Before & After Safe Answer Composition
    st.markdown("---")
    st.markdown("**5. Verified Safe Answer Composition**")

    before_col, after_col = st.columns(2, gap="large")
    with before_col:
        st.markdown("**Original Unverified Answer**")
        st.container(border=True).write(output.get("raw_answer", ""))
        if safe["withheld_claims"]:
            st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#991b1b; margin-top:6px;'>Withheld / Disqualified Statements:</div>", unsafe_allow_html=True)
            for w in safe["withheld_claims"]:
                st.markdown(f"<del style='color:#b91c1c; font-size:0.88rem;'>{html.escape(w['claim'])}</del>", unsafe_allow_html=True)

    with after_col:
        st.markdown("**Conservative Verified Answer**")
        st.container(border=True).write(safe["answer"])
        st.caption(f"Composer Decision: **{safe['verdict'].replace('_', ' ').title()}**")

    if safe["withheld_claims"]:
        with st.expander(f"Detailed Justifications for Withheld Claims ({len(safe['withheld_claims'])})", expanded=False):
            for w in safe["withheld_claims"]:
                st.markdown(f"**{w['status'].title()}** — {w['claim']}")
                if w.get("reason"):
                    st.caption(w["reason"])

    st.download_button(
        "Download Audit Trail CSV",
        data=generate_audit_csv(results),
        file_name="evidence_review_audit.csv",
        mime="text/csv",
        use_container_width=False,
    )

