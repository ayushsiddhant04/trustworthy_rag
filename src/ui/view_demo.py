"""Demo Centre View — Synthetic Test Scenarios for Interactive Verification."""
from __future__ import annotations

import html
from pathlib import Path
import streamlit as st

from src.pipeline import run as run_pipeline
from src.safe_composer import compose_safe_answer
from src.verification_adapter import verify_pipeline_output
from .components import (
    render_page_header,
    status_badge_html,
    action_badge_html,
    format_citation,
)


SCENARIOS = [
    {
        "id": "scenario_1",
        "title": "Scenario 1: Numeric & Date Discrepancy",
        "tag": "Contradiction / Withhold",
        "doc_path": "data/demo/application_notice.txt",
        "doc_label": "Synthetic Document · Fellowship Application Circular 2026",
        "description": "Tests deterministic number and date conflict detection when an ungrounded or hallucinated answer contains false figures.",
        "question": "What is the application submission deadline and fee amount?",
        "problematic_answer": "The application deadline is 30 November 2026 and the registration fee is 500 rupees.",
        "expected_behavior": "The verifier detects that the ground truth specifies 15 November 2026 and ₹1,500. It issues a decisive Contradicted status and Withhold action.",
    },
    {
        "id": "scenario_2",
        "title": "Scenario 2: Missing Fact & Extraneous Claim",
        "tag": "Insufficient / Withhold",
        "doc_path": "data/demo/attendance_policy.txt",
        "doc_label": "Synthetic Document · Academic Attendance Regulation",
        "description": "Tests system abstention when asked about a policy exemption that does not exist in the source document.",
        "question": "Does the attendance policy grant automatic exemptions for sports competitions?",
        "problematic_answer": "Students participating in sports tournaments are automatically exempted from the 75% attendance rule without prior application.",
        "expected_behavior": "While the document discusses the 75% attendance threshold and medical certificates, it never mentions sports exemptions. The verifier assigns Insufficient evidence and Withholds the ungrounded claim.",
    },
    {
        "id": "scenario_3",
        "title": "Scenario 3: Conflicting Regulatory Circulars",
        "tag": "Conflicting / Caveat",
        "doc_path": "data/demo/conflicting_notices.txt",
        "doc_label": "Synthetic Document · Administrative Circulars A & B",
        "description": "Tests multi-source conflict arbitration when two authentic notices within the same institution disagree on an operational date.",
        "question": "What is the deadline for submitting the research proposal?",
        "problematic_answer": "The research proposal submission deadline is 15 November 2026.",
        "expected_behavior": "Notice A specifies 15 November, but Notice B revises it to 20 November. Rather than suppressing both notices or taking an unhedged stance, the verifier assigns Conflicting status and Caveat action, presenting both notices to the user.",
    },
]


def render_demo_centre(get_verifier_fn) -> None:
    """Render the Demo Centre with pre-configured synthetic scenarios."""
    render_page_header(
        eyebrow="Interactive Demonstration & Audit Scenarios",
        title="Demo Centre",
        subtitle="Explore three controlled synthetic scenarios demonstrating deterministic date checks, missing evidence abstention, and conflicting notice arbitration.",
    )

    st.markdown(
        """
        <div class="er-card-muted">
          <strong style="color:#0f172a;">Synthetic Benchmark Guarantee:</strong>
          All scenarios below use synthetic sample documents created specifically to test error detection and conservative abstention.
          All executions run through the actual backend ingestion, retrieval, DeBERTa-v3 cross-encoder, and safe composer modules.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for sc in SCENARIOS:
        with st.container(border=True):
            header_col, tag_col = st.columns([3, 1])
            with header_col:
                st.markdown(f"### {sc['title']}")
                st.caption(sc["doc_label"])
            with tag_col:
                st.markdown(
                    f"<div style='text-align:right;'><span class='er-citation-tag'>{html.escape(sc['tag'])}</span></div>",
                    unsafe_allow_html=True,
                )

            st.write(sc["description"])

            doc_file = Path(sc["doc_path"])
            doc_text = doc_file.read_text(encoding="utf-8") if doc_file.exists() else "Document file missing."

            with st.expander("View Synthetic Document Content", expanded=False):
                st.code(doc_text, language="text")

            col_q, col_a = st.columns(2, gap="medium")
            with col_q:
                st.markdown("**Prepared Investigation Query:**")
                st.info(sc["question"])
            with col_a:
                st.markdown("**Problematic / Candidate Answer:**")
                st.warning(sc["problematic_answer"])

            st.caption(f"**Expected Verification Behavior:** {sc['expected_behavior']}")

            btn_key = f"run_{sc['id']}"
            if st.button(f"Execute {sc['title'].split(':')[0]} Live", type="primary", key=btn_key):
                with st.spinner("Executing real document ingestion, hybrid retrieval, and DeBERTa-v3 verification..."):
                    try:
                        # Real pipeline execution
                        pipe_out = run_pipeline(
                            question=sc["question"],
                            doc=doc_file,
                            answer=sc["problematic_answer"],
                            mode="manual",
                        )

                        engine = get_verifier_fn()
                        verif_out = verify_pipeline_output(pipe_out, engine=engine)
                        results = verif_out.get("verification_results", [])
                        safe_out = compose_safe_answer(results)

                        st.session_state[f"demo_results_{sc['id']}"] = {
                            "pipe_out": pipe_out,
                            "results": results,
                            "safe_out": safe_out,
                        }
                        st.success("Live execution completed.")
                    except Exception as exc:
                        st.error(f"Live execution failed: {exc}")

            # Display results for this scenario if executed
            demo_data = st.session_state.get(f"demo_results_{sc['id']}")
            if demo_data:
                st.markdown("---")
                st.markdown("**Live Execution Findings:**")
                results = demo_data["results"]
                safe_out = demo_data["safe_out"]

                for res in results:
                    st.markdown(
                        f"**Claim:** `{res.get('claim')}` &nbsp; {status_badge_html(res.get('status'))} &nbsp; {action_badge_html(res.get('action'))}",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Reason: {res.get('reason')} | Confidence score: {float(res.get('confidence', 0.0)):.3f}")

                    ev_list = res.get("evidence") or []
                    if ev_list:
                        for ev in ev_list:
                            cite = format_citation(ev)
                            rel = ev.get("relation", "evidence").title()
                            box_class = "er-evidence-box-contradict" if rel.lower() == "contradiction" else "er-evidence-box"
                            st.markdown(
                                f"<div class='{box_class}'><span class='er-citation-tag'>{html.escape(cite)}</span> &nbsp; <strong>{html.escape(ev.get('source', 'Doc'))}</strong> · <em>{html.escape(rel)}</em><br>{html.escape(ev.get('text', ''))}</div>",
                                unsafe_allow_html=True,
                            )

                st.markdown("**Composed Safe Answer:**")
                st.container(border=True).write(safe_out["answer"])
                st.caption(f"Decision: **{safe_out['verdict'].replace('_', ' ').title()}**")

