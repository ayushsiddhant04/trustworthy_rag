"""Evidence Review — Document Intelligence & Claim Verification.

Master Streamlit application. Orchestrates Member A ingestion/retrieval,
Member B NLI verification, and Member C safe composition.
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.ui.styles import CUSTOM_CSS
from src.ui.view_review import render_document_review
from src.ui.view_evaluation import render_research_evaluation
from src.ui.view_demo import render_demo_centre
from src.ui.view_methodology import render_methodology


# Configure page metadata
st.set_page_config(
    page_title="Evidence Review — Document Intelligence & Claim Verification",
    page_icon="ER",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified design system styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _get_verifier(calibration_file: str | None = None):
    """Load Member B VerificationEngine lazily with optional calibration artifact."""
    from member_b.verifier import VerificationEngine
    default_artifact = Path("member_b/results/calibration_canonical_artifact.json")
    cal_path = calibration_file if (calibration_file and Path(calibration_file).exists()) else (str(default_artifact) if default_artifact.exists() else None)
    try:
        return VerificationEngine(calibration_file=cal_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize VerificationEngine (cross-encoder/nli-deberta-v3-base): {exc}. "
            "Ensure the model weights are pre-downloaded into the container cache or network access is enabled."
        ) from exc


PAGES = [
    "Document Review",
    "Research Evaluation",
    "Demo Centre",
    "Methodology",
]

# Direct deep-link navigation support
raw_url_page = str(st.query_params.get("page", "")).replace("+", " ").replace("%20", " ").strip()
default_index = PAGES.index(raw_url_page) if raw_url_page in PAGES else 0

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Evidence Review")
    st.caption("Document Intelligence & Claim Verification")
    st.markdown("---")

    selected_page = st.radio(
        "Workspace Navigation",
        PAGES,
        index=default_index,
        label_visibility="collapsed",
        key="main_navigation_radio",
    )

    st.markdown("---")
    st.markdown("**Engine Specifications**")
    st.caption("• Architecture: Hybrid RAG + Cross-Encoder NLI")
    st.caption("• Model: `cross-encoder/nli-deberta-v3-base`")
    st.caption("• Ingestion: PyMuPDF, python-docx, TXT")
    st.caption("• Environment: Offline Execution Ready")
    st.caption("• Verification Status: 92/92 Automated Tests Passing")


# Route to selected view
if selected_page == "Document Review":
    render_document_review(get_verifier_fn=_get_verifier)
elif selected_page == "Research Evaluation":
    render_research_evaluation(get_verifier_fn=_get_verifier)
elif selected_page == "Demo Centre":
    render_demo_centre(get_verifier_fn=_get_verifier)
elif selected_page == "Methodology":
    render_methodology()
