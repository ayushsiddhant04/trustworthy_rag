"""Unit tests for UI views, components, and design system."""
from pathlib import Path
import pandas as pd
import pytest

from src.ui.styles import CUSTOM_CSS
from src.ui.components import (
    status_badge_html,
    action_badge_html,
    format_citation,
    generate_audit_csv,
)
from src.ui.view_demo import SCENARIOS


def test_custom_css_contains_design_tokens():
    assert "font-family" in CUSTOM_CSS
    assert ".er-header" in CUSTOM_CSS
    assert ".er-eyebrow" in CUSTOM_CSS
    assert ".er-badge-supported" in CUSTOM_CSS
    assert ".er-badge-contradicted" in CUSTOM_CSS
    assert ".er-badge-conflicting" in CUSTOM_CSS
    assert ".er-badge-insufficient" in CUSTOM_CSS
    assert ".er-action-state" in CUSTOM_CSS
    assert ".er-action-caveat" in CUSTOM_CSS
    assert ".er-action-withhold" in CUSTOM_CSS


def test_status_and_action_badges():
    # Supported
    sup = status_badge_html("supported")
    assert "er-badge-supported" in sup
    assert "Supported" in sup

    # Contradicted
    con = status_badge_html("contradicted")
    assert "er-badge-contradicted" in con
    assert "Contradicted" in con

    # Conflicting
    cnf = status_badge_html("conflicting")
    assert "er-badge-conflicting" in cnf
    assert "Conflicting" in cnf

    # Insufficient
    ins = status_badge_html("insufficient")
    assert "er-badge-insufficient" in ins
    assert "Insufficient" in ins

    # Actions
    st_act = action_badge_html("state")
    assert "er-action-state" in st_act
    assert "State" in st_act

    cav_act = action_badge_html("caveat")
    assert "er-action-caveat" in cav_act
    assert "Caveat" in cav_act

    wth_act = action_badge_html("withhold")
    assert "er-action-withhold" in wth_act
    assert "Withhold" in wth_act


def test_format_citation():
    # PDF page citation
    pdf_cite = format_citation({"page": 3, "source": "test.pdf"})
    assert pdf_cite == "Page 3"

    # TXT paragraph citation
    txt_cite = format_citation({"paragraph": 5, "source": "test.txt"})
    assert txt_cite == "Paragraph 5"

    # Missing location
    unindexed = format_citation({})
    assert unindexed == "Location unindexed"


def test_generate_audit_csv():
    results = [
        {
            "claim_id": "c_1",
            "claim": "Water boils at 100 C.",
            "status": "supported",
            "action": "state",
            "confidence": 0.998,
            "reason": "Supported by text.",
            "evidence": [
                {"source": "doc.pdf", "page": 1, "text": "Water boils at 100 C."}
            ]
        }
    ]
    csv_bytes = generate_audit_csv(results)
    assert isinstance(csv_bytes, bytes)
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    assert len(df) == 1
    assert df.loc[0, "claim_id"] == "c_1"
    assert df.loc[0, "status"] == "supported"
    assert df.loc[0, "action"] == "state"
    assert df.loc[0, "evidence_citations"] == "Page 1"


def test_demo_centre_scenarios_exist():
    assert len(SCENARIOS) == 3
    for sc in SCENARIOS:
        assert Path(sc["doc_path"]).exists(), f"Demo document {sc['doc_path']} missing"
        assert len(sc["question"].strip()) > 0
        assert len(sc["problematic_answer"].strip()) > 0
        assert "Synthetic Document" in sc["doc_label"]

