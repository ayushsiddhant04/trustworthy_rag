"""Reusable, accessible UI components for the Evidence Review application."""
from __future__ import annotations

import html
import os
import pandas as pd
import streamlit as st


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a standardized high-hierarchy page header."""
    st.markdown(
        f"""
        <div class="er-header">
          <div class="er-eyebrow">{html.escape(eyebrow)}</div>
          <h1 class="er-title">{html.escape(title)}</h1>
          <p class="er-subtitle">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge_html(status: str) -> str:
    """Return HTML for a status badge with consistent color token."""
    s = str(status).strip().lower()
    label = s.replace("_", " ").title()
    css_class = f"er-badge er-badge-{s}" if s in ("supported", "contradicted", "conflicting", "insufficient") else "er-badge er-badge-insufficient"
    return f'<span class="{css_class}">{html.escape(label)}</span>'


def action_badge_html(action: str) -> str:
    """Return HTML for an action tag (State, Caveat, Withhold)."""
    a = str(action).strip().lower()
    label = a.title()
    css_class = f"er-action-tag er-action-{a}" if a in ("state", "caveat", "withhold") else "er-action-tag er-action-withhold"
    return f'<span class="{css_class}">Policy: {html.escape(label)}</span>'


def render_empty_state(title: str, description: str, hint: str | None = None) -> None:
    """Render a clean, restrained empty state container."""
    hint_html = f'<p style="font-size:0.8rem; color:#94a3b8; margin-top:0.5rem;">{html.escape(hint)}</p>' if hint else ""
    st.markdown(
        f"""
        <div class="er-empty-state">
          <div class="er-empty-title">{html.escape(title)}</div>
          <div class="er-empty-desc">{html.escape(description)}</div>
          {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_citation(item: dict) -> str:
    """Format exact provenance reference (Page {page} for PDF or Paragraph {para} for TXT/DOCX)."""
    page = item.get("page")
    paragraph = item.get("paragraph")
    if page is not None:
        return f"Page {page}"
    if paragraph is not None:
        return f"Paragraph {paragraph}"
    return "Location unindexed"


def _sanitize_csv_cell(val: object) -> object:
    """Neutralize potential CSV/spreadsheet formula injection (OWASP CSV Injection)."""
    if isinstance(val, str) and val.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{val}"
    return val


def generate_audit_csv(results: list[dict]) -> bytes:
    """Generate structured audit CSV bytes for export."""
    rows = []
    for item in results:
        ev_list = item.get("evidence", [])
        sources = "; ".join(sorted({str(e.get("source")) for e in ev_list if e.get("source")}))
        citations = "; ".join(sorted({format_citation(e) for e in ev_list if format_citation(e) != "Location unindexed"}))
        pages = "; ".join(sorted({str(e.get("page")) for e in ev_list if e.get("page") is not None}))

        rows.append({
            "claim_id": _sanitize_csv_cell(item.get("claim_id", "")),
            "claim": _sanitize_csv_cell(item.get("claim", "")),
            "status": _sanitize_csv_cell(item.get("status", "")),
            "action": _sanitize_csv_cell(item.get("action", "")),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
            "reason": _sanitize_csv_cell(item.get("reason", "")),
            "evidence_sources": _sanitize_csv_cell(sources),
            "evidence_citations": _sanitize_csv_cell(citations),
            "evidence_pages": _sanitize_csv_cell(pages),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


_PYARROW_AVAILABLE: bool | None = None


def is_pyarrow_available() -> bool:
    """Check whether PyArrow native C extension can be imported without Windows App Control blocks."""
    global _PYARROW_AVAILABLE
    if _PYARROW_AVAILABLE is None:
        if os.environ.get("FORCE_SAFE_TABLES", "0").lower() in ("1", "true", "yes"):
            _PYARROW_AVAILABLE = False
            return False
        try:
            import pyarrow as pa
            import pyarrow.lib  # Force native C-extension load
            _ = pa.Table.from_pandas(pd.DataFrame({"_c": [1]}))
            _PYARROW_AVAILABLE = True
        except Exception:
            _PYARROW_AVAILABLE = False
    return _PYARROW_AVAILABLE


def render_safe_table(
    data: object,
    use_container_width: bool = True,
    hide_index: bool = True,
    max_rows: int = 100,
) -> None:
    """Render a table via native st.dataframe if PyArrow is functional, or safe HTML table if PyArrow is blocked."""
    if is_pyarrow_available():
        try:
            st.dataframe(data, use_container_width=use_container_width, hide_index=hide_index)
            return
        except Exception:
            pass  # Fall through to HTML table fallback

    if data is None:
        return

    # Extract clean DataFrame
    if hasattr(data, "data") and isinstance(data.data, pd.DataFrame):
        df = data.data.copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (list, tuple, dict)):
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.DataFrame(data)
        except Exception:
            st.write(data)
            return

    if df.empty:
        st.caption("No tabular records to display.")
        return

    total_rows = len(df)
    if total_rows > max_rows:
        display_df = df.head(max_rows).copy()
        row_note = f"<div style='font-size:0.75rem; color:#64748b; padding:0.4rem 0.6rem; border-top:1px solid #f1f5f9; background-color:#f8fafc;'>Showing first {max_rows} of {total_rows} records.</div>"
    else:
        display_df = df.copy()
        row_note = ""

    # Format numeric floats for clean visual presentation
    for col in display_df.select_dtypes(include=["float", "float64", "float32"]).columns:
        display_df[col] = display_df[col].apply(lambda v: f"{v:.3f}" if pd.notnull(v) and isinstance(v, (int, float)) else v)

    html_table = display_df.to_html(
        index=not hide_index,
        escape=True,
        classes="er-table",
        border=0,
    )

    st.markdown(
        f"""
        <div class="er-table-container">
          {html_table}
          {row_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

