"""Professional, restrained styling for Evidence Review.

Employs a clean light corporate aesthetic:
- White / warm-slate surfaces
- High-contrast graphite / dark navy typography
- Muted blue accents
- Subtle 1px borders and disciplined spacing
- Zero gradients, zero neon glows, zero decorative fluff
"""

CUSTOM_CSS = """
<style>
  /* Base layout constraints */
  .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
  }

  /* Typography */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0f172a;
  }

  /* Page Header Styles */
  .er-header {
    margin-bottom: 1.75rem;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1.25rem;
  }
  .er-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.25rem;
  }
  .er-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
    margin: 0.15rem 0 0.35rem 0;
  }
  .er-subtitle {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.5;
    margin: 0;
  }

  /* Section Containers & Cards */
  .er-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
  }

  .er-card-muted {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
  }

  /* Status Badges */
  .er-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    line-height: 1;
  }
  .er-badge-supported {
    background-color: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
  }
  .er-badge-contradicted {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
  }
  .er-badge-conflicting {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
  }
  .er-badge-insufficient {
    background-color: #f8fafc;
    color: #475569;
    border: 1px solid #cbd5e1;
  }

  /* Action Tags */
  .er-action-tag {
    display: inline-flex;
    align-items: center;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    letter-spacing: 0.02em;
  }
  .er-action-state {
    background-color: #dcfce7;
    color: #15803d;
  }
  .er-action-caveat {
    background-color: #fef3c7;
    color: #b45309;
  }
  .er-action-withhold {
    background-color: #fee2e2;
    color: #b91c1c;
  }

  /* Citations & Evidence */
  .er-citation-tag {
    font-size: 0.75rem;
    font-weight: 600;
    color: #1e40af;
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
  }

  .er-evidence-box {
    background-color: #f8fafc;
    border-left: 3px solid #2563eb;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #1e293b;
    line-height: 1.5;
  }

  .er-evidence-box-contradict {
    background-color: #fff1f2;
    border-left: 3px solid #e11d48;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #881337;
    line-height: 1.5;
  }

  /* Streamlit metric override */
  div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 0.85rem 1rem;
    border-radius: 6px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
  }
  div[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
  }

  /* Form & Button cleanups */
  .stButton > button {
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.45rem 1.15rem;
    transition: all 0.15s ease-in-out;
  }
  .stButton > button:hover {
    border-color: #2563eb;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }

  /* Clean tables */
  .dataframe {
    font-size: 0.85rem !important;
    border: 1px solid #e2e8f0 !important;
  }

  /* Empty state box */
  .er-empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    background-color: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    margin: 1.5rem 0;
  }
  .er-empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.35rem;
  }
  .er-empty-desc {
    font-size: 0.85rem;
    color: #64748b;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
  }
</style>
"""

