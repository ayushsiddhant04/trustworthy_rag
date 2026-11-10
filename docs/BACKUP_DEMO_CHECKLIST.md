# Competition Backup & Demonstration Contingency Checklist

**Purpose**: Operational procedure and fail-safe recovery manual for competition day, live demonstrations, and network-degraded environments.

---

## 1. Pre-Demonstration System Verification

Execute this checklist **30 minutes prior** to the competition presentation:

| Check Item | Verification Command / Action | Expected Result | Verified? |
|---|---|---|---|
| **Python Virtual Environment** | `.venv\Scripts\python.exe --version` | `Python 3.12.10` | [ ] |
| **Air-Gapped Offline Mode** | `$env:HF_HUB_OFFLINE = "1"`<br>`$env:TRANSFORMERS_OFFLINE = "1"` | Zero network requests to Hugging Face | [ ] |
| **Model Weight Pre-Cache** | Preload DeBERTa in memory | Model loads in $< 3.5$ seconds | [ ] |
| **Full Test Suite Status** | `cmd.exe /c run_tests.bat` | **92 passed, 0 failed** in $\approx 25$ seconds | [ ] |
| **Streamlit Interface Launch** | `cmd.exe /c run_app.bat` | Opens at `http://localhost:8501` | [ ] |
| **Demo Files Integrity** | Verify `data/demo/` directory | All 3 text documents present and readable | [ ] |
| **Results & Calibration Artifacts** | Check `member_b/results/` | `calibration_canonical_artifact.json` and CSVs present | [ ] |

---

## 2. Model Preloading Protocol

To avoid cold-start model loading delays during the live 3-minute pitch, warm up the model cache in memory before opening the presentation:

```powershell
# Preload verification model in background PowerShell terminal:
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'member_b');
from member_b.verifier import VerificationEngine
print('Warming up DeBERTa-v3 model cache...')
eng = VerificationEngine(calibration_file='member_b/results/calibration_canonical_artifact.json')
res = eng.verify_claim('Fee is 500 rupees.', [{'chunk_id': 'warmup', 'text': 'Fee is 500 rupees.', 'source': 'warmup'}])
print('Warmup complete. Status:', res['status'], 'Action:', res['action'])
"
```

---

## 3. Demo Documents Availability & Locations

Ensure the following demonstration files exist in [`data/demo/`](file:///c:/Users/ayush/Downloads/EvidenceReview_Project/trustworthy_rag_rebuild/data/demo/):
1. `application_notice.txt`: Contains official admissions deadline (15 November 2026) and fee (500 rupees).
2. `attendance_policy.txt`: Contains undergraduate attendance requirements and medical condonation rules.
3. `conflicting_notices.txt`: Contains Notice A (15 November) and Notice B (20 November) for conflict arbitration.

---

## 4. Live Failure Contingency & Recovery Runbook

If live Streamlit execution is interrupted or encounters a port binding issue during the presentation, execute the appropriate recovery step:

### Failure Scenario A: Streamlit Port 8501 Already in Use
- **Cause**: A previous Streamlit background process was left running.
- **Recovery Command**:
  ```powershell
  # Kill existing Streamlit process and relaunch:
  Get-Process -Name "python" | Where-Object { $_.CommandLine -like "*streamlit*" } | Stop-Process -Force
  cmd.exe /c run_app.bat
  ```

### Failure Scenario B: Browser Tab Hangs or Does Not Refresh
- **Recovery Action**:
  1. Hard refresh browser tab: `Ctrl + F5` (or `Cmd + Shift + R`).
  2. If session state is lost, navigate directly to **Demo Centre** in the left sidebar. Presets are hardcoded and populate instantly.

### Failure Scenario C: Live GPU/CPU Thermal Throttling
- **Recovery Action**:
  1. The demo scenarios are lightweight ($< 150$ tokens per passage).
  2. If inference latency exceeds 2 seconds, switch to **Tab 3 ("Controlled Cases Suite")** under **Research Evaluation**, which displays pre-evaluated scenario results with identical mathematical grounding.

---

## 5. Export & Audit Trail Verification

Verify that all tabular exports function correctly in the browser:
- [ ] **Document Review Export**: Click *"Download Audit Trail (CSV)"* on a verified document review query. Ensure RFC 4180 compliant CSV downloads.
- [ ] **Research Evaluation Export**: Click *"Download Baseline Comparison CSV"* under Tab 2.
- [ ] **Fair Ablation Export**: Verify `member_b/results/fair_numeric_ablation_summary.csv` opens in Microsoft Excel or Google Sheets without encoding artifacts.

---

## 6. Static Screenshot & Recording Guidelines

> [!NOTE]
> **Recording Status Disclosure**: As of this audit, no pre-recorded backup video file has been saved in the repository. Do not claim a video recording exists during the presentation unless one has been recorded and placed in `docs/screenshots/`.

### Recommended Static Capture Sequence:
Capture high-resolution PNG screenshots into `docs/screenshots/`:
1. `01_document_review_provenance.png`: Document Review view showing citation chips and highlighted evidence card.
2. `02_scenario_1_atomic_split.png`: Scenario 1 showing atomic decomposition (deadline contradicted, fee supported).
3. `03_scenario_2_abstention.png`: Scenario 2 showing explicit `abstain` status and missing evidence warning.
4. `04_scenario_3_conflict_caveat.png`: Scenario 3 showing conflicting notices and side-by-side citations.
5. `05_fair_ablation_dashboard.png`: Research Evaluation Tab 3 showing the fair numeric ablation summary.

### Screen Recording Instructions (If Creating Video Backup):
- **Resolution**: 1920x1080 (1080p), 30 fps or 60 fps.
- **Tool**: OBS Studio or Windows Snipping Tool (Win + Alt + R).
- **Target File**: Save as `docs/backup_demonstration_video.mp4`.
- **Duration**: Exactly 3 minutes following `docs/DEMO_SCRIPT.md`.

