# Evidence Review — Production Deployment & Operations Guide

This guide details the complete deployment lifecycle for **Evidence Review: Document Intelligence & Claim Verification**, covering local workstation setup, containerized Docker execution, and cloud deployment to **Hugging Face Spaces** using the Docker SDK.

---

## 1. System Architecture & Prerequisites

Evidence Review runs as a self-contained, offline-capable Python 3.12 web service orchestrated by Streamlit.

### Production Environment Requirements
- **Runtime**: Python 3.12 (tested on 3.12.10)
- **Host Memory**: $\ge 4$ GB RAM (recommended: 8 GB RAM for batch NLI evaluation)
- **Container Port**: `7860` (Hugging Face Spaces default)
- **Container Host**: `0.0.0.0`
- **Model Checkpoints**:
  - Verification Cross-Encoder: `cross-encoder/nli-deberta-v3-base` (~500 MB)
  - Dense Embedding Pre-filter: `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

---

## 2. Local Setup & Virtual Environment (Tasks A–F)

### A. Local Setup & Repository Clone
Ensure Git is installed and clone the repository:
```bash
git clone <repository_url>
cd trustworthy_rag_rebuild
```

### B. Virtual Environment Creation
Create a clean virtual environment using Python 3.12:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

### C. Dependency Installation
Install production dependencies:
```bash
# Standard production dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Development and testing dependencies (for test harness)
python -m pip install -r requirements-dev.txt
```

### D. Environment Variables
The application is pre-configured with safe defaults in `src/config.py`. To customize behavior, copy `.env.example` to `.env`:
```bash
# Operational settings
APP_ENV=production
MAX_UPLOAD_SIZE_MB=25
MAX_DOC_PAGES=100
ENABLE_RELEVANCE_GATE=false
RELEVANCE_THRESHOLD=0.30

# Offline execution safeguard (prevents spontaneous outbound HuggingFace API calls)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```
> [!CAUTION]
> Never commit `.env` or any file containing API credentials to version control. The repository `.gitignore` explicitly excludes `.env` and `.streamlit/secrets.toml`.

### E. Running the Application Locally
Launch the application using the local batch script or command line:
```bash
# Windows quick launcher
run_app.bat

# Direct execution
streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true
```
Access the application at `http://localhost:7860`.

### F. Running Automated Tests
Execute the complete test suite (92 passing tests):
```bash
# Windows batch harness
run_tests.bat

# Linux / cross-platform harness
pytest tests -q
PYTHONPATH=member_b pytest member_b/tests -q
```

---

## 3. Docker Deployment (Tasks G–H)

### G. Building the Docker Image
The provided `Dockerfile` uses `python:3.12-slim` and applies a non-root container user (`user` with UID 1000) compliant with Hugging Face Spaces security guidelines.

Build the container image:
```bash
docker build -t evidence-review:latest .
```

#### Layer Optimization Strategy
- **Layer 1**: Installs Debian build essentials and CA certificates.
- **Layer 2**: Installs PyTorch CPU wheels from PyTorch's dedicated index, reducing image size by eliminating CUDA dependencies.
- **Layer 3**: Installs application dependencies from `requirements.txt`.
- **Layer 4**: Pre-downloads `cross-encoder/nli-deberta-v3-base` and `sentence-transformers/all-MiniLM-L6-v2` directly into the container cache.
- **Layer 5**: Copies application source code, evaluation datasets, and synthetic demonstration files.

### H. Running Docker Locally
Run the containerized application on port 7860:
```bash
docker run -d --name evidence-review-app -p 7860:7860 evidence-review:latest
```

Verify the running container:
```bash
docker ps
docker logs -f evidence-review-app
```
Open `http://localhost:7860` in your web browser.

---

## 4. DeBERTa Model Deployment Strategy (Task O)

The verification engine uses `cross-encoder/nli-deberta-v3-base`. Three deployment options exist:

| Strategy | Image Size | Startup Time | Memory Footprint | Network Dependency | Resilience |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Baked into Image (Default)** | ~3.8 GB | ~3 seconds | ~1.1 GB RSS | **None** (100% offline at runtime) | **Highest**; immune to HF Hub downtime. |
| **Option B: Download at Boot** | ~2.2 GB | ~90–180 seconds | ~1.1 GB RSS | High (requires HF Hub download on boot) | Medium; fails if rate-limited or offline. |
| **Option C: Persistent Volume Cache** | ~2.2 GB | ~3 seconds (post-init) | ~1.1 GB RSS | Initial download only | High; requires paid persistent volume. |

**Selected Strategy**: **Option A**. The Dockerfile pre-caches model weights during `docker build`. This eliminates cold-start network latency, avoids runtime rate-limits, and guarantees consistent container initialization on Hugging Face Spaces.

If model loading fails for any reason, the system raises an explicit `RuntimeError` describing the failure. **It will never silently substitute a mock model.**

---

## 5. Hugging Face Spaces Deployment (Tasks I–N)

### I. Creating the Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Navigate to **New Space** (`https://huggingface.co/new-space`).
3. Set the **Space name** (e.g., `evidence-review`).
4. Select **License** (e.g., `MIT` or `Apache 2.0`).
5. Choose **Docker** as the Space SDK.
6. Select **Blank** Docker template.
7. Choose the **Free CPU** hardware tier (2 vCPU, 16 GB RAM).

### J. Space Metadata Configuration
Hugging Face Spaces uses YAML frontmatter in `README.md` to identify the runtime. Ensure the following header is at the very top of `README.md`:
```yaml
---
title: Evidence Review
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

### K. Configuring Space Secrets & Variables
1. In the Space repository settings, navigate to **Settings** > **Variables and secrets**.
2. If optional external LLM features are enabled, add secrets here:
   - `ANTHROPIC_API_KEY`: *(Optional, for generative answer mode)*
   - `APP_ENV`: `production`
3. Never commit API keys to Git; manage them exclusively via Space Secrets.

### L. Pushing Repository Files
Initialize or connect your Git remote to the Hugging Face Space:
```bash
# Add Hugging Face Space remote
git remote add space https://huggingface.co/spaces/<your-username>/evidence-review

# Commit production files
git add app.py requirements.txt Dockerfile .dockerignore .streamlit/ data/ docs/ member_b/ src/ README.md
git commit -m "Deploy Evidence Review v1.0 to Hugging Face Spaces"

# Push to Hugging Face Space
git push space main
```

#### Files to Commit for Production
- `app.py`
- `Dockerfile` & `.dockerignore`
- `requirements.txt`
- `.streamlit/config.toml`
- `src/` (all modules)
- `member_b/` (all modules, datasets, and calibration artifacts)
- `data/demo/` (synthetic demonstration circulars)
- `docs/` (documentation and provenance manifests)
- `README.md`

#### Files Excluded from Production Git
- `.venv/`
- `.artifacts/`
- `scratch/`
- `.env`
- `*.pyc` & `__pycache__/`

### M. Monitoring Build Logs
1. Navigate to the **Build** tab on your Hugging Face Space.
2. Observe container build progress:
   - OS dependencies installation.
   - PyTorch CPU installation.
   - Requirements installation.
   - Model checkpoint download and layer caching.
   - Container startup.
3. Once the build completes, the status will transition from `Building` to `Running`.

### N. Testing the Deployed Application
Verify all key functions in the deployed Space:
1. **Document Review**: Upload `data/demo/application_notice.txt` and verify fee extraction.
2. **Demo Centre**: Click "Execute Scenario 1 Live" and verify that the contradicted deadline is withheld while the supported fee is stated.
3. **Research Evaluation**: Verify that the 5-baseline evaluation table and Fair Numeric Ablation tab render correctly with interactive metrics.
4. **Methodology**: Confirm all architectural guidelines and limitation disclosures are readable.

---

## 6. Troubleshooting & Operational Runbook (Task P)

### Issue 1: Port Binding Failure / Container Crashes on Boot
- **Symptom**: Space reports `Error` or `Connection refused on port 7860`.
- **Cause**: Streamlit bound to `localhost` (`127.0.0.1`) instead of `0.0.0.0`, or port was misconfigured.
- **Fix**: Verify `.streamlit/config.toml` contains `address = "0.0.0.0"` and `port = 7860`. Verify `CMD` in Dockerfile specifies `--server.port=7860 --server.address=0.0.0.0`.

### Issue 2: Model Loading Out-of-Memory (OOM)
- **Symptom**: Process killed with exit code `137` during model loading.
- **Cause**: PyTorch GPU version attempted to allocate CUDA memory on CPU-only space.
- **Fix**: Ensure `pip install torch --index-url https://download.pytorch.org/whl/cpu` is used so that PyTorch runs strictly in CPU mode.

### Issue 3: Streamlit Permission Denied on File Writes
- **Symptom**: `PermissionError: [Errno 13] Permission denied` when creating temporary files.
- **Cause**: Container running as non-root user `user` while directories belong to `root`.
- **Fix**: Ensure `COPY --chown=user:user . $HOME/app` is specified in the Dockerfile so that all application files belong to user `user`.

### Issue 4: CSV Formula Injection Warning
- **Symptom**: Exported CSV opens with formula warnings in spreadsheet software.
- **Resolution**: All CSV exports generated via `src/ui/components.py` apply automatic `_sanitize_csv_cell` shielding, prepending `'` to any cell value beginning with `=`, `+`, `-`, `@`, `\t`, or `\r`.

### Issue 5: Uploaded Document Privacy
- **Policy**: Uploaded documents are processed entirely in-memory within the Streamlit session state and are never persisted to public disk volumes or transmitted to third-party endpoints unless external LLM mode is explicitly configured with user-provided API credentials. Resetting the session or uploading a new file automatically flushes previous document buffers.

### Issue 6: CORS & Cross-Origin WebSocket Policy
- **Standard Configuration**: `.streamlit/config.toml` configures `enableCORS = true` and `enableXsrfProtection = true` for local development and containerized deployments.
- **Cross-Origin Iframe Embedding**: If deploying inside an embedded cross-origin iframe where the hosting platform proxies WebSocket traffic from a distinct origin, configure allowed origins explicitly via `server.corsAllowedOrigins = ["https://huggingface.co"]` rather than disabling CORS globally. Never disable XSRF protection (`enableXsrfProtection = true`), which safeguards file upload endpoints against cross-site request forgery.

---

## 7. Vercel Container Deployment (Phase 9)

### Architecture & Runtime Model
Vercel supports Dockerfile-based container deployments via its Container Runtime. Evidence Review deploys to Vercel as a containerized Streamlit service without altering the application architecture or rewriting UI logic into a separate web framework.

### Key Deployment Characteristics

| Dimension | Hugging Face Spaces (`Dockerfile`) | Vercel Container Runtime (`Dockerfile.vercel`) | Local Development |
| :--- | :--- | :--- | :--- |
| **Port Binding** | Fixed port `7860` | Dynamic port `${PORT:-80}` via runtime env | Port `7860` or user-defined |
| **Upload Limit** | 25 MB | **4 MB** (enforces Vercel's 4.5 MB request payload ceiling) | 25 MB |
| **Container User** | Non-root `user` (UID 1000) | Non-root `user` (UID 1000) | Current OS user |
| **Model Caching** | Pre-baked in image build | Pre-baked in image build | Local cache directory |
| **Network Mode** | 100% Offline verification | 100% Offline verification | Offline / Optional LLM |
| **Storage Model** | Ephemeral | Ephemeral | Local filesystem |

### 1. Dynamic Port Binding (`$PORT`)
Unlike Hugging Face Spaces, which routes to a fixed target port (`7860`), Vercel assigns a dynamic port through the standard `PORT` environment variable at container startup.
`Dockerfile.vercel` handles this dynamically via:
```dockerfile
CMD ["sh", "-c", "python -m streamlit run app.py --server.port=${PORT:-80} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true --server.enableXsrfProtection=true --server.maxUploadSize=4"]
```
If `$PORT` is set by Vercel, Streamlit binds to that port; if omitted (e.g. in local Docker testing), it defaults cleanly to standard HTTP port `80`.

### 2. Strict 4 MB Upload Limit (Payload Ceiling Compatibility)
Vercel edge gateways enforce a strict **4.5 MB maximum request body size**. Any multipart file upload exceeding this limit is terminated at the edge with HTTP status 413 (*Payload Too Large*), bypassing application-level handlers.

To guarantee seamless user experience and prevent silent dropped uploads:
- `src/config.py` detects Vercel environments automatically (`VERCEL=1` or `DEPLOYMENT_TARGET=vercel`) and clamps `max_upload_size_bytes` to **4 MB** ($4 \times 1024 \times 1024$ bytes).
- `Dockerfile.vercel` passes `--server.maxUploadSize=4` and `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=4`.
- The UI in `src/ui/view_review.py` dynamically adjusts upload hints and validation messages.
- Local development preserves the full **25 MB** threshold.

### 3. Pre-Cached Offline Model Strategy
Both transformer models:
- Verification Cross-Encoder: `cross-encoder/nli-deberta-v3-base` (~500 MB)
- Dense Embedding: `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

are downloaded and pre-cached inside `/home/user/.cache/huggingface` during the Docker image build stage.
This guarantees:
1. Immunity to external repository downtime or Hugging Face rate limits.
2. Completely offline, zero-network runtime verification.
3. Rapid process startup without boot-time file downloads.

#### Measured Startup & Initialization Breakdown
- **Streamlit Process Startup**: ~2.1 seconds (loads application modules and launches UI).
- **Container Cold Start**: ~1–2 seconds (initial container allocation).
- **Model Initializations (Lazy On-Demand)**:
  - First MiniLM dense embedding initialization: ~20.0 seconds (occurs on first embedding retrieval call).
  - First DeBERTa-v3 cross-encoder initialization: ~1.3 seconds (occurs on first claim verification call).
- **Subsequent Claim Verification**: Sub-second execution (~0.4s–1.5s per claim or scenario) once models reside in process memory.

### 4. WebSocket & Streamlit Protocol Handling
Streamlit communicates between the browser client and backend via a bidirectional WebSocket connection (`/_stcore/stream`).
- `Dockerfile.vercel` configures `--server.enableCORS=true` and `--server.enableXsrfProtection=true`.
- Vercel Container Runtime routes HTTP and WebSocket connections through its reverse proxy. Note that long-lived WebSocket connections are subject to platform connection lifecycle and scale-to-zero timeouts. Because Evidence Review processes files in-memory per session without stateful database locks, fresh sessions connect immediately upon page reload.

### 5. Zero-Database Ephemeral Lifecycle
Vercel container instances are stateless and ephemeral. Evidence Review requires no PostgreSQL, Redis, or local disk volume; document text, chunk embeddings, and NLI verification scores reside in memory within `st.session_state`. This eliminates database configuration overhead and matches Vercel's operational model.

### 6. Deployment Procedure
To deploy using the Vercel CLI:
```bash
# Link project and deploy container
vercel --prod
```
Ensure your Vercel project configuration points to `Dockerfile.vercel` as its container build target.



