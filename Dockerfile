# Production Dockerfile for Evidence Review (Hugging Face Spaces - Docker SDK)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Install minimal OS runtime packages required for document parsing and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root container user according to Hugging Face Spaces specifications (UID 1000)
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/.cache/huggingface && \
    mkdir -p /home/user/.cache/torch && \
    chown -R user:user /home/user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub \
    TORCH_HOME=/home/user/.cache/torch

WORKDIR $HOME/app

# Switch to non-root user so all dependencies and model caches are user-owned
USER user

# Copy dependency definition
COPY --chown=user:user requirements.txt $HOME/app/

# Install PyTorch CPU and production dependencies in user space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download both models directly into user-owned HF_HOME during build
# Guarantees instant startup and eliminates runtime Hugging Face Hub dependencies
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
    CrossEncoder('cross-encoder/nli-deberta-v3-base'); \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source code, configuration, demo fixtures, and research artifacts
COPY --chown=user:user . $HOME/app

# Expose standard Hugging Face Spaces application port
EXPOSE 7860

# Launch production Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
