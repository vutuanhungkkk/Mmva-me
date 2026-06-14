# ─────────────────────────────────────────────────────────────────
# Base image: NVIDIA CUDA 12.6 + cuDNN on Ubuntu 22.04
# Switched from 12.1 to 12.6 to match local cu126 torch builds
# ─────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# ── Build arguments ──
ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

# ── Set timezone to avoid interactive prompt ──
ENV TZ=Asia/Ho_Chi_Minh

# ─────────────────────────────────────────────────────────────────
# Install system dependencies
# - ffmpeg: required for audio processing
# - portaudio19-dev: required for PyAudio / speech recognition
# - libsndfile1: required for soundfile library
# - software-properties-common: needed to add Python PPA
# ─────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libgomp1 \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Set python3.11 as default python ──
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

# ─────────────────────────────────────────────────────────────────
# Set working directory inside container
# ─────────────────────────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────────────────────────
# Copy requirements first for better layer caching
# ─────────────────────────────────────────────────────────────────
COPY requirements.txt .

# ── Upgrade pip core tools ──
RUN pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.6 from official PyTorch index
# FIX 1: torch==2.12.0+cu126 does NOT exist on any public index
#         2.6.0 is the latest STABLE release for cu126
# NOTE: torch/torchvision/torchaudio are removed from requirements.txt
#       and installed here separately with the correct index URL
# ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# ─────────────────────────────────────────────────────────────────
# FIX 2: Remove system distutils-installed packages BEFORE pip install
# Ubuntu 22.04 pre-installs blinker 1.4 and distro 1.7.0 via apt
# These use distutils format → pip cannot uninstall them normally
# Solution: remove via apt first, then pip installs clean versions
# ─────────────────────────────────────────────────────────────────
RUN apt-get remove -y \
    python3-blinker \
    python3-distro \
    2>/dev/null || true

# ── Install remaining project requirements ──
# torch/torchvision/torchaudio already satisfied above
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# Copy application source code only
# models/, documents/, vector_db/ are excluded via .dockerignore
# and will be bind-mounted at runtime via docker-compose volumes
# ─────────────────────────────────────────────────────────────────
COPY app.py .
COPY run.py .
COPY README.md .
COPY assistant/ ./assistant/

# ─────────────────────────────────────────────────────────────────
# Create mount point directories
# These will be populated by Docker volume mounts at runtime
# ─────────────────────────────────────────────────────────────────
RUN mkdir -p models documents vector_db

# ─────────────────────────────────────────────────────────────────
# Environment variables
# NOTE: Inline comments are NOT allowed inside ENV blocks in Docker
#       They are placed above each group instead
# ─────────────────────────────────────────────────────────────────

# Python runtime settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Streamlit server settings
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# CUDA / NVIDIA settings
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── Expose Streamlit default port ──
EXPOSE 8501

# ─────────────────────────────────────────────────────────────────
# Health check: verify the app is responding
# start-period=60s accounts for model loading time on first start
# ─────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ─────────────────────────────────────────────────────────────────
# Default command: run Streamlit app
# ─────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]