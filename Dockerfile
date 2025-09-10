# Multi-stage Dockerfile for PS-06 Competition System
# Supports both development and production builds

# ==============================================================================
# Base Stage - Common dependencies
# ==============================================================================
# Parameterize CUDA/Ubuntu to pin to valid tags
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    software-properties-common \
    # Audio processing dependencies
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    # System utilities
    htop \
    vim \
    nano \
    tree \
    # Network tools
    net-tools \
    iputils-ping \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install poetry
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install poetry==1.7.1

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# ==============================================================================
# Dependencies Stage - Install Python packages
# ==============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy dependency files
COPY requirements.txt .
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages via poetry if available
RUN if [ -f pyproject.toml ]; then poetry install --only=main --no-dev; fi

# Download and setup models directory
RUN mkdir -p /app/models /app/data /app/logs /app/configs

# ==============================================================================
# Development Stage
# ==============================================================================
FROM dependencies AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipython

# Install poetry dev dependencies if available
RUN if [ -f pyproject.toml ]; then poetry install; fi

# Create non-root user for development
RUN useradd -m -u 1000 developer && \
    chown -R developer:developer /app

USER developer

# Set up development environment
COPY --chown=developer:developer . /app/

# Expose ports
EXPOSE 8000 5555 8888

# Default command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ==============================================================================
# Production Build Stage
# ==============================================================================
FROM dependencies AS build

# Copy source code
COPY . /app/

# Create optimized Python bytecode
RUN python3 -m compileall /app/src

# Remove unnecessary files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -exec rm -rf {} + || true && \
    find /app -type f -name "*.md" -delete && \
    rm -rf /app/tests /app/.git /app/.github

# ==============================================================================
# Runtime Stage - Final production image
# ==============================================================================
FROM base AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from build stage
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Create application user
RUN useradd -m -u 1000 ps06user && \
    mkdir -p /app/models /app/data /app/logs /app/configs && \
    chown -R ps06user:ps06user /app

# Copy application code
COPY --from=build --chown=ps06user:ps06user /app /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER ps06user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ==============================================================================
# Model Download Stage (for CI/CD)
# ==============================================================================
FROM runtime AS model-download

USER root

# Install git-lfs for large model files
RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy model download script
COPY scripts/download_models.sh /app/scripts/
RUN chmod +x /app/scripts/download_models.sh

# Create models directory and set permissions
RUN mkdir -p /app/models && \
    chown -R ps06user:ps06user /app/models

USER ps06user

# Download models (this stage can be built separately)
RUN /app/scripts/download_models.sh

# ==============================================================================
# Testing Stage
# ==============================================================================
FROM development AS testing

# Copy test configuration
COPY tests/ /app/tests/
COPY pytest.ini /app/
COPY .coveragerc /app/

# Install additional test dependencies
RUN pip install --no-cache-dir \
    coverage \
    pytest-xdist \
    pytest-benchmark \
    pytest-timeout

# Run tests (can be overridden)
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# ==============================================================================
# GPU Optimization Stage
# ==============================================================================
FROM runtime AS gpu-optimized

USER root

# Install additional GPU libraries
RUN pip install --no-cache-dir \
    nvidia-ml-py3 \
    cupy-cuda12x \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12

# Optimize for specific GPU architectures
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0

USER ps06user

# ==============================================================================
# Production Multi-GPU Stage
# ==============================================================================
FROM gpu-optimized AS multi-gpu

USER root

# Install multi-GPU support
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-geometric

# Configure for multiple GPUs
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NCCL_DEBUG=INFO

USER ps06user

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ==============================================================================
# Build Arguments and Labels
# ==============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="PS-06 Competition System" \
    org.label-schema.description="Language Agnostic Speaker Identification & Diarization System" \
    org.label-schema.url="https://github.com/your-org/ps06-system" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/your-org/ps06-system" \
    org.label-schema.vendor="PS-06 Team" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0"
