# ==============================================================================
# Dockerfile for Vision Builder Core (Edge-to-Cloud Scientific CV-Ops Engine)
# Course: Applied Software Development Project / Project Management (Assignment 5)
# Author: Andasov Temirlan (CSE-2507M, Astana IT University)
# ==============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="Andasov Temirlan <255395@astanait.edu.kz>" \
      org.opencontainers.image.title="vision-builder-core" \
      org.opencontainers.image.description="Edge-to-Cloud Computer Vision Operations (CV-Ops) & LLM Incident Synthesis Engine" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/DaldenU/vision-builder-core"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal OS libraries required by OpenCV headless & healthcheck utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml requirements.txt requirements-dev.txt README.md LICENSE ./

# Copy source package and unit test suite
COPY vision_builder ./vision_builder
COPY tests ./tests

# Install package and testing/QA toolchain
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .[dev]

# Create a non-root user for secure container execution (DevSecOps best practice)
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose HTTP port for CD test server & Kubernetes liveness/readiness probes
EXPOSE 8000

# Container healthcheck querying the built-in CV-Ops health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default command runs the lightweight CV-Ops HTTP health & inference microservice
CMD ["python", "-m", "vision_builder.server", "--host", "0.0.0.0", "--port", "8000"]
