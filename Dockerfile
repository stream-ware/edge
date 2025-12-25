# Streamware Orchestrator Dockerfile
# Optimized multi-stage build with layer caching

# ============================================
# Stage 1: Build dependencies (cached layer)
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy ONLY requirements first (layer caching)
COPY requirements-docker.txt ./requirements.txt

# Build wheels (cached if requirements unchanged)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt

# ============================================
# Stage 2: Runtime image (minimal)
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install pre-built wheels (fast)
COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application code (changes most often - last layer)
COPY orchestrator/ ./orchestrator/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 streamware && chown -R streamware:streamware /app
USER streamware

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "print('ok')" || exit 1

# Entry point
CMD ["python", "-m", "orchestrator.main"]
