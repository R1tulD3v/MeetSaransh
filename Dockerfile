# syntax=docker/dockerfile:1
#
# Multi-stage build. The builder compiles wheels and holds the toolchain; the runtime
# stage gets only the finished virtualenv, so build tools never ship to production.

# ------------------------------------------------------------------------- build stage
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only the dependency manifest first: this layer is cached and re-used across every
# build where the dependencies have not changed, which is most of them.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ----------------------------------------------------------------------- runtime stage
FROM python:3.14-slim AS runtime

# Prefetch the ~90 MB embedding model into the image so the first question after a
# deploy is fast and does not depend on huggingface.co being reachable at runtime.
# Pass --build-arg PREFETCH_MODEL=0 for a lean, quick build (CI does this); the model
# then downloads lazily on first use, exactly as it does in local development.
ARG PREFETCH_MODEL=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production \
    DATA_DIR=/data \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HOME=/opt/models

# Run as an unprivileged user: a container process that does not need root should not
# have it, so a code-execution bug does not immediately own the container.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data /opt/models \
    && chown -R appuser:appuser /data /opt/models

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser data/sample/ ./data/sample/

# Tolerant on purpose: a network hiccup at build time must not fail the build, because
# the application already degrades correctly when the model is unavailable.
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
        || echo "WARNING: model prefetch failed; it will download lazily at runtime"; \
    fi \
    && chown -R appuser:appuser /opt/models

USER appuser

# `data/sample` is baked into the image; `/data` is the writable volume for the database
# and uploaded audio, so container restarts do not lose meetings.
ENV SAMPLE_DIR=/app/data/sample
VOLUME ["/data"]
EXPOSE 8000

# The image has no curl, so the check speaks HTTP from Python. It hits the real health
# endpoint, which round-trips the database rather than just proving the port is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4); \
sys.exit(0 if json.load(r)['status']=='ok' else 1)"

# One worker on purpose: the rate limiter and the migration guard hold per-process
# state, and SQLite has a single writer. Horizontal scaling is unblocked by the
# Redis + Postgres work, not by adding workers here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
