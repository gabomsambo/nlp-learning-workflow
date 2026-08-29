# NLP Learning Workflow - Production Dockerfile
#
# Pinned to a patch release, not the floating `3.12-slim` tag: the whole point
# of pinning requirements is that a rebuild resolves to the same stack, and a
# moving interpreter undoes half of that. 3.12 (not 3.11) is the real floor —
# atomic-agents requires >= 3.12.
FROM python:3.12.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libpq-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements.lock.txt ./

# Install Python dependencies.
#
# Both files are passed to a single pip invocation on purpose. The lock file is
# the authoritative transitive set; requirements.txt is the human-facing direct
# list. Resolving them together means pip fails the build if the two ever drift
# apart, instead of the lock quietly winning.
#
# REQS exists only for the lock-regeneration recipe documented in
# requirements.txt: build with `--build-arg REQS=requirements.txt` to resolve
# the direct pins fresh, then `pip freeze` the result into the lock.
ARG REQS="requirements.lock.txt requirements.txt"
RUN pip install --no-cache-dir $(for f in $REQS; do echo -n "-r $f "; done)

# Copy application code
COPY nlp_pillars/ ./nlp_pillars/
COPY webui/ ./webui/
COPY scripts/ ./scripts/

# Retained uploaded PDFs live here (docker-compose backs it with the
# nlp_uploads named volume). Created in the image, before the chown below, so a
# fresh empty named volume is seeded with appuser ownership — Docker copies the
# image directory's contents *and* permissions into it, and a root-owned mount
# point would make every upload fail for the non-root user.
RUN mkdir -p /app/data/uploads /app/data/podcast_audio /app/data/tts-downloads /app/data/tts-previews

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for WebUI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the WebUI
CMD ["uvicorn", "webui.app:app", "--host", "0.0.0.0", "--port", "8000"]

