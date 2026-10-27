# Playground runner: executes notebook code in Jupyter kernels, locked down.
#
# Built on its own (context is this directory, not the repo root), so it
# carries none of the web app's code, keys or dependencies. The lockdown that
# needs Docker flags (network, read-only root, caps, limits) lives in
# docker-compose.yml; docs/PLAYGROUND_RUNNER.md lists all of it.
#
# Same interpreter patch release as the web app's Dockerfile.
FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libseccomp2 backs the kernel-side socket filter (pyseccomp loads it with
# ctypes). It is normally already present in the slim image; installing it
# names the dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY playground_runner/ /opt/runner/playground_runner/
RUN python -m compileall -q /opt/runner

# Build matplotlib's font cache once, here. The root filesystem is read-only at
# run time; the server copies this into MPLCONFIGDIR (on /tmp) at startup so
# the first plot does not spend seconds rebuilding it.
RUN MPLCONFIGDIR=/opt/matplotlib-cache python -c "import matplotlib.font_manager" \
    && chmod -R a+rX /opt/matplotlib-cache

RUN useradd --uid 10001 --user-group --no-create-home --home-dir /work --shell /usr/sbin/nologin runner \
    && mkdir -p /work /data \
    && chown runner:runner /work

ENV PYTHONPATH=/opt/runner \
    HOME=/work \
    MPLCONFIGDIR=/tmp/matplotlib \
    PLAYGROUND_DATA_DIR=/data \
    PLAYGROUND_WORK_DIR=/work \
    PLAYGROUND_RUNTIME_DIR=/tmp/playground-runner \
    PLAYGROUND_KERNEL_SECCOMP=1

USER runner
WORKDIR /work

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4)"]

# One worker on purpose: sessions and their kernels live in this process.
CMD ["uvicorn", "--factory", "playground_runner.app:app_from_env", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
