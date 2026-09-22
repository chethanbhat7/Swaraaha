# Swaraaha backend — production image for Zoho Catalyst AppSail custom runtime.
# Build: docker buildx build --platform linux/amd64 -t swaraaha-backend:<tag> -f deploy/Dockerfile .
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY model/requirements.txt ./model-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

RUN uv pip install \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install \
        $(sed 's/[[:space:]]*#.*$//' model-requirements.txt | grep -v -iE '^(torch|$)' | tr '\n' ' ') && \
    uv pip install -r backend-requirements.txt

COPY model/ ./model/
COPY backend/ ./backend/
COPY shared/ ./shared/

# Replace dev registry (references missing per-class/CNN checkpoints) with
# the deploy-trimmed copy containing only shipped entries.
COPY deploy/registry.deploy.json ./model/registry.json

# Bake whisper-tiny caches so requests never hit the HF Hub.
COPY deploy/bake_whisper.py /tmp/bake_whisper.py
RUN python /tmp/bake_whisper.py && rm /tmp/bake_whisper.py

EXPOSE 9000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-9000}"]