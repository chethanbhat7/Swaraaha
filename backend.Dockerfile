FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

WORKDIR /app

COPY model/requirements.txt ./model-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

RUN uv pip install \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install \
        $(sed 's/[[:space:]]*#.*$//' model-requirements.txt | grep -v -iE '^(torch|$)' | tr '\n' ' ') && \
    uv pip install -r backend-requirements.txt

COPY model/ ./model/
COPY backend/ ./backend/
COPY shared/ ./shared/
COPY start.sh ./start.sh

EXPOSE 8000

CMD ["./start.sh"]