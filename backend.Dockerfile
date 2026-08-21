FROM python:3.11-slim

WORKDIR /app

COPY model/requirements.txt ./model-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        $(sed 's/[[:space:]]*#.*$//' model-requirements.txt | grep -v -iE '^(torch|$)' | tr '\n' ' ') && \
    pip install --no-cache-dir -r backend-requirements.txt

COPY model/ ./model/
COPY backend/ ./backend/
COPY shared/ ./shared/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
