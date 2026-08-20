FROM python:3.11-slim

WORKDIR /app

COPY model/requirements.txt ./model-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        $(grep -v -iE '^(torch|#|$)' model-requirements.txt | tr '\n' ' ') && \
    pip install --no-cache-dir -r backend-requirements.txt

COPY model/ ./model/
COPY backend/ ./backend/
COPY shared/ ./shared/

EXPOSE ${PORT:-8000}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
