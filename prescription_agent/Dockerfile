FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for PyMuPDF/fitz compilation if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE is just documentation — the actual port comes from the $PORT env var set in Render
EXPOSE 8006

# Shell-form CMD (not exec-form ["..."]) so that $PORT is expanded at runtime from Render's env vars.
# Falls back to 8006 if PORT is not set (e.g. local docker run without -e PORT=...)
CMD uvicorn prescription_agent:app --host 0.0.0.0 --port ${PORT:-8006}
