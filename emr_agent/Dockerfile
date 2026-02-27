FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr  : OCR engine used by pytesseract
# poppler-utils  : pdf2image needs pdftoppm from poppler
# libgl1         : required by some image-processing wheels (OpenCV transitive dep)
# build-essential: compiles any packages that ship without pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Persistent data directory — mount a volume here in production so patient data
# and ChromaDB embeddings survive container restarts.
RUN mkdir -p /app/emr_data/chroma

# Port 8010 is the fixed port for this service (matches emr_agent.py uvicorn config)
EXPOSE 8010

# Shell-form CMD so ${PORT:-8010} is expanded at runtime.
# Render / Fly.io set $PORT automatically; local docker run uses the fallback 8010.
CMD uvicorn emr_agent:app --host 0.0.0.0 --port ${PORT:-8010}
