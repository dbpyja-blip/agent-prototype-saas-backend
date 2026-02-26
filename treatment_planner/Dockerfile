FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

# Shell-form CMD so $PORT env var (set in Render dashboard) is expanded at runtime.
# Falls back to 8002 if PORT is not set (e.g. local docker run without -e PORT=...).
# Using uvicorn directly (not python treatment_planner.py) so app is imported properly.
CMD uvicorn treatment_planner:app --host 0.0.0.0 --port ${PORT:-8002}
