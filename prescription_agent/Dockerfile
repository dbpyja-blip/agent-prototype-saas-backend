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

# Expose port 8006 for Render default
EXPOSE 8006

# Use uvicorn directly to bind to port 8006, overriding the hardcoded 8006 in name==main block
CMD ["uvicorn", "prescription_agent:app", "--host", "0.0.0.0", "--port", "8006"]
