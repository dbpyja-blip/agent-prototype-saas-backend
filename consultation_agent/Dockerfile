FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build-essential helps compile wheels if pre-built ones don't exist)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["python", "consultation_agent.py"]
