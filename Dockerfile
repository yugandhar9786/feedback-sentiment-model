# =========================
# Stage 1 — Builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /install

# System dependencies needed for torch
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages into a folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =========================
# Stage 2 — Runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api.py .
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
