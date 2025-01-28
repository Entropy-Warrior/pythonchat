# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/storage/models \
    CONFIG_DIR=/app/storage/config \
    HISTORY_DIR=/app/storage/history

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# Setup storage directories
RUN mkdir -p /app/storage/{models,config,history} \
    && chmod -R 777 /app/storage

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pythonchat.py download_model.py ./

# Download model at build time
RUN python download_model.py

# Persistent storage
VOLUME ["/app/storage/models", "/app/storage/config", "/app/storage/history"]

# Run the application
CMD ["python", "pythonchat.py"]