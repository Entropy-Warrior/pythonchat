# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/storage/models \
    /app/storage/config \
    /app/storage/history \
    && chmod -R 777 /app/storage

# Copy application code
COPY pythonchat.py .

# Set environment variables
ENV MODELS_DIR=/app/storage/models
ENV CONFIG_DIR=/app/storage/config
ENV HISTORY_DIR=/app/storage/history

# Create volume mount points
VOLUME ["/app/storage/models", "/app/storage/config", "/app/storage/history"]

# Set the default command
CMD ["python", "pythonchat.py"] 