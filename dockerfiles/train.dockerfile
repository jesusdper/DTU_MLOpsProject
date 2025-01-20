# Base image
FROM python:3.11-slim AS base

# Update packages and install necessary dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY src /app/src
COPY configs /app/configs
COPY requirements.txt /app/requirements.txt
COPY requirements_dev.txt /app/requirements_dev.txt
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml

# Install Python dependencies
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements_dev.txt

# Create directories for storing models and processed data
RUN mkdir -p /data/raw /data/processed /models/logs /models/weights

# Ensure correct permissions for directories
RUN chmod -R 777 /data /models

# Set environment variables
ENV PYTHONPATH=/app/src
ENV DATA_DIR=/data
ENV PROCESSED_DIR=/data/processed

# Define entry point for the container
ENTRYPOINT ["python", "-u", "src/object_detection/train.py"]

