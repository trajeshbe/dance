# Use NVIDIA CUDA base image for GPU support
# For CPU-only: FROM python:3.11-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies with increased timeout
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir --default-timeout=300 -r requirements.txt

# Install additional model dependencies from source
RUN pip3 install --no-cache-dir \
    git+https://github.com/AliaksandrSiarohin/first-order-model.git@master#egg=first-order-model || echo "FOMM install skipped" && \
    pip3 install --no-cache-dir \
    git+https://github.com/magic-research/magic-animate.git@main#egg=magic-animate || echo "MagicAnimate install skipped"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
