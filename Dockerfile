# Dockerfile optimisé pour CUDA 12.6
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.6 support
RUN pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install sentencepiece for SeamlessM4T tokenization
RUN pip install sentencepiece

# Copy application code
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FASTAPI_DEBUG=False \
    USE_GPU=True \
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8000 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create necessary directories
RUN mkdir -p /app/audio_files /app/output_files /app/logs

# Expose ports
EXPOSE 8000 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Command to run the application
CMD ["python", "main.py", "--both"]