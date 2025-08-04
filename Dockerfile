FROM python:3.11-slim

# Create app user early
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --shell /bin/bash app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimization
ENV HF_HOME=/tmp/huggingface_cache
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache/transformers
ENV HF_HUB_CACHE=/tmp/huggingface_cache/hub
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface_cache/sentence_transformers
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories with proper permissions
RUN mkdir -p /tmp/huggingface_cache/hub \
    /tmp/huggingface_cache/transformers \
    /tmp/huggingface_cache/sentence_transformers \
    && chmod -R 777 /tmp/huggingface_cache

# Set working directory
WORKDIR /home/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Expose port
EXPOSE 7860

# Start command with optimizations
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--loop", "uvloop"]