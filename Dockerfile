# ---- Stage 1: Build Stage ----
# Use a full Python image to build dependencies.
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# ---- Stage 2: Final Stage ----
# Use a minimal base image for the final container.
FROM python:3.11-slim

# Create a non-root user for security best practices.
RUN addgroup --system app && adduser --system --group app

# Define the writable cache directory using an environment variable.
# The sentence-transformers library will automatically use this path.
ENV HF_HOME=/data/huggingface_cache

# Create the cache directory and give our app user ownership.
RUN mkdir -p $HF_HOME && chown -R app:app /data

# Set the working directory.
WORKDIR /home/app

# Copy pre-built wheels from the builder stage.
COPY --from=builder /usr/src/app/wheels /wheels

# Copy application code and set ownership.
# **THE FIX**: Corrected "COPY.." to "COPY . ."
COPY --chown=app:app . .

# Install dependencies from local wheels.
RUN pip install --no-cache /wheels/*

# Switch to the non-root user.
USER app

# Expose the port Hugging Face Spaces expects.
EXPOSE 7860

# The command to run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
