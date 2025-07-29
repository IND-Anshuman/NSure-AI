# ---- Stage 1: Build Stage ----
# Use a full Python image to build dependencies.
FROM python:3.11-slim as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
# Corrected: Removed extra period from requirements.txt
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# ---- Stage 2: Final Stage ----
# Use a minimal base image for the final container.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels

# Corrected: Changed "COPY.." to "COPY . ."
# Copy the application code
COPY . .

# Install the dependencies from the wheels
RUN pip install --no-cache /wheels/*

# **THE DEFINITIVE FIX**:
# Pre-download the model during the Docker build process.
# This creates the cache and populates it with the model files.
# The application will then load the model from this cache at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# The command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
