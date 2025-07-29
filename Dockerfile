# ---- Stage 1: Build Stage ----
# Use a full Python image to build dependencies, which may have system requirements.
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /usr/src/app

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install build-time dependencies
RUN pip install --upgrade pip

# Copy requirements and install project dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# ---- Stage 2: Final Stage ----
# Use a minimal base image for the final container to reduce size and attack surface.
FROM python:3.11-slim

# Create a non-root user for security best practices
RUN addgroup --system app && adduser --system --group app

# Set the working directory
WORKDIR /home/app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels

# **THE DEFINITIVE FIX**: Use the --chown flag during COPY.
# This ensures all application files and directories are owned by the 'app' user
# from the moment they are created, solving any permission issues.
COPY --chown=app:app . .

# Install the dependencies from the wheels without hitting the network again
RUN pip install --no-cache /wheels/*

# Switch to the non-root user
USER app

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# The command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
