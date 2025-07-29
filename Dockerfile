# Use a standard Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements and install all dependencies
COPY requirements.txt.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# **THE DEFINITIVE FIX**:
# Pre-download and save the model to a known, fixed path during the build.
# This creates the model files directly inside the image.
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('./embedding_model')"

# Copy the rest of the application code
COPY..

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# The command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
