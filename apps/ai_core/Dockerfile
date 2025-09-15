# Base: small, reliable, with prebuilt wheels for most libs
FROM python:3.11-slim

# Keep Python output unbuffered & pip lean
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal system deps (libgomp helps hnswlib/transformers/torch on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app code
COPY . .

# Make sure a persistent path exists for RAG (compose maps a volume to /data)
RUN mkdir -p /data/rag_store
ENV RAG_PERSIST=/data/rag_store

EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
