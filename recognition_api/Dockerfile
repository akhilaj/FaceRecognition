FROM python:3.11-slim-bookworm

WORKDIR /code

# Install essential system libraries for OpenCV and HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ \
      git \
      libgl1 \
      libglib2.0-0 \
      libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch + dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user and give ownership of /code (so writing to app/static/ works)
RUN useradd -m -u 1000 myuser && chown -R myuser:myuser /code
USER myuser

# Copy application code and models
COPY --chown=myuser:myuser app ./app

# Start FastAPI on Render's dynamic $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]