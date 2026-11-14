FROM python:3.11-slim-bookworm

WORKDIR /code

# Install native image system layers for headless OpenCV execution
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ \
      git \
      libgl1 \
      libglib2.0-0 \
      libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Force install lightweight CPU PyTorch wheels directly from the official index url
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://pytorch.org

# Copy and install the rest of your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a secure non-root user
RUN useradd -m -u 1000 myuser && chown -R myuser:myuser /code
USER myuser

# Stage source code tree
COPY --chown=myuser:myuser app ./app

# Expose web service endpoint bound dynamically to Render's routing layer
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
