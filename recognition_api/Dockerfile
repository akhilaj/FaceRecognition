FROM python:3.11-slim

WORKDIR /code

# Install native core layers for headless OpenCV execution
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ \
      git \
      libgl1 \
      libglib2.0-0 \
      libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Step A: Upgrade base pip setup
RUN pip install --no-cache-dir --upgrade pip

# Step B: Install torch first using --extra-index-url flag (Keeps PyPI active as primary fallback)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Step C: Load and install the remaining requirements.txt packages from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Secure non-root routing configuration
RUN useradd -m -u 1000 myuser && chown -R myuser:myuser /code
USER myuser

# Stage source code tree
COPY --chown=myuser:myuser app ./app

# Expose endpoint bound to Render's container routing layer
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]