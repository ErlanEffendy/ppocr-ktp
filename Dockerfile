FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip/setuptools/wheel first. The slim image's pip can reject the +cpu
# wheel tags and fall back to a source build, which needs flit_core (absent from
# PyTorch's index) and fails. Recent pip accepts the wheels directly.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install CPU-only PyTorch to dramatically reduce image size and build time.
# PyPI is kept as an extra index so any dependency/build backend still resolves.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
