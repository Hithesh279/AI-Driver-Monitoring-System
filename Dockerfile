# Base Image with Python 3.10 and OpenCV support
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV, MediaPipe, and offline TTS (espeak)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    libespeak1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose web dashboard port
EXPOSE 8000

# Run FastAPI web application in headless container mode
CMD ["python", "main.py", "--no-browser"]
