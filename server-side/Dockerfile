# Use Ubuntu
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libomp5 \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate venv by default
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency list first (better Docker caching)
COPY requirements.txt .

# Upgrade pip & install deps INTO VENV
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY models ./models

# Expose Flask port
EXPOSE 8000

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run server using venv Python
CMD ["python", "server.py"]
