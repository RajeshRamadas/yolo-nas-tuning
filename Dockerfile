FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create directories for data and model outputs
RUN mkdir -p data runs logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AWS_REGION=us-east-1

# Expose port for NNI web UI
EXPOSE 8080

# Default command
CMD ["/bin/bash"]