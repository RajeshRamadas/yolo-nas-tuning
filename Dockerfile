# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install ultralytics

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run training (can be overridden)
CMD ["python", "train_yolo.py", "--data_yaml", "data/dataset.yaml"]
