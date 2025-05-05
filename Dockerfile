# Use NVIDIA CUDA image as base for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip3 install --no-cache-dir jupyter

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose Jupyter port
EXPOSE 8888

# Default command to run when container starts
CMD ["python", "simulation_src/standard_run.py"]