# Use the official NVIDIA CUDA image as base.
# 12.4.1 is specified. Using the 'devel' image includes headers and libraries
# needed for compiling extensions like xformers or triton. Ubuntu 22.04 is a solid LTS.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables non-interactively during build
ENV DEBIAN_FRONTEND=noninteractive

# Explicitly set CUDA paths, although the base image should handle this.
# It's good practice for clarity and ensures tools find the CUDA toolkit.
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HF_HOME=/app/models

# Update package lists and install essential build tools and Python.
# - build-essential: Needed for compiling code (like Python extensions).
# - python3, python3-pip: Base Python environment.
# - git: For potentially fetching code or dependencies.
# - curl, wget, ca-certificates: Standard utilities for downloading.
# - ninja-build: Often speeds up compilation of C++/CUDA extensions (like xformers).
# Using --no-install-recommends keeps the image smaller.
# Clean up apt cache afterwards to reduce layer size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        git \
        curl \
        wget \
        ca-certificates \
        ninja-build \
        software-properties-common \
        ubuntu-drivers-common \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:apt-fast/stable && apt-get update && apt-get -y install apt-fast

# Install 'uv' - the fast Python package installer and resolver.
# We'll use pip to install it initially.
RUN pip3 install --no-cache-dir uv

# Set the working directory for the application.
WORKDIR /app

# Copy the Python project definition and lock file.
# Copying these *before* running install leverages Docker caching.
# If only app code changes later, dependency installation won't rerun.
COPY pyproject.toml uv.lock .python-version ./
COPY .env ./

# Install Python dependencies using uv.
# This should install PyTorch, diffusers, stable-fast, and other core libraries
# as defined in pyproject.toml/uv.lock.
# Using uv sync is generally much faster than pip install -r requirements.txt.
# We use --no-cache to keep the image size down during build, similar to pip's --no-cache-dir.
RUN uv sync --no-cache

# Install acceleration libraries explicitly.
# While ideally these are in pyproject.toml, installing them separately
# can sometimes simplify debugging version compatibility issues between
# torch, cuda, and these libraries.
# They need PyTorch (installed via uv sync above) to be present.
# Ensure versions in pyproject.toml (for torch) are compatible with CUDA 12.4 and these libs.
# Note: Triton installation via pip might only work reliably on certain Linux dists/Python versions.
# Check the official Triton docs if issues arise. Using --no-cache-dir for pip.
RUN uv pip install --no-cache \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
# RUN uv pip install --no-cache-dir stable-fast
RUN uv pip install --no-cache-dir triton

# Huggingface accelerate
RUN uv run accelerate config default

# Define default command (example - replace with your actual app entrypoint)
# CMD ["python3", "app.py"]

# Expose any necessary ports (if your application is a web service, etc.)
EXPOSE 10295

# Final check for CUDA - this command should succeed if CUDA is set up correctly.
# RUN nvidia-smi
RUN apt-get update && apt-get install -y 

# RUN ubuntu-drivers devices && apt-fast install -y nvidia-570 && modprobe nvidia
# RUN nvidia-smi
RUN which nvcc

# copy the project files
COPY .docker_env .env
COPY static static
COPY main.py main.py
COPY merlin.py merlin.py
COPY mirror_ai mirror_ai


# start the application
CMD ["uv", "run", "python", "main.py"]