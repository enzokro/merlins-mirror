# Use the official NVIDIA CUDA image as base.
# 12.4.1 is specified. Using the 'devel' image includes headers and libraries
# needed for compiling extensions like xformers or triton. Ubuntu 22.04 is a solid LTS.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables non-interactively during build
ENV DEBIAN_FRONTEND=noninteractive

# Explicitly set CUDA paths, although the base image should handle this.
# It's good practice for clarity and ensures tools find the CUDA toolkit.
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    CUDA_DOCKER_ARCH=all \
    CUDAVER=12.4.1
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
        cmake \
        gcc \
        g++ \
        tar \ 
        ninja-build \
        python3 \
        python3-pip \
        git \
        curl \
        wget \
        ca-certificates \
        software-properties-common \
        libgl1-mesa-dev \ 
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' - the fast Python package installer and resolver.
# We'll use pip to install it initially.
RUN pip3 install --no-cache-dir uv

# Set the working directory for the application.
WORKDIR /app

# Copy the Python project definition and lock file.
# Copying these *before* running install leverages Docker caching.
# If only app code changes later, dependency installation won't rerun.
COPY pyproject.toml uv.lock .python-version ./

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
RUN uv pip install --no-cache-dir tensorrt
RUN uv pip install --no-cache-dir torch-tensorrt

# Huggingface accelerate
RUN uv run accelerate config default --mixed_precision fp16

# Define default command (example - replace with your actual app entrypoint)
# CMD ["python3", "app.py"]

# Expose any necessary ports (if your application is a web service, etc.)
EXPOSE 10295

# Sanity check the paths
RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-12.4" >> /etc/bash.bashrc

# Final check for CUDA - this command should succeed if CUDA is set up correctly.
RUN which nvcc

# install cv2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install the nightly diffusers version
RUN uv pip install git+https://github.com/huggingface/diffusers

# # install xformers
# RUN uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

# patch in onnxruntime-gpu
RUN uv pip install onnx
RUN uv pip install onnxruntime-gpu
RUN uv pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

# patch in polygraphy
RUN uv pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com

# install cuda-python
RUN uv pip install cuda-python

# # install streamdiffusion
# RUN uv pip install "git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]"
# RUN uv run python -m streamdiffusion.tools.install-tensorrt

# copy the project files, put here for cache purposes
COPY .docker_env .env
COPY static static
COPY main.py main.py
COPY merlin.py merlin.py
COPY mirror_ai mirror_ai
COPY run_compile.py run_compile.py
COPY load_compiled.py load_compiled.py

# # start the application
# CMD ["uv", "run", "python", "main.py"]

# # compile the models
CMD ["uv", "run", "python", "run_compile.py"]

# test load a compiled model
# CMD ["uv", "run", "python", "load_compiled.py"]