# =============================================================================
# LLM-Speed Development Container
# Multi-stage build for CUDA development environment
# =============================================================================

# Stage 1: Builder
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source files
COPY CMakeLists.txt CMakePresets.json ./
COPY src ./src
COPY include ./include
COPY cuda_llm_ops ./cuda_llm_ops

# Build CUDA kernels
RUN cmake --preset default && cmake --build build --parallel $(nproc)

# =============================================================================
# Stage 2: Runtime
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy Python package files
COPY pyproject.toml setup.py requirements.txt ./
COPY cuda_llm_ops ./cuda_llm_ops

# Copy built CUDA libraries from builder
COPY --from=builder /build/build/*.so /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install --no-cache-dir -e .

# Copy test and benchmark files
COPY tests ./tests
COPY benchmarks ./benchmarks

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "-c", "import cuda_llm_ops; print(f'LLM-Speed v{cuda_llm_ops.__version__} loaded successfully')"]
