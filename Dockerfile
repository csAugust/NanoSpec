# Use NVIDIA CUDA base image with compatible version
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    wget \
    vim \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate Python environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone the repository (recursively for submodules)
RUN git clone https://github.com/thunlp/FR-Spec.git --recursive /workspace/FR-Spec
WORKDIR /workspace/FR-Spec

# Install Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found, installing base packages"
RUN pip install setuptools wheel ninja fastchat

# Modify setup.py for CUDA architecture (default to Ampere A100, adjust as needed)
# RUN sed -i 's/arch="80"/arch="80"/' setup.py  # Keep 80 for A100 compatibility. Change if needed!

# Install FR-Spec package
# RUN pip install .

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data

# Set default command to show usage
CMD echo "FR-Spec container is ready. To run the example:"
CMD echo "1. Mount your model weights to /workspace/models"
CMD echo "2. Download FR-Spec vocabulary subset to /workspace/data"
CMD echo "3. Run: docker run --gpus all -it --rm -v /host/models:/workspace/models -v /host/data:/workspace/data <image> python examples/example_generate.py"
