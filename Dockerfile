# Use lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .

# Install minimal PyTorch and dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers datasets evaluate rouge_score numpy

# Ensure Metal backend for Apple Silicon
ENV PYTORCH_ENABLE_MPS=1
export MKL_SERVICE_FORCE_INTEL=1
CMD ["python", "GeneticAlgo.py"]
