#!/bin/bash
# Setup script for LoRA training on GPU pod

set -e

echo "=========================================="
echo "Setup LoRA Training Environment"
echo "=========================================="
echo ""

# Check Python version
echo "Python version:"
python --version
echo ""

# Check PyTorch & CUDA
echo "Checking PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-gpu.txt
echo ""

# Verify installations
echo "Verifying installations..."
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
echo ""

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To start training:"
echo "python lora_finetuning.py --num_epochs 1 --batch_size 4"
echo ""
echo "To run comparison experiments:"
echo "bash compare_cpu_gpu.sh"
