#!/bin/bash
# Quick test script untuk menguji setup LoRA fine-tuning
# Menggunakan GPT-2 dan sample dataset untuk testing cepat

set -e

echo "========================================="
echo "LoRA Fine-tuning Quick Test"
echo "========================================="
echo ""

# Check Python
echo "Checking Python installation..."
python --version
echo ""

# Check GPU availability (optional)
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed yet"
echo ""

# Install requirements
echo "Installing requirements..."
pip install -q transformers peft datasets accelerate torch
echo "✓ Requirements installed"
echo ""

# Optional: Prepare dataset locally for faster loading
echo "Preparing dataset (optional - caching HuggingFace dataset)..."
python prepare_custom_dataset.py \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --output_dir ./test_dataset \
    --max_samples 100
echo "✓ Dataset prepared"
echo ""

# Run quick fine-tuning test (very small model, few steps)
echo "Running quick fine-tuning test with GPT-2..."
echo "Dataset: Customer Support Chatbot Training"
echo "This will take a few minutes..."
python lora_finetuning.py \
    --model_name gpt2 \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --output_dir ./test_output \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --num_epochs 1 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_length 128 \
    --gradient_accumulation_steps 2 \
    --save_steps 50 \
    --logging_steps 10

echo ""
echo "✓ Fine-tuning test completed"
echo ""

# Test inference
echo "Testing inference with customer service query..."
python inference.py \
    --model_path ./test_output/final \
    --prompt "How can I track my order?" \
    --max_new_tokens 50 \
    --temperature 0.7 \
    --do_sample

echo ""
echo "========================================="
echo "Quick test completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Check output in ./test_output/training_metrics.txt"
echo "2. Try interactive mode: python inference.py --model_path ./test_output/final --interactive"
echo "3. Run full experiments: bash run_lora_experiments.sh"
echo ""
