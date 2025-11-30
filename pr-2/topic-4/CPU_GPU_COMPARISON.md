# CPU vs GPU Training Comparison Guide

## Overview
Script ini membandingkan performa training LoRA fine-tuning antara CPU dan GPU dengan berbagai konfigurasi.

## Quick Start

### 1. Run Comparison (Automatic)
```bash
# Make script executable
chmod +x compare_cpu_gpu.sh

# Run full comparison
bash compare_cpu_gpu.sh
```

Script akan otomatis:
- Deteksi ketersediaan GPU/CUDA
- Menjalankan eksperimen CPU dengan berbagai konfigurasi
- Menjalankan eksperimen GPU (jika tersedia)
- Generate comparison report

### 2. Manual CPU-only Testing
```bash
# Configuration 1: Small batch, high accumulation (memory efficient)
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 128 \
    --output_dir ./cpu_test_bs1

# Configuration 2: Balanced (recommended for CPU)
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 128 \
    --output_dir ./cpu_test_bs2

# Configuration 3: Larger batch (if RAM allows)
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 128 \
    --output_dir ./cpu_test_bs4
```

### 3. Manual GPU Testing (Requires CUDA)
```bash
# Configuration 1: Small batch (baseline)
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 128 \
    --output_dir ./gpu_test_bs2

# Configuration 2: Medium batch
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 128 \
    --output_dir ./gpu_test_bs4

# Configuration 3: Large batch
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_length 128 \
    --output_dir ./gpu_test_bs8

# Configuration 4: With 8-bit quantization
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --use_8bit \
    --max_length 128 \
    --output_dir ./gpu_test_8bit

# Configuration 5: With 4-bit quantization (QLoRA)
python lora_finetuning.py \
    --model_name gpt2 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --use_4bit \
    --max_length 128 \
    --output_dir ./gpu_test_4bit
```

## Experiment Configurations

### CPU Experiments

| Config | Batch Size | Grad Accum | Effective Batch | Memory Usage | Expected Time |
|--------|------------|------------|-----------------|--------------|---------------|
| CPU-1  | 1          | 8          | 8               | ~2 GB        | ~35-40 min    |
| CPU-2  | 2          | 4          | 8               | ~3 GB        | ~25-30 min    |
| CPU-3  | 4          | 2          | 8               | ~4 GB        | ~20-25 min    |

**Best for CPU:**
- Batch size: 2
- Gradient accumulation: 4
- Max length: 128-256
- Effective batch: 8

### GPU Experiments

| Config | Batch Size | Grad Accum | Effective Batch | VRAM Usage | Expected Time | Quantization |
|--------|------------|------------|-----------------|------------|---------------|--------------|
| GPU-1  | 2          | 4          | 8               | ~4 GB      | ~3-5 min      | None         |
| GPU-2  | 4          | 2          | 8               | ~6 GB      | ~2-3 min      | None         |
| GPU-3  | 8          | 1          | 8               | ~8 GB      | ~1.5-2 min    | None         |
| GPU-4  | 4          | 2          | 8               | ~4 GB      | ~2.5-3.5 min  | 8-bit        |
| GPU-5  | 4          | 2          | 8               | ~3 GB      | ~3-4 min      | 4-bit        |

**Best for GPU (8GB VRAM):**
- Batch size: 4-8
- Gradient accumulation: 1-2
- Max length: 256-512
- Effective batch: 8-16

**Best for GPU (4GB VRAM):**
- Batch size: 2-4
- Use 4-bit quantization
- Max length: 128-256
- Effective batch: 8

## Expected Results

### Performance Comparison (Estimated)

**GPT-2 (125M params), 1 epoch, ~24K samples:**

| Device | Config | Time | Speed (it/s) | Speedup |
|--------|--------|------|--------------|---------|
| CPU    | bs=2, ga=4 | ~25 min | ~2.4 | 1x (baseline) |
| GPU (RTX 3060) | bs=4, ga=2 | ~3 min | ~20 | ~8x |
| GPU (RTX 3080) | bs=8, ga=1 | ~2 min | ~30 | ~12x |
| GPU (A100) | bs=16, ga=1 | ~1 min | ~50 | ~25x |

### Memory Comparison

**GPT-2 Training:**

| Configuration | CPU RAM | GPU VRAM |
|---------------|---------|----------|
| Base (bs=2) | ~3 GB | ~4 GB |
| Medium (bs=4) | ~5 GB | ~6 GB |
| Large (bs=8) | ~8 GB | ~8 GB |
| 8-bit Quant | ~3 GB | ~3 GB |
| 4-bit Quant | ~3 GB | ~2 GB |

## Check Hardware Capabilities

### Check CUDA Availability
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Check GPU Memory
```bash
nvidia-smi
```

### Check CPU/RAM
```bash
# macOS
sysctl -n hw.physicalcpu hw.logicalcpu hw.memsize

# Linux
lscpu | grep -E '^CPU\(s\)|^Model name'
free -h
```

## Benchmarking Tips

### 1. Fair Comparison
- Use same dataset
- Same model configuration
- Same hyperparameters (except batch size/accumulation)
- Run multiple times and average
- Measure wall-clock time

### 2. What to Measure
- **Training time**: Total time for 1 epoch
- **Iterations/second**: Training speed
- **Memory usage**: Peak RAM/VRAM
- **Final loss**: Model quality
- **Throughput**: Samples processed per second

### 3. Document System Info
```bash
# Create system info file
echo "=== System Information ===" > system_info.txt
echo "Date: $(date)" >> system_info.txt
echo "" >> system_info.txt

echo "=== CPU ===" >> system_info.txt
python -c "import platform; print(f'Processor: {platform.processor()}')" >> system_info.txt

echo "" >> system_info.txt
echo "=== GPU ===" >> system_info.txt
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" >> system_info.txt

echo "" >> system_info.txt
echo "=== Python Packages ===" >> system_info.txt
pip list | grep -E "torch|transformers|peft" >> system_info.txt
```

## Analysis Commands

### Parse Training Logs
```bash
# Extract loss values
grep "loss" cpu_gpu_comparison/comparison_*.log | grep -oP "loss': \K[0-9.]+" > losses.txt

# Count iterations
grep "it/s" cpu_gpu_comparison/comparison_*.log | wc -l

# Find average speed
grep -oP "\K[0-9.]+(?=it/s)" cpu_gpu_comparison/comparison_*.log | awk '{sum+=$1; count++} END {print sum/count}'
```

### Compare Results
```bash
# View CSV results
cat cpu_gpu_comparison/timing_results.csv | column -t -s,

# Generate report
python generate_comparison_report.py \
    --results_csv cpu_gpu_comparison/timing_results.csv \
    --output_dir cpu_gpu_comparison
```

## Troubleshooting

### CPU Too Slow
```bash
# Reduce dataset size for testing
python lora_finetuning.py \
    --batch_size 2 \
    --max_length 64 \
    --num_epochs 1 \
    --save_steps 500

# Or prepare smaller dataset
python prepare_custom_dataset.py \
    --dataset_name bitext/Bitext-customer-support-llm-chatbot-training-dataset \
    --max_samples 1000 \
    --output_dir ./small_dataset
```

### GPU Out of Memory
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Reduce sequence length
--max_length 64

# Use quantization
--use_4bit

# Or all combined
python lora_finetuning.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 64 \
    --use_4bit
```

### No GPU Available
```bash
# Skip GPU experiments
# Only run CPU configurations
# Or use cloud GPU services:
# - Google Colab (Free T4 GPU)
# - Kaggle Notebooks (Free GPU)
# - Paperspace Gradient (Free/Paid)
```

## Cloud GPU Options

### Google Colab (Free)
```python
# Upload files to Colab
# Run in notebook:
!pip install -r requirements.txt
!python lora_finetuning.py --num_epochs 1 --batch_size 4
```

### Kaggle (Free)
```python
# Similar to Colab
# 30h/week GPU quota
!python lora_finetuning.py --num_epochs 1 --batch_size 4
```

### Local Setup Recommendations

**For Students/Testing:**
- CPU: Any modern 4+ core CPU
- RAM: 8GB minimum
- No GPU needed for small experiments

**For Development:**
- GPU: NVIDIA GTX 1660 Ti / RTX 3060 (6-8GB VRAM)
- RAM: 16GB
- Good balance of cost/performance

**For Production:**
- GPU: RTX 3080/3090 or A100 (10-24GB VRAM)
- RAM: 32GB+
- For serious training workloads

## Summary

**Key Takeaways:**
1. ✅ GPU provides 5-25x speedup over CPU
2. ✅ CPU viable for testing, not production
3. ✅ Quantization enables larger models on consumer GPUs
4. ✅ Gradient accumulation simulates larger batches
5. ✅ Monitor memory usage to optimize batch size

**Recommended Workflow:**
1. Test on CPU with small dataset
2. Validate on GPU with full dataset
3. Optimize batch size for your hardware
4. Use quantization for larger models
5. Document results for reproducibility
