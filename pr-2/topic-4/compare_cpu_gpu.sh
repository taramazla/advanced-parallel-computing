#!/bin/bash
# Script untuk membandingkan performa training di CPU vs GPU
# Menjalankan eksperimen dengan berbagai konfigurasi

set -e

# Create output directory
RESULTS_DIR="./cpu_gpu_comparison"
# Allow overriding Python interpreter via env; default to WSL venv if present
PYTHON="${PYTHON:-$HOME/lora-venv/bin/python}"
mkdir -p "$RESULTS_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMPARISON_LOG="$RESULTS_DIR/comparison_${TIMESTAMP}.log"

echo "========================================" | tee -a "$COMPARISON_LOG"
echo "CPU vs GPU Training Comparison" | tee -a "$COMPARISON_LOG"
echo "Started at: $(date)" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"
echo "" | tee -a "$COMPARISON_LOG"

# Function to check if CUDA is available
check_cuda() {
    python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null
}

echo "Checking CUDA availability..." | tee -a "$COMPARISON_LOG"
check_cuda | tee -a "$COMPARISON_LOG"
echo "" | tee -a "$COMPARISON_LOG"

# Common parameters
MODEL_NAME="gpt2"
DATASET_NAME="bitext/Bitext-customer-support-llm-chatbot-training-dataset"
NUM_EPOCHS=1
MAX_LENGTH=128
LORA_R=8
LORA_ALPHA=16

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local device=$2
    local batch_size=$3
    local grad_accum=$4
    local use_quantization=$5

    echo "----------------------------------------" | tee -a "$COMPARISON_LOG"
    echo "Experiment: $exp_name" | tee -a "$COMPARISON_LOG"
    echo "Device: $device" | tee -a "$COMPARISON_LOG"
    echo "Batch size: $batch_size" | tee -a "$COMPARISON_LOG"
    echo "Gradient accumulation: $grad_accum" | tee -a "$COMPARISON_LOG"
    echo "Quantization: $use_quantization" | tee -a "$COMPARISON_LOG"
    echo "----------------------------------------" | tee -a "$COMPARISON_LOG"

    output_dir="$RESULTS_DIR/$exp_name"

    # Build command
    cmd="$PYTHON lora_finetuning.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --output_dir $output_dir \
        --num_epochs $NUM_EPOCHS \
        --batch_size $batch_size \
        --gradient_accumulation_steps $grad_accum \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --max_length $MAX_LENGTH \
        --learning_rate 2e-4 \
        --save_steps 1000 \
        --logging_steps 100"

    # Add quantization flag if requested
    if [ "$use_quantization" == "4bit" ]; then
        cmd="$cmd --use_4bit"
    elif [ "$use_quantization" == "8bit" ]; then
        cmd="$cmd --use_8bit"
    fi

    # Run experiment
    echo "Starting training..." | tee -a "$COMPARISON_LOG"
    start_time=$(date +%s)

    if eval $cmd >> "$COMPARISON_LOG" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Experiment $exp_name completed in ${duration}s" | tee -a "$COMPARISON_LOG"

        # Save duration to file
        echo "$exp_name,$device,$batch_size,$grad_accum,$use_quantization,$duration" >> "$RESULTS_DIR/timing_results.csv"
    else
        echo "✗ Experiment $exp_name failed" | tee -a "$COMPARISON_LOG"
    fi

    echo "" | tee -a "$COMPARISON_LOG"
}

# Initialize CSV
echo "experiment,device,batch_size,grad_accum,quantization,duration_seconds" > "$RESULTS_DIR/timing_results.csv"

# ========================================
# CPU Experiments
# ========================================
echo "========================================" | tee -a "$COMPARISON_LOG"
echo "CPU EXPERIMENTS" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"
echo "" | tee -a "$COMPARISON_LOG"

# CPU Experiment 1: Small batch size
run_experiment "cpu_bs2_ga4" "CPU" 2 4 "none"

# CPU Experiment 2: Very small batch (memory constrained)
run_experiment "cpu_bs1_ga8" "CPU" 1 8 "none"

# CPU Experiment 3: Larger batch if memory allows
run_experiment "cpu_bs4_ga2" "CPU" 4 2 "none"

# ========================================
# GPU Experiments (if available)
# ========================================
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "========================================" | tee -a "$COMPARISON_LOG"
    echo "GPU EXPERIMENTS" | tee -a "$COMPARISON_LOG"
    echo "========================================" | tee -a "$COMPARISON_LOG"
    echo "" | tee -a "$COMPARISON_LOG"

    # GPU Experiment 1: Small batch (comparable to CPU)
    run_experiment "gpu_bs2_ga4" "GPU" 2 4 "none"

    # GPU Experiment 2: Medium batch
    run_experiment "gpu_bs4_ga2" "GPU" 4 2 "none"

    # GPU Experiment 3: Larger batch
    run_experiment "gpu_bs8_ga1" "GPU" 8 1 "none"

    # GPU Experiment 4: With 8-bit quantization
    run_experiment "gpu_bs4_ga2_8bit" "GPU" 4 2 "8bit"

    # GPU Experiment 5: With 4-bit quantization (QLoRA)
    run_experiment "gpu_bs4_ga2_4bit" "GPU" 4 2 "4bit"

else
    echo "GPU not available. Skipping GPU experiments." | tee -a "$COMPARISON_LOG"
fi

# ========================================
# Generate Comparison Report
# ========================================
echo "" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"
echo "GENERATING COMPARISON REPORT" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"

python generate_comparison_report.py \
    --results_csv "$RESULTS_DIR/timing_results.csv" \
    --output_dir "$RESULTS_DIR" \
    --log_file "$COMPARISON_LOG"

echo "" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"
echo "Comparison completed at: $(date)" | tee -a "$COMPARISON_LOG"
echo "Results saved to: $RESULTS_DIR" | tee -a "$COMPARISON_LOG"
echo "========================================" | tee -a "$COMPARISON_LOG"
