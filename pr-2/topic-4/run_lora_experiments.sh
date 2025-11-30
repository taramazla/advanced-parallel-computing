#!/bin/bash
# Script untuk menjalankan eksperimen LoRA dengan berbagai konfigurasi
# Run multiple LoRA fine-tuning experiments with different hyperparameters

set -e

# Configuration
MODEL_NAME="gpt2"
DATASET_NAME="bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_CONFIG=""
BASE_OUTPUT_DIR="./lora_experiments"
NUM_EPOCHS=3
BATCH_SIZE=4
MAX_LENGTH=512

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT_DIR/logs"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$BASE_OUTPUT_DIR/logs/experiments_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "LoRA Fine-tuning Experiments" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local lora_r=$2
    local lora_alpha=$3
    local lora_dropout=$4
    local learning_rate=$5
    local use_quantization=$6
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Running experiment: $exp_name" | tee -a "$LOG_FILE"
    echo "Parameters:" | tee -a "$LOG_FILE"
    echo "  LoRA r: $lora_r" | tee -a "$LOG_FILE"
    echo "  LoRA alpha: $lora_alpha" | tee -a "$LOG_FILE"
    echo "  LoRA dropout: $lora_dropout" | tee -a "$LOG_FILE"
    echo "  Learning rate: $learning_rate" | tee -a "$LOG_FILE"
    echo "  Quantization: $use_quantization" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    output_dir="$BASE_OUTPUT_DIR/$exp_name"
    
    # Build command
    cmd="python lora_finetuning.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --dataset_config $DATASET_CONFIG \
        --output_dir $output_dir \
        --lora_r $lora_r \
        --lora_alpha $lora_alpha \
        --lora_dropout $lora_dropout \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $learning_rate \
        --max_length $MAX_LENGTH"
    
    # Add quantization flag if requested
    if [ "$use_quantization" != "none" ]; then
        cmd="$cmd --$use_quantization"
    fi
    
    # Run experiment
    echo "Starting training..." | tee -a "$LOG_FILE"
    start_time=$(date +%s)
    
    if eval $cmd >> "$LOG_FILE" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ Experiment $exp_name completed successfully in ${duration}s" | tee -a "$LOG_FILE"
    else
        echo "✗ Experiment $exp_name failed" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
}

# Experiment 1: Baseline (low rank)
run_experiment "exp01_baseline_r4" 4 8 0.1 2e-4 "none"

# Experiment 2: Medium rank
run_experiment "exp02_medium_r8" 8 16 0.1 2e-4 "none"

# Experiment 3: Higher rank
run_experiment "exp03_high_r16" 16 32 0.1 2e-4 "none"

# Experiment 4: Very high rank
run_experiment "exp04_veryhigh_r32" 32 64 0.1 2e-4 "none"

# Experiment 5: Lower learning rate
run_experiment "exp05_low_lr" 8 16 0.1 1e-4 "none"

# Experiment 6: Higher learning rate
run_experiment "exp06_high_lr" 8 16 0.1 3e-4 "none"

# Experiment 7: Lower dropout
run_experiment "exp07_low_dropout" 8 16 0.05 2e-4 "none"

# Experiment 8: Higher dropout
run_experiment "exp08_high_dropout" 8 16 0.2 2e-4 "none"

# Experiment 9: With 8-bit quantization (if available)
# Uncomment if you have bitsandbytes installed
# run_experiment "exp09_8bit" 8 16 0.1 2e-4 "use_8bit"

# Experiment 10: With 4-bit quantization (if available)
# Uncomment if you have bitsandbytes installed
# run_experiment "exp10_4bit" 8 16 0.1 2e-4 "use_4bit"

echo "========================================" | tee -a "$LOG_FILE"
echo "All experiments completed!" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Generate summary report
python generate_experiment_report.py \
    --experiments_dir "$BASE_OUTPUT_DIR" \
    --output_file "$BASE_OUTPUT_DIR/experiment_summary.txt"

echo "" | tee -a "$LOG_FILE"
echo "Summary report generated at: $BASE_OUTPUT_DIR/experiment_summary.txt" | tee -a "$LOG_FILE"
echo "Full log available at: $LOG_FILE" | tee -a "$LOG_FILE"
