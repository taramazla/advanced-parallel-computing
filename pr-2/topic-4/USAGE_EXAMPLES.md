# Usage Examples - LoRA Fine-tuning with E-commerce Dataset

## Dataset Loading dari HuggingFace (Recommended)

Dataset `qgyd2021/e_commerce_customer_service` akan otomatis didownload dari HuggingFace Hub. **Tidak perlu file lokal!**

### 1. Basic Fine-tuning (Paling Sederhana)

```bash
# GPT-2 dengan default settings
python lora_finetuning.py
```

Dataset default sudah di-set ke `qgyd2021/e_commerce_customer_service` dengan config `amazon`.

### 2. Fine-tuning dengan Custom Parameters

```bash
python lora_finetuning.py \
    --model_name gpt2 \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./my_model \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_length 512
```

### 3. Quick Test untuk Validasi Setup

```bash
bash quick_test.sh
```

Ini akan:
1. Install dependencies
2. Download dataset dari HuggingFace (100 samples)
3. Fine-tune GPT-2 dengan 1 epoch
4. Test inference dengan query customer service

### 4. Gunakan Dataset Config Berbeda

```bash
# Amazon customer service
python lora_finetuning.py --dataset_config amazon

# E-commerce FAQ
python lora_finetuning.py --dataset_config ecommerce_faq

# Shopee (jika tersedia)
python lora_finetuning.py --dataset_config shopee
```

### 5. Model Lebih Besar dengan QLoRA (4-bit quantization)

```bash
python lora_finetuning.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --use_4bit \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4
```

### 6. Model Mistral-7B

```bash
python lora_finetuning.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --use_4bit \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64
```

## Prepare Dataset (Optional - untuk Caching)

Jika ingin download dan cache dataset terlebih dahulu:

### 1. Download Full Dataset

```bash
python prepare_custom_dataset.py \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./ecommerce_dataset
```

### 2. Download dengan Limit Samples

```bash
python prepare_custom_dataset.py \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./ecommerce_small \
    --max_samples 1000
```

### 3. Custom Train/Val Split

```bash
python prepare_custom_dataset.py \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./ecommerce_dataset \
    --train_split 0.85
```

## Menggunakan Local Files (Optional)

Jika ingin menggunakan file lokal sebagai gantinya:

### 1. Format Instruction (JSON)

```bash
python prepare_custom_dataset.py \
    --input_file sample_ecommerce_instructions.json \
    --format instruction \
    --output_dir ./custom_dataset \
    --train_split 0.9
```

### 2. Format JSON Biasa

```bash
python prepare_custom_dataset.py \
    --input_file my_data.json \
    --format json \
    --text_column text \
    --output_dir ./my_dataset
```

### 3. Format CSV

```bash
python prepare_custom_dataset.py \
    --input_file data.csv \
    --format csv \
    --text_column customer_question \
    --output_dir ./csv_dataset
```

## Inference / Testing Model

### 1. Single Prompt

```bash
python inference.py \
    --model_path ./lora_outputs/gpt2_lora_20241201_123456/final \
    --prompt "How do I track my order?" \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --do_sample
```

### 2. Interactive Mode

```bash
python inference.py \
    --model_path ./lora_outputs/gpt2_lora_20241201_123456/final \
    --interactive
```

### 3. Batch Testing dengan Multiple Prompts

```bash
# Test berbagai customer service queries
python inference.py \
    --model_path ./lora_outputs/gpt2_lora_20241201_123456/final \
    --prompt "What is your return policy?" \
    --max_new_tokens 200

python inference.py \
    --model_path ./lora_outputs/gpt2_lora_20241201_123456/final \
    --prompt "How long does shipping take?" \
    --max_new_tokens 200

python inference.py \
    --model_path ./lora_outputs/gpt2_lora_20241201_123456/final \
    --prompt "Can I cancel my order?" \
    --max_new_tokens 200
```

## Running Experiments

### 1. Quick Experiments (Default)

```bash
bash run_lora_experiments.sh
```

Ini akan menjalankan 10+ eksperimen dengan berbagai konfigurasi LoRA.

### 2. Custom Experiment Script

Buat file `my_experiments.sh`:

```bash
#!/bin/bash

# Experiment 1: Small model
python lora_finetuning.py \
    --lora_r 8 --lora_alpha 16 \
    --output_dir ./exp1_r8

# Experiment 2: Larger rank
python lora_finetuning.py \
    --lora_r 32 --lora_alpha 64 \
    --output_dir ./exp2_r32

# Experiment 3: Different learning rate
python lora_finetuning.py \
    --lora_r 16 --lora_alpha 32 \
    --learning_rate 1e-4 \
    --output_dir ./exp3_lr1e4
```

## Generate Experiment Report

```bash
python generate_experiment_report.py \
    --experiments_dir ./lora_experiments \
    --output_file experiment_results.md
```

## Tips & Recommendations

### Memory Constraints

| GPU Memory | Recommended Model | Batch Size | Quantization |
|------------|------------------|------------|--------------|
| 4-8 GB     | GPT-2 / GPT-2 Medium | 2-4 | None or 8-bit |
| 8-16 GB    | GPT-2 Large / Small LLaMA | 2-4 | 4-bit (QLoRA) |
| 16-24 GB   | LLaMA-7B / Mistral-7B | 2-4 | 4-bit |
| 24+ GB     | LLaMA-7B / Mistral-7B | 4-8 | None or 4-bit |

### Parameter Recommendations

**For E-commerce Customer Service:**
- LoRA rank (r): 16-32
- LoRA alpha: 32-64
- Learning rate: 1e-4 to 2e-4
- Epochs: 3-5
- Max length: 512-1024

**Quick Testing:**
- LoRA rank (r): 4-8
- Epochs: 1
- Max samples: 100-500
- Batch size: 2

**Production:**
- LoRA rank (r): 32-64
- Epochs: 5-10
- Full dataset
- Multiple experiments to find best config

## Troubleshooting

### Dataset tidak terdownload
```bash
# Manual check
python -c "from datasets import load_dataset; ds = load_dataset('qgyd2021/e_commerce_customer_service', 'amazon'); print(ds)"
```

### CUDA out of memory
- Reduce batch size: `--batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 16`
- Use 4-bit quantization: `--use_4bit`
- Reduce max_length: `--max_length 256`

### Slow training
- Increase batch size (if memory allows)
- Reduce gradient accumulation steps
- Use smaller max_length
- Use fewer epochs
