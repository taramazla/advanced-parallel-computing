# Fine-tuning LLM with LoRA (Low-Rank Adaptation)

Implementasi fine-tuning Large Language Models (LLM) menggunakan metode LoRA yang efisien untuk parallel computing. LoRA memungkinkan fine-tuning model besar dengan resource minimal dengan hanya melatih parameter tambahan yang jumlahnya jauh lebih sedikit.

Dataset yang digunakan: **qgyd2021/e_commerce_customer_service** - dataset customer service e-commerce untuk melatih model menjawab pertanyaan pelanggan.

## 📋 Fitur Utama

- **LoRA Fine-tuning**: Fine-tune model LLM dengan parameter efficient
- **QLoRA Support**: Quantization 4-bit/8-bit untuk efisiensi memory
- **Multiple Models**: Support GPT-2, LLaMA, Mistral, dan model lainnya
- **E-commerce Dataset**: Pre-configured dengan dataset customer service
- **Custom Datasets**: Tools untuk prepare dataset custom
- **Inference Tools**: Script untuk testing dan interactive chat
- **Batch Processing**: Eksperimen dengan berbagai konfigurasi

## 🛠️ Instalasi

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (untuk GPU)
- 8GB+ GPU memory (16GB+ recommended untuk model besar)

### Install Dependencies

```bash
apt-get update
apt-get install -y git python3-venv
apt-get install -y python3-pip
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

tmux new -s nama_sesi
tmux ls
tmux attach -t nama_sesi
```

Untuk QLoRA (4-bit quantization), pastikan CUDA tersedia:
```bash
pip install bitsandbytes
```

## 🚀 Quick Start

### 1. Fine-tuning dengan E-commerce Customer Service Dataset

Fine-tune GPT-2 untuk menjawab pertanyaan customer service:

```bash
python lora_finetuning.py \
    --model_name gpt2 \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./outputs \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

Dataset configs yang tersedia:
- `amazon` - Amazon customer service
- `ecommerce_faq` - E-commerce FAQs
- `shopee` - Shopee customer service

### 2. Fine-tuning dengan QLoRA (4-bit)

Untuk model besar dengan memory terbatas:

```bash
python lora_finetuning.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --use_4bit \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32
```

### 3. Prepare Custom Dataset

#### From HuggingFace (Recommended)
```bash
# E-commerce customer service dataset (default)
python prepare_custom_dataset.py \
    --dataset_name qgyd2021/e_commerce_customer_service \
    --dataset_config amazon \
    --output_dir ./ecommerce_dataset

# Other HuggingFace datasets
python prepare_custom_dataset.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./wiki_dataset
```

#### From Local Files (Optional)
```bash
# Format JSON
python prepare_custom_dataset.py \
    --input_file data.json \
    --output_dir ./custom_dataset \
    --format json \
    --text_column text

# Format Instruction-tuning
python prepare_custom_dataset.py \
    --input_file instructions.json \
    --output_dir ./instruction_dataset \
    --format instruction
```

### 4. Inference

#### Single Prompt
```bash
python inference.py \
    --model_path ./outputs/final \
    --prompt "How can I track my order?" \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --do_sample
```

#### Interactive Mode
```bash
python inference.py \
    --model_path ./outputs/final \
    --interactive
```

## 📊 Parameter LoRA

### Parameter Utama

- **lora_r**: Rank matrix LoRA (8, 16, 32, 64)
  - Lebih besar = lebih banyak parameter, training lebih lambat tapi potensial lebih baik
  - Rekomendasi: 8-16 untuk eksperimen, 32-64 untuk production

- **lora_alpha**: Scaling factor (biasanya 2x dari lora_r)
  - Mengontrol learning rate efektif untuk LoRA weights
  - Rekomendasi: 16-32

- **lora_dropout**: Dropout rate (0.0-0.2)
  - Mencegah overfitting
  - Rekomendasi: 0.05-0.1

### Target Modules

LoRA biasanya diterapkan pada attention layers:
- GPT models: `["q_proj", "v_proj"]` atau `["c_attn"]`
- LLaMA models: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Mistral models: `["q_proj", "k_proj", "v_proj", "o_proj"]`

## 📁 Struktur Dataset

### HuggingFace Dataset (Recommended)
Dataset langsung diload dari HuggingFace Hub. Tidak perlu download manual.

**E-commerce Customer Service Dataset:**
- Dataset: `qgyd2021/e_commerce_customer_service`
- Configs: `amazon`, `ecommerce_faq`, `shopee`
- Format: Question-Answer pairs untuk customer service

**Cara menggunakan:**
```python
from datasets import load_dataset
dataset = load_dataset("qgyd2021/e_commerce_customer_service", "amazon")
```

### Format Local Files (Optional)

#### Format JSON
```json
[
    {"text": "Sample text 1..."},
    {"text": "Sample text 2..."}
]
```

#### Format Instruction
```json
[
    {
        "instruction": "How do I track my order?",
        "input": "",
        "output": "You can track your order by..."
    }
]
```

## 🎯 Use Cases

### 1. Text Generation
Fine-tune untuk generasi teks kreatif, artikel, atau konten.

### 2. Instruction Following
Train model untuk mengikuti instruksi spesifik.

### 3. Domain Adaptation
Adaptasi model ke domain spesifik (medis, hukum, teknis).

### 4. Style Transfer
Train model untuk menulis dengan gaya tertentu.

## 📈 Tips & Best Practices

### Memory Optimization
1. **Use 4-bit quantization** untuk model >7B parameters
2. **Gradient checkpointing** untuk menghemat memory
3. **Batch size kecil** + gradient accumulation
4. **Lower precision** (fp16/bf16)

### Training Tips
1. **Start small**: Test dengan GPT-2 atau model kecil dulu
2. **Monitor loss**: Validation loss harus turun
3. **Learning rate**: 1e-4 sampai 3e-4 untuk LoRA
4. **Warmup steps**: 100-500 steps
5. **Save checkpoints**: Simpan setiap 500-1000 steps

### Quality Improvement
1. **Increase lora_r**: Untuk task kompleks
2. **More data**: Dataset lebih besar = hasil lebih baik
3. **Longer training**: 3-5 epochs biasanya cukup
4. **Data quality**: Clean, relevant data penting

## 🔬 Eksperimen

### Variasi Parameter LoRA
```bash
# Eksperimen 1: Low-rank (fast, less parameters)
python lora_finetuning.py --lora_r 4 --lora_alpha 8

# Eksperimen 2: Medium-rank (balanced)
python lora_finetuning.py --lora_r 16 --lora_alpha 32

# Eksperimen 3: High-rank (slow, more parameters)
python lora_finetuning.py --lora_r 64 --lora_alpha 128
```

### Eksperimen Script
```bash
# Run multiple experiments with different configs
bash run_lora_experiments.sh
```

## 📊 Monitoring Training

### Loss Tracking
Training metrics disimpan di `<output_dir>/training_metrics.txt`

### GPU Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Solusi:
--use_4bit  # Enable 4-bit quantization
--batch_size 1  # Reduce batch size
--gradient_accumulation_steps 16  # Increase accumulation
--max_length 512  # Reduce sequence length
```

### Slow Training
```bash
# Optimasi:
--batch_size 8  # Increase if memory allows
--gradient_accumulation_steps 2  # Reduce if possible
--max_length 256  # Reduce for shorter texts
```

### Poor Quality Output
```bash
# Perbaikan:
--lora_r 32  # Increase rank
--num_epochs 5  # Train longer
# Check data quality
# Increase dataset size
```

## 📚 Referensi

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **PEFT Library**: [Hugging Face PEFT](https://github.com/huggingface/peft)
- **Transformers**: [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 🎓 Konsep LoRA

LoRA bekerja dengan menambahkan low-rank matrices ke layer transformer:

```
W = W₀ + ΔW = W₀ + BA
```

Dimana:
- W₀: Pretrained weights (frozen)
- B, A: Trainable low-rank matrices
- rank(BA) << rank(W₀)

Keuntungan:
- Parameter trainable << pretrained parameters
- Memory efficient
- Faster training
- Easy to switch/merge adapters

## 📊 Hasil Eksperimen

Setelah training, cek hasil di:
- `<output_dir>/training_metrics.txt`: Training logs
- `<output_dir>/final/`: Final model weights
- `<output_dir>/checkpoint-*/`: Intermediate checkpoints

## 🤝 Contributing

Untuk kontribusi atau pertanyaan, silakan buat issue atau pull request.

## 📝 License

MIT License - lihat LICENSE file untuk detail.

---

**Happy Fine-tuning! 🚀**
