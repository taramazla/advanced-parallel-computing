# Setup & Run LoRA Training on GPU Pod

## Quick Start

### 1. Deploy Pod dengan PyTorch 2.x
```bash
# Dari local machine
kubectl apply -f pod-gpu-lora.yaml

# Cek status pod
kubectl get pods | grep lora

# Tunggu sampai status Running
kubectl describe pod user02-gpu-lora-azzam
```

### 2. Masuk ke Pod
```bash
kubectl exec -it user02-gpu-lora-azzam /bin/bash
```

### 3. Setup Environment
```bash
cd /workspace/topik-4

# Jalankan setup script
bash setup_gpu.sh
```

### 4. Run Training
```bash
# Single training
python lora_finetuning.py \
    --num_epochs 1 \
    --batch_size 4 \
    --max_length 128

# Atau jalankan comparison experiments
bash compare_cpu_gpu.sh
```

---

## Using tmux (Recommended)

Agar training tetap berjalan meskipun connection terputus:

```bash
# Install tmux (jika belum ada)
apt-get update && apt-get install -y tmux

# Start tmux session
tmux new -s lora-training

# Run training
bash compare_cpu_gpu.sh

# Detach dari tmux: Ctrl+b lalu tekan d

# Re-attach ke session
tmux attach -t lora-training
```

---

## Monitoring Training

### Check GPU Usage
```bash
nvidia-smi

# Watch GPU usage real-time
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# Tail log file
tail -f cpu_gpu_comparison/comparison_*.log

# Tail training output
tail -f lora_outputs/*/logs/training.log
```

---

## Copy Results Back to Local

```bash
# Dari local machine
kubectl cp user02-gpu-lora-azzam:/workspace/topik-4/cpu_gpu_comparison ./results

# Atau via NFS (sudah otomatis tersync)
# Results ada di: /mnt/sharedfolder/user02/topik-4/cpu_gpu_comparison
```

---

## Troubleshooting

### Pod Tidak Start
```bash
kubectl describe pod user02-gpu-lora-azzam
kubectl logs user02-gpu-lora-azzam
```

### Out of Memory
Kurangi batch size:
```bash
python lora_finetuning.py --batch_size 2 --gradient_accumulation_steps 8
```

Atau gunakan quantization:
```bash
python lora_finetuning.py --use_4bit --batch_size 4
```

### CUDA Not Available
Pastikan pod request GPU:
```bash
kubectl describe pod user02-gpu-lora-azzam | grep nvidia.com/gpu
```

---

## Alternative: Menggunakan Docker Image Lain

Jika ingin image berbeda, edit `pod-gpu-lora.yaml`:

```yaml
# PyTorch 2.1 with CUDA 12.1
image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Atau TensorFlow dengan CUDA (jika butuh)
image: nvcr.io/nvidia/tensorflow:23.08-tf2-py3

# Atau custom image (jika ada)
image: your-registry/lora-training:latest
```

---

## Expected Performance

**GPT-2 (125M params) dengan batch_size=4:**
- GPU (RTX 3060/3070): ~20-30 it/s
- GPU (A100): ~50-80 it/s
- Training time (1 epoch): 3-5 minutes

**Dengan 4-bit quantization:**
- VRAM usage: ~3-4GB
- Speed: Sedikit lebih lambat (~15-20 it/s)
- Quality: Minimal degradation (<1%)
